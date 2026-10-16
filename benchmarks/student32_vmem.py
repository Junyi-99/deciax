"""Experimental entire BF16 student in a single VMEM-resident Pallas call.

One row and two options. All parameters including embedding and decision heads
are resident together. Each call loads weights once; repetitions reuse them
within that call (not across calls). Compare against the unfused JAX model.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from deciax import nn
from deciax.kernels import fused as f
from deciax.kernels.attention.pallas import head_cols, mask_bias, pv_t, scores_t
from deciax.models.laya.model import LayaModel
from deciax.protocol.types import parse_questions
from deciax.runtime.buckets import collate


def prepare(model, seq):
    p = model.params
    data = {}

    def add(name, value):
        data[name] = value.reshape(1, -1) if value.ndim == 1 else value

    # Keep every model parameter, replacing matrices with compute layouts.
    for k, v in p.items():
        if v.ndim == 2 and k.endswith("weight") and k not in ("type_emb.weight", "encoder.embeddings.tok_embeddings.weight"):
            v = v.T
        add(k, v)
    for i in range(10):
        key = f"encoder.layers.{i}"
        wi, wo = f.pad_gated(p[key + ".mlp.Wi.weight"], p[key + ".mlp.Wo.weight"])
        data[key + ".mlp.Wi.weight"] = wi
        data[key + ".mlp.Wo.weight"] = wo
    for i in range(2):
        key = f"head.layers.{i}.self_attn.in_proj_weight"
        data[key] = p[key].T
    # Padding for TPU lanes, preserving the true 388 input channels of act_head.
    data["act_head.0.weight"] = jnp.pad(p["act_head.0.weight"].T, ((0, 124), (0, 0)))
    for key in ("scorer.3", "act_head.2"):
        v = data[key + ".weight"]
        data[key + ".weight"] = jnp.pad(v, ((0, 0), (0, 128 - v.shape[1])))
        v = data[key + ".bias"]
        data[key + ".bias"] = jnp.pad(v, ((0, 0), (0, 128 - v.shape[1])))
    rope = f.rope_tables(nn.rope_inv_freq(64, 160000), seq)
    data["cos"], data["sin"] = rope.cos, rope.sin
    return data


def build(data, seq, repeats, vmem_mib, *, stream=False, embed_group=1, head_group=1):
    assert embed_group in (1, 8) and head_group in (1, 2, 3, 6)
    names = tuple(data)

    def kernel(meta, mask, *refs):
        p = dict(zip(names, refs[: len(names)]))
        out, xbuf, qkv, att, score = refs[len(names) :]

        def val(k):
            return p[k][...]

        def norm(x, k):
            return f._layer_norm(
                x, val(k + ".weight").astype(jnp.float32), val(k + ".bias").astype(jnp.float32) if k + ".bias" in p else jnp.zeros((1, 384), jnp.float32), 1e-5
            )

        def linear(x, k):
            w = val(k + ".weight")
            y = jnp.dot(x.astype(w.dtype), w, preferred_element_type=jnp.float32)
            return y + val(k + ".bias").astype(jnp.float32) if k + ".bias" in p else y

        @jax.named_scope("resident_forward")
        def once(repetition, unused):
            if stream:
                pltpu.trace_value("request_index", repetition)
            offset = repetition * (seq + 3)

            def embed(i, unused):
                idx = meta[offset + i]
                block = p["encoder.embeddings.tok_embeddings.weight"][pl.ds(idx // 8 * 8, 8), :].astype(jnp.float32)
                rows = jax.lax.broadcasted_iota(jnp.int32, (8, 384), 0)
                row = jnp.sum(jnp.where(rows == idx % 8, block, 0.0), axis=0, keepdims=True)
                old = xbuf[pl.ds(i // 8 * 8, 8), :]
                xbuf[pl.ds(i // 8 * 8, 8), :] = jnp.where(rows == i % 8, row, old)

            # Gather eight independent tokens before one aligned output store.
            def embed_block(group, unused):
                selected = []
                for lane in range(8):
                    idx = meta[offset + group * 8 + lane]
                    block = p["encoder.embeddings.tok_embeddings.weight"][pl.ds(idx // 8 * 8, 8), :].astype(jnp.float32)
                    rows = jax.lax.broadcasted_iota(jnp.int32, (8, 384), 0)
                    selected.append(jnp.sum(jnp.where(rows == idx % 8, block, 0.0), axis=0, keepdims=True))
                xbuf[pl.ds(group * 8, 8), :] = jnp.concatenate(selected, axis=0)

            if embed_group == 8:
                jax.lax.fori_loop(0, seq // 8, embed_block, None)
            else:
                jax.lax.fori_loop(0, seq, embed, None)
            xbuf[...] = norm(xbuf[...], "encoder.embeddings.norm")
            for layer in range(12):
                enc = layer < 10
                k = f"encoder.layers.{layer}" if enc else f"head.layers.{layer - 10}"
                if layer == 10:
                    xbuf[...] = norm(xbuf[...], "encoder.final_norm") + jnp.sum(
                        jnp.where(jax.lax.broadcasted_iota(jnp.int32, (3, 384), 0) == meta[offset + seq + 2], val("type_emb.weight").astype(jnp.float32), 0.0),
                        axis=0,
                        keepdims=True,
                    )
                x = xbuf[...]
                y = norm(x, k + (".attn_norm" if enc else ".norm1")) if layer != 0 else x
                wkey = k + ".attn.Wqkv.weight" if enc else k + ".self_attn.in_proj_weight"
                bkey = k + ".self_attn.in_proj_bias"
                for j in range(3):
                    cols = slice(j * 384, (j + 1) * 384)
                    z = jnp.dot(y.astype(jnp.bfloat16), p[wkey][:, cols], preferred_element_type=jnp.float32)
                    if not enc:
                        z += p[bkey][:, cols].astype(jnp.float32)
                    z = z.astype(jnp.bfloat16)
                    if enc and j < 2:
                        z = f._rope(z, val("cos"), val("sin"))
                    qkv[:, cols] = z
                bias = mask_bias(mask[repetition] if stream else mask[...], (seq, seq), 64 if enc and layer % 3 else 0)
                # Expose independent softmax/QK work before consuming each PV.
                for hg in range(0, 6, head_group):
                    probabilities = []
                    for h in range(hg, hg + head_group):
                        q, kk = (qkv[:, head_cols(h, j * 384)] for j in range(2))
                        st = scores_t(q, kk) + bias
                        prob = jnp.exp(st - st.max(0, keepdims=True))
                        probabilities.append(prob / prob.sum(0, keepdims=True))
                    for h in range(hg, hg + head_group):
                        v = qkv[:, head_cols(h, 2 * 384)]
                        att[:, head_cols(h)] = pv_t(v, probabilities[h - hg]).T.astype(jnp.bfloat16)
                x = x + linear(att[...], k + (".attn.Wo" if enc else ".self_attn.out_proj"))
                y = norm(x, k + (".mlp_norm" if enc else ".norm2"))
                y = linear(y, k + (".mlp.Wi" if enc else ".linear1"))
                if enc:
                    # Match unfused encoder's BF16 projection and activation semantics.
                    y = y.astype(jnp.bfloat16)
                    a, b = y[:, :640], y[:, 640:]
                    y = f._gelu(a.astype(jnp.float32)).astype(jnp.bfloat16) * b
                else:
                    y = jnp.maximum(y, 0.0)
                xbuf[...] = x + linear(y, k + (".mlp.Wo" if enc else ".linear2"))
            score[...] = linear(f._gelu(linear(norm(xbuf[...], "scorer.0"), "scorer.1")), "scorer.3")

            def select_score(idx):
                block = score[pl.ds(idx // 8 * 8, 8), :]
                rows = jax.lax.broadcasted_iota(jnp.int32, (8, 128), 0)
                return jnp.sum(jnp.where(rows == idx % 8, block, 0.0), axis=0, keepdims=True)

            a = select_score(meta[offset + seq])
            b = select_score(meta[offset + seq + 1])
            # Only first column contains scores. Broadcast to lanes for scalar math.
            a = jnp.sum(a, axis=1, keepdims=True)
            b = jnp.sum(b, axis=1, keepdims=True)
            shift = jnp.maximum(a, b)
            pa, pb = jnp.exp(a - shift), jnp.exp(b - shift)
            total = pa + pb
            pa, pb = pa / total, pb / total
            top = jnp.maximum(pa, pb)
            entropy = -(pa * jnp.log(jnp.maximum(pa, 1e-9)) + pb * jnp.log(jnp.maximum(pb, 1e-9))) / jnp.log(2.0)
            lane = jax.lax.broadcasted_iota(jnp.int32, (1, 128), 1)
            feats = jnp.where(lane == 0, top, jnp.where(lane == 1, jnp.abs(pa - pb), jnp.where(lane == 2, entropy, jnp.where(lane == 3, 2 / 255.0, 0.0))))
            act_input = jnp.concatenate((xbuf[:1, :], feats), axis=1)
            act = linear(f._gelu(linear(act_input, "act_head.0")), "act_head.2")
            out[repetition, 3:8, :] = jnp.zeros((5, 128), jnp.float32)
            out[repetition, 0:1, :] = jnp.broadcast_to(a, (1, 128))
            out[repetition, 1:2, :] = jnp.broadcast_to(b, (1, 128))
            out[repetition, 2:3, :] = act

        jax.lax.fori_loop(0, repeats, once, None)

    return jax.jit(
        pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((repeats, 8, 128), jnp.float32),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                in_specs=[pl.BlockSpec(), *[pl.BlockSpec() for _ in data]],
                out_specs=pl.BlockSpec(),
                scratch_shapes=[
                    pltpu.VMEM((seq, 384), jnp.float32),
                    pltpu.VMEM((seq, 1152), jnp.bfloat16),
                    pltpu.VMEM((seq, 384), jnp.bfloat16),
                    pltpu.VMEM((seq, 128), jnp.float32),
                ],
            ),
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_mib * 1024**2),
            name=f"student32_all_weights_vmem_r{repeats}",
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--vmem-mib", type=int, default=80)
    ap.add_argument("--embed-group", type=int, choices=(1, 8), default=1)
    ap.add_argument("--head-group", type=int, choices=(1, 2, 3, 6), default=1)
    args = ap.parse_args()
    assert jax.default_backend() == "tpu" and len(jax.devices()) == 1
    print("device", jax.devices(), flush=True)
    model = LayaModel(args.checkpoint, jnp.bfloat16, "jax")
    pallas_model = LayaModel(args.checkpoint, jnp.bfloat16, "pallas")
    print("models loaded", flush=True)
    question = parse_questions(
        {
            "sentiment": {
                "type": "choice",
                "instructions": "Is the sentiment positive or negative?",
                "criteria": {"positive": "positive sentiment", "negative": "negative sentiment"},
            }
        }
    )
    cases = [
        "I love this product. It works wonderfully.",
        "This is terrible. It broke on the first day.",
        "The package arrived this morning.",
        "I want a refund. This is the worst service ever.",
        "Absolutely excellent, thank you!",
        "It is okay, neither great nor awful.",
        "I am delighted with the quality and speed.",
        "The customer service was rude and unhelpful.",
    ]
    encoded = [collate(model.encode(s, question), model.pad_token_id) for s in cases]
    seq = max(args.seq, max(x["input_ids"].shape[1] for x in encoded))
    data = prepare(model, seq)
    print("layouts prepared", flush=True)
    fn = build(data, seq, args.repeats, args.vmem_mib, embed_group=args.embed_group, head_group=args.head_group)
    results = []
    exe = None
    comparison_inputs = []
    baseline_executables = {}
    for arrays in encoded:
        for key in ("input_ids", "attention_mask"):
            arrays[key] = np.pad(arrays[key], ((0, 0), (0, seq - arrays[key].shape[1])), constant_values=model.pad_token_id if key == "input_ids" else 0)
        arrays = {k: jax.device_put(v) for k, v in arrays.items()}
        ref = model.forward(arrays)
        jax.block_until_ready(ref)
        pallas_out = pallas_model.forward(arrays)
        jax.block_until_ready(pallas_out)
        if not baseline_executables:
            for name, instance in (("jax", model), ("pallas", pallas_model)):
                baseline_args = (instance.params, instance.fused, *[arrays[k] for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")])
                compiled = instance._forward.lower(*baseline_args).compile()
                baseline_executables[name] = compiled
                args.output.with_suffix(f".{name}.hlo.txt").write_text(compiled.as_text())
        print("references ready", flush=True)
        meta = jnp.concatenate((arrays["input_ids"][0], arrays["marker_pos"][0, :2], arrays["qtype"]))
        inputs = (jnp.tile(meta, args.repeats), arrays["attention_mask"], *data.values())
        if exe is None:
            start = time.perf_counter()
            exe = fn.lower(*inputs).compile()
            print("compiled", time.perf_counter() - start, flush=True)
            args.output.with_suffix(".hlo.txt").write_text(exe.as_text())
        comparison_inputs.append((arrays, inputs))
        all_out = np.asarray(exe(*inputs))
        out = all_out[-1]
        option = out[:2, 0]
        act = out[2, :2]
        np.testing.assert_allclose(all_out, np.broadcast_to(out, all_out.shape), rtol=0, atol=0)
        prob = lambda x: np.asarray(jax.nn.softmax(jnp.asarray(x)))
        probability_error = float(np.max(np.abs(prob(option) - prob(np.asarray(ref[0])[0, :2]))))
        assert probability_error < 0.02, probability_error
        results.append(
            {
                "reference_logits": np.asarray(ref[0]).tolist(),
                "pallas_logits": np.asarray(pallas_out[0]).tolist(),
                "pallas_probability_max_abs": float(np.max(np.abs(prob(np.asarray(pallas_out[0])[0]) - prob(np.asarray(ref[0])[0])))),
                "pallas_choice_agreement": bool(np.argmax(np.asarray(pallas_out[0])[0]) == np.argmax(np.asarray(ref[0])[0])),
                "pallas_act_max_abs": float(np.max(np.abs(np.asarray(pallas_out[1]) - np.asarray(ref[1])))),
                "vmem_logits": option.tolist(),
                "reference_act": np.asarray(ref[1]).tolist(),
                "vmem_act": act.tolist(),
                "probability_max_abs": probability_error,
                "choice_agreement": bool(np.argmax(option) == np.argmax(np.asarray(ref[0])[0, :2])),
                "logit_max_abs": float(np.max(np.abs(option - np.asarray(ref[0])[0, :2]))),
                "act_max_abs": float(np.max(np.abs(act - np.asarray(ref[1])[0]))),
            }
        )
        print(results[-1], flush=True)

    # Same compiled-call boundary for all three paths; rotate order and inputs.
    samples = {"jax": [], "pallas": [], "pallas_full_vmem": []}
    names = tuple(samples)
    for cycle in range(12):
        for arrays, inputs in comparison_inputs:
            baseline_args = {
                name: (instance.params, instance.fused, *[arrays[k] for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")])
                for name, instance in (("jax", model), ("pallas", pallas_model))
            }
            for name in names[cycle % 3 :] + names[: cycle % 3]:
                start = time.perf_counter_ns()
                output = exe(*inputs) if name == "pallas_full_vmem" else baseline_executables[name](*baseline_args[name])
                jax.block_until_ready(output)
                elapsed = (time.perf_counter_ns() - start) / 1e6
                if cycle >= 2:
                    samples[name].append(elapsed)
    timing = {name: {"p50_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)), "samples_ms": values} for name, values in samples.items()}

    report = {
        "device": str(jax.devices()[0]),
        "seq": seq,
        "repeats_per_call": args.repeats,
        "vmem_limit_mib": args.vmem_mib,
        "embed_group": args.embed_group,
        "head_group": args.head_group,
        "resident_input_bytes": sum(x.size * x.dtype.itemsize for x in data.values()),
        "timing": timing,
        "protocol": "Compiled calls, synchronized host timing, 8 inputs x 10 measured cycles after 2 warmup cycles, rotating backend order; tokenization excluded. Full VMEM includes weight load per call.",
        "cases": results,
        "memory_analysis": str(exe.memory_analysis()),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
