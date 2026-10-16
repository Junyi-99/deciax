"""Device traces of hot JAX/Pallas and weight-preloaded VMEM forward scopes.

VMEM call includes a warmup request and eight different requests. Only named
forward scopes after the warmup are measured; preload is outside these scopes.
Baselines use hot compiled calls and device XLA Modules intervals, not host time.
"""

import argparse
import gzip
import importlib.metadata
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from student32_vmem import build, prepare
from xprof.convert import raw_to_tool_data

from deciax.models.laya.model import LayaModel
from deciax.protocol.types import parse_questions
from deciax.runtime.buckets import collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--modes", nargs="+", choices=("TRACE_COMPUTE", "TRACE_COMPUTE_AND_SYNC"), default=["TRACE_COMPUTE_AND_SYNC"])
    ap.add_argument("--backends", nargs="+", choices=("jax", "pallas", "vmem"), default=["jax", "pallas", "vmem"])
    ap.add_argument("--embed-group", type=int, default=1)
    ap.add_argument("--head-group", type=int, default=1)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "environment.json").write_text(
        json.dumps(
            {
                "versions": {k: importlib.metadata.version(k) for k in ("xprof-nightly", "jax", "jaxlib", "libtpu")},
                "LIBTPU_INIT_ARGS": os.environ.get("LIBTPU_INIT_ARGS"),
                "modes": args.modes,
                "steps": args.steps,
                "embed_group": args.embed_group,
                "head_group": args.head_group,
            },
            indent=2,
        )
    )
    assert len(jax.devices()) == 1 and jax.default_backend() == "tpu"
    models = {name: LayaModel(args.checkpoint, jnp.bfloat16, name) for name in ("jax", "pallas")}
    model = models["jax"]
    question = parse_questions(
        {
            "sentiment": {
                "type": "choice",
                "instructions": "Is the sentiment positive or negative?",
                "criteria": {"positive": "positive sentiment", "negative": "negative sentiment"},
            }
        }
    )
    texts = [
        "I love this product. It works wonderfully.",
        "This is terrible. It broke on the first day.",
        "The package arrived this morning.",
        "I want a refund. This is the worst service ever.",
        "Absolutely excellent, thank you!",
        "It is okay, neither great nor awful.",
        "I am delighted with the quality and speed.",
        "The customer service was rude and unhelpful.",
    ]
    arrays = []
    for text in texts:
        a = collate(model.encode(text, question), model.pad_token_id)
        for key in ("input_ids", "attention_mask"):
            a[key] = np.pad(a[key], ((0, 0), (0, 128 - a[key].shape[1])), constant_values=model.pad_token_id if key == "input_ids" else 0)
        arrays.append({k: jax.device_put(v) for k, v in a.items()})
    baseline_args = {
        name: [(m.params, m.fused, *[a[k] for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]) for a in arrays]
        for name, m in models.items()
    }
    exes = {name: m._forward.lower(*baseline_args[name][0]).compile() for name, m in models.items()}
    stream = [arrays[0], *arrays]  # index 0 warms compute; indices 1..8 vary the inputs.
    meta = jnp.concatenate([jnp.concatenate((a["input_ids"][0], a["marker_pos"][0, :2], a["qtype"])) for a in stream])
    masks = jnp.stack([a["attention_mask"] for a in stream])
    data = prepare(model, 128)
    inputs = (meta, masks, *data.values())
    fn = build(data, 128, len(stream), 80, stream=True, embed_group=args.embed_group, head_group=args.head_group)
    vm = fn.lower(*inputs).compile()
    (args.out / "vmem.hlo.txt").write_text(vm.as_text())
    actual = np.asarray(vm(*inputs))
    original = build(data, 128, len(stream), 80, stream=True).lower(*inputs).compile()
    original_output = np.asarray(original(*inputs))
    # Reordering must preserve all output lanes, including the action head.
    np.testing.assert_array_equal(actual, original_output)
    (args.out / "variant-validation.json").write_text(json.dumps({"baseline_vmem_bitwise_equal": True, "outputs_compared": int(actual.size)}))
    validation = []
    for i, a in enumerate(arrays):
        reference = np.asarray(exes["jax"](*baseline_args["jax"][i])[0])[0]
        logits = actual[i + 1, :2, 0]
        err = float(np.max(np.abs(np.asarray(jax.nn.softmax(jnp.asarray(logits))) - np.asarray(jax.nn.softmax(jnp.asarray(reference))))))
        assert err < 0.02, err
        validation.append({"input": texts[i], "reference_logits": reference.tolist(), "vmem_logits": logits.tolist(), "probability_max_abs": err})
    assert not np.all(actual[1:, :2, 0] == actual[1, :2, 0]), "Inputs did not produce distinct outputs"
    (args.out / "validation.json").write_text(json.dumps(validation, indent=2))
    for _ in range(3):
        jax.block_until_ready(vm(*inputs))
        for name, exe in exes.items():
            for a in baseline_args[name]:
                jax.block_until_ready(exe(*a))
    for mode in args.modes:
        options = jax.profiler.ProfileOptions()
        options.python_tracer_level = 0
        options.advanced_configuration = {"tpu_trace_mode": mode, "tpu_num_chips_to_profile_per_task": 1}
        for name in args.backends:
            directory = args.out / mode / name
            with jax.profiler.trace(str(directory), profiler_options=options):
                for step in range(args.steps):
                    with jax.profiler.StepTraceAnnotation(name, step_num=step):
                        if name == "vmem":
                            jax.block_until_ready(vm(*inputs))
                        else:
                            for a in baseline_args[name]:
                                jax.block_until_ready(exes[name](*a))
            paths = [str(p) for p in directory.rglob("*.xplane.pb")]
            assert paths
            raw, _ = raw_to_tool_data.xspace_to_tool_data(paths, "trace_viewer@", {})
            if isinstance(raw, str):
                raw = raw.encode()
            (directory / "trace.json.gz").write_bytes(gzip.compress(raw))
            events = json.loads(raw)["traceEvents"]
            tracks = [e for e in events if e.get("name") == "thread_name"]
            scopes = [e for e in events if "resident_forward" in e.get("name", "")]
            (directory / "inspection.json").write_text(json.dumps({"tracks": tracks, "scopes": scopes[:30], "scope_count": len(scopes)}, indent=2))
            print(mode, name, "events", len(events), "resident scopes", len(scopes), flush=True)


if __name__ == "__main__":
    main()
