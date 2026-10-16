"""Clef TP4 Pallas MLP: raw/topology mesh × psum/explicit RS→AG.

Synthetic R2048,D5120,F4352 per chip by default. Timings include host dispatch;
this measures a single MLP, not complete Clef inference or communication overlap.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from deciax.kernels.swiglu import fused_mlp_partial
from deciax.models.qwen3_5.parallel import sum_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    devices = jax.local_devices()
    if len(devices) != 4 or jax.default_backend() != "tpu":
        parser.error("requires four local TPU devices")
    result = {"jax": jax.__version__, "rows": args.rows, "hidden": 5120, "local_intermediate": 4352, "measurements": []}
    for order, device_mesh in [("raw", np.asarray(devices)), ("topology", mesh_utils.create_device_mesh((4,), devices=devices))]:
        mesh = Mesh(device_mesh, ("tp",))

        def array(shape, spec, seed, scale, mesh=mesh):
            def callback(index):
                rng = np.random.default_rng(seed + (index[0].start or 0))
                shape_local = tuple(len(range(*s.indices(n))) for s, n in zip(index, shape, strict=True))
                return (rng.standard_normal(shape_local, dtype=np.float32) * scale).astype(jnp.bfloat16)

            return jax.make_array_from_callback(shape, NamedSharding(mesh, spec), callback)

        inputs = (
            array((args.rows, 5120), P(), 0, 1),
            array((5120,), P(), 1, 0.1),
            array((68, 5120, 512), P("tp", None, None), 2, 5120**-0.5),
            array((68, 256, 5120), P("tp", None, None), 3, 4352**-0.5),
        )
        reference = None
        for split in (False, True):

            def local(x, norm, gu, down, split=split):
                partial = fused_mlp_partial(x, norm, gu, down, eps=1e-6)
                summed = sum_rows(partial, axis_size=4) if split else jax.lax.psum(partial, "tp")
                return (x + summed.astype(x.dtype)).astype(x.dtype)

            fn = jax.jit(jax.shard_map(local, mesh=mesh, in_specs=(P(), P(), P("tp", None, None), P("tp", None, None)), out_specs=P(), check_vma=False))
            compiled = fn.lower(*inputs).compile()
            actual = np.asarray(compiled(*inputs), np.float32)
            if reference is None:
                reference = actual
            np.testing.assert_allclose(actual, reference, atol=0.04, rtol=0.02)
            assert np.isfinite(actual).all()
            for _ in range(4):
                jax.block_until_ready(compiled(*inputs))
            samples = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                jax.block_until_ready(compiled(*inputs))
                samples.append((time.perf_counter() - start) * 1e6)
            result["measurements"].append(
                {
                    "mesh": order,
                    "device_order": [d.id for d in device_mesh.flat],
                    "collective": "rs_ag" if split else "psum",
                    "median_host_sync_us": statistics.median(samples),
                    "samples_us": samples,
                    "max_abs_error_vs_psum": float(np.max(np.abs(actual - reference))),
                }
            )
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result["measurements"][-1]), flush=True)


if __name__ == "__main__":
    main()
