"""Micro-benchmark of the flash attention kernel variants on laya's 64 x 256 x 16 heads shape.

python benchmarks/attn_micro.py [--seq 256] [--rows 64] [--heads 16] [--window 0]
"""

from __future__ import annotations

import argparse
import itertools
import time

import jax
import jax.numpy as jnp
import numpy as np

from deciax.kernels.attention import pallas, reference


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--window", type=int, default=0)
    args = ap.parse_args()
    b, s, h = args.rows, args.seq, args.heads
    d = h * 64
    key = jax.random.key(0)
    qkv = (jax.random.normal(key, (b, s, 3 * d), jnp.float32) * 1.5).astype(jnp.bfloat16)
    mask = jnp.ones((b, s), jnp.int32).at[:, s - 40 :].set(0)
    q, k, v = (t.reshape(b, s, h, 64) for t in jnp.split(qkv, 3, -1))
    want = np.asarray(reference.attention(q, k, v, mask, args.window)).reshape(b, s, d)
    valid = np.asarray(mask, bool)
    flops = 4 * b * h * s * s * 64

    def bench(fn):
        fn()
        ms = []
        for _ in range(20):
            t = time.perf_counter()
            jax.block_until_ready(fn())
            ms.append((time.perf_counter() - t) * 1000)
        return float(np.median(ms))

    ref = jax.jit(lambda: reference.attention(q, k, v, mask, args.window))
    print(f"reference (XLA): {bench(ref):.3f} ms")
    default = jax.jit(lambda: pallas.flash_attention(qkv, mask, heads=h, window=args.window))
    print(f"pallas, default blocks: {bench(default):.3f} ms")
    for tq, tk in itertools.product((128, 256, 512), (128, 256, 512)):
        if tq > s or tk > s:
            continue
        fn = jax.jit(lambda tq=tq, tk=tk: pallas.flash_attention(qkv, mask, heads=h, window=args.window, tq=tq, tk=tk))
        try:
            ms = bench(fn)
            err = float(np.abs(np.asarray(fn(), np.float32)[valid] - want[valid]).max())
            print(f"tq={tq:3d} tk={tk:3d}: {ms:.3f} ms  {flops / ms / 1e9:6.0f} TFLOP/s  max|err| {err:.4f}")
        except Exception as e:  # noqa: BLE001
            print(f"tq={tq:3d} tk={tk:3d}: ERR {str(e).splitlines()[0][:160]}")


if __name__ == "__main__":
    main()
