"""TPU RMSNorm + SwiGLU + down projection, fused over intermediate tiles.

Weights are packed once as [tiles, D, 2*Ftile] and [tiles, Ftile, D].
Gate/up tiles fuse activation in their epilogue and retain the full local
activation in VMEM. Down tiles accumulate across all F before one output store.
Both weight streams use ping-pong DMA with independent, disjoint lifetimes.
"""

from dataclasses import dataclass
from math import gcd
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


@dataclass(frozen=True)
class MlpTiling:
    rows: int = 512
    intermediate: int = 256
    buffers: int = 2
    vmem_limit_bytes: int = 127 * 1024 * 1024
    output_buffers: int = 1
    input_buffers: int = 1

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.rows % 256 or self.intermediate <= 0 or self.intermediate % 256:
            raise ValueError("MLP row and intermediate tiles must be positive multiples of 256")
        if self.buffers not in (1, 2):
            raise ValueError("MLP supports one or two pipeline buffers")
        if self.output_buffers not in (1, 2):
            raise ValueError("MLP supports one or two output buffers")
        if self.input_buffers not in (1, 2):
            raise ValueError("MLP supports one or two input buffers")


DEFAULT_TILING = MlpTiling()


def pack(gate: jax.Array, up: jax.Array, down: jax.Array, tile: int) -> tuple[jax.Array, jax.Array]:
    """Pack one TP shard; zero-pad F so every MXU dimension is 256-aligned."""
    f, d = gate.shape
    padding = -f % tile
    gate, up = (jnp.pad(w, ((0, padding), (0, 0))).reshape(-1, tile, d).swapaxes(1, 2) for w in (gate, up))
    down = jnp.pad(down, ((0, 0), (0, padding))).T.reshape(-1, tile, d)
    return jnp.concatenate((gate, up), axis=-1), down


def fused_mlp_partial(
    x: jax.Array,
    norm: jax.Array,
    gate_up: jax.Array,
    down: jax.Array,
    *,
    eps: float,
    tiling: MlpTiling = DEFAULT_TILING,
    interpret: bool = False,
) -> jax.Array:
    """[rows,D] BF16 -> FP32 local down sum, before TP psum and rounding.

    Inputs/outputs use [rows,channels] tiles: both trailing axes align to the
    VReg (8,128) geometry. BF16 rounding follows Qwen3.5's RMSNorm/SwiGLU.
    """
    rows, d = x.shape
    m, f, nt = min(tiling.rows, (rows + 255) // 256 * 256), tiling.intermediate, gate_up.shape[0]
    if d % 256 or gate_up.shape != (nt, d, 2 * f) or down.shape != (nt, f, d):
        raise ValueError("invalid packed MLP shape or hidden size not divisible by 256")
    if any(a.dtype != jnp.bfloat16 for a in (x, norm, gate_up, down)):
        raise ValueError("fused Clef MLP requires BF16 inputs and weights")
    x = jnp.pad(x, ((0, -rows % m), (0, 0)))
    bm, bn, bk = (gcd(v, 512) for v in (m, d, d))
    bf = 256
    down_group = gcd(d, 1024)
    groups = d // down_group
    buffers = tiling.buffers

    def kernel(xr: Any, nr: Any, gur: Any, dr: Any, yr: Any, hr: Any, ar: Any) -> None:
        def normalize(i: jax.Array, _: None) -> None:
            r = pl.ds(i * 256, 256)
            h = xr[r, :].astype(jnp.float32)
            h *= jax.lax.rsqrt(jnp.mean(h * h, axis=-1, keepdims=True) + eps)
            hr[r, :] = (h * (nr[...].astype(jnp.float32) + 1)).astype(jnp.bfloat16)

        jax.lax.fori_loop(0, m // 256, normalize, None)

        # Gate/up and down weight buffers have disjoint scopes/lifetimes.
        def gate_stage(w: Any, sem: Any) -> None:
            def transfer(i: Any) -> Any:
                return pltpu.make_async_copy(gur.at[i], w.at[i % buffers], sem.at[i % buffers])

            if not interpret:
                transfer(0).start()

            def step(i: jax.Array, _: None) -> None:
                slot = i % buffers
                if interpret:
                    w[slot, ...] = gur[i, ...]
                else:
                    transfer(i).wait()
                    if buffers == 2:

                        @pl.when(i + 1 < nt)
                        def prefetch() -> None:
                            transfer(i + 1).start()

                def tile(t: jax.Array, _: None) -> None:
                    row = pl.ds(t // (f // bf) * bm, bm)
                    col = t % (f // bf) * bf

                    def reduce(k: jax.Array, acc: jax.Array) -> jax.Array:
                        inner = pl.ds(k * bk, bk)
                        # Adjacent gate/up columns form a <=512-column dot,
                        # with BF16 rounding before the fused SiLU epilogue.
                        weights = jnp.concatenate((w[slot, inner, pl.ds(col, bf)], w[slot, inner, pl.ds(f + col, bf)]), axis=1)
                        return acc + jnp.dot(hr[row, inner], weights, preferred_element_type=jnp.float32)

                    gu = jax.lax.fori_loop(0, d // bk, reduce, jnp.zeros((bm, 2 * bf), jnp.float32)).astype(jnp.bfloat16)
                    activated = (jax.nn.silu(gu[:, :bf]).astype(jnp.bfloat16) * gu[:, bf:]).astype(jnp.bfloat16)
                    ar[row, pl.ds(i * f + col, bf)] = activated

                jax.lax.fori_loop(0, (m // bm) * (f // bf), tile, None)
                if not interpret and buffers == 1:

                    @pl.when(i + 1 < nt)
                    def prefetch_single() -> None:
                        transfer(i + 1).start()

            jax.lax.fori_loop(0, nt, step, None)

        pl.run_scoped(gate_stage, pltpu.VMEM((buffers, d, 2 * f), jnp.bfloat16), pltpu.SemaphoreType.DMA((buffers,)))

        def down_stage(w: Any, sem: Any) -> None:
            def transfer(i: Any) -> Any:
                # Fetch all F for an output-column group in one DMA. Adjacent
                # <=512-column dots share this buffer; no per-F DMA waits.
                col = pl.ds(i * down_group, down_group)
                return pltpu.make_async_copy(dr.at[:, :, col], w.at[i % buffers], sem.at[i % buffers])

            if not interpret:
                transfer(0).start()

            def group(i: jax.Array, _: None) -> None:
                slot = i % buffers
                if interpret:
                    w[slot, ...] = dr[:, :, pl.ds(i * down_group, down_group)]
                else:
                    transfer(i).wait()
                    if buffers == 2:

                        @pl.when(i + 1 < groups)
                        def prefetch() -> None:
                            transfer(i + 1).start()

                def output_tile(t: jax.Array, _: None) -> None:
                    row = pl.ds(t // (down_group // bn) * bm, bm)
                    offset = t % (down_group // bn) * bn
                    col = pl.ds(offset, bn)

                    def reduce(k: jax.Array, acc: jax.Array) -> jax.Array:
                        f_tile = k // (f // bf)
                        inner = pl.ds(k % (f // bf) * bf, bf)
                        return acc + jnp.dot(ar[row, pl.ds(k * bf, bf)], w[slot, f_tile, inner, col], preferred_element_type=jnp.float32)

                    # The FP32 tile is loop-carried through the entire F
                    # reduction, then stored exactly once to the output Ref.
                    acc = jax.lax.fori_loop(0, nt * (f // bf), reduce, jnp.zeros((bm, bn), jnp.float32))
                    yr[row, pl.ds(i * down_group + offset, bn)] = acc

                jax.lax.fori_loop(0, (m // bm) * (down_group // bn), output_tile, None)
                if not interpret and buffers == 1:

                    @pl.when(i + 1 < groups)
                    def prefetch_single() -> None:
                        transfer(i + 1).start()

            jax.lax.fori_loop(0, groups, group, None)

        pl.run_scoped(down_stage, pltpu.VMEM((buffers, nt, f, down_group), jnp.bfloat16), pltpu.SemaphoreType.DMA((buffers,)))

    result = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, jnp.float32),  # type: ignore[no-untyped-call]
        grid_spec=pl.GridSpec(
            grid=(x.shape[0] // m,),
            in_specs=[
                pl.BlockSpec((m, d), lambda i: (i, 0), pipeline_mode=pl.Buffered(buffer_count=tiling.input_buffers)),
                pl.BlockSpec((1, d), lambda i: (0, 0)),
                pl.BlockSpec(memory_space=pl.ANY),
                pl.BlockSpec(memory_space=pl.ANY),
            ],
            out_specs=pl.BlockSpec((m, d), lambda i: (i, 0), pipeline_mode=pl.Buffered(buffer_count=tiling.output_buffers)),
            scratch_shapes=[pltpu.VMEM((m, d), jnp.bfloat16), pltpu.VMEM((m, nt * f), jnp.bfloat16)],
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",), vmem_limit_bytes=tiling.vmem_limit_bytes),
        interpret=interpret,
        name=f"clef_swiglu_m{m}_f{f}_b{tiling.buffers}",
    )(x, norm.reshape(1, d), gate_up, down)
    return result[:rows]  # type: ignore[no-any-return]
