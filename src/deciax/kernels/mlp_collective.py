"""Experimental TP4 MLP with row-tile compute / RS / AG overlap.

Remote transfers stay in VMEM. Capacity credits protect each two-slot receive
buffer against a fast sender overtaking its consumer. No partial HBM output.
"""

from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from deciax.kernels.swiglu import make_mlp_tile


def mlp_rs_ag(x: jax.Array, norm: jax.Array, gu: jax.Array, down: jax.Array, *, eps: float, rows: int = 512, overlap: bool = True) -> jax.Array:
    """Local TP4 shard -> replicated BF16 projection+residual, inside shard_map."""
    if jax.lax.axis_size("tp") != 4:
        raise ValueError("experimental collective requires TP4")
    r, d = x.shape
    m = min(rows, (r + 255) // 256 * 256)
    if m not in (256, 512) or d % 256:
        raise ValueError("requires M256/512 and 256-aligned hidden dimension")
    nt, _, twice_f = gu.shape
    f = twice_f // 2
    x = jnp.pad(x, ((0, -r % m), (0, 0)))
    tiles, owned = x.shape[0] // m, m // 4
    compute = make_mlp_tile(m, d, f, nt, eps, 2)

    def kernel(
        xr: Any,
        nr: Any,
        gur: Any,
        dr: Any,
        yr: Any,
        xv: Any,
        nv: Any,
        hr: Any,
        ar: Any,
        partial: Any,
        received: Any,
        reduced: Any,
        gathered: Any,
        rs_send: Any,
        rs_recv: Any,
        ag_send: Any,
        ag_recv: Any,
        rs_capacity: Any,
        ag_capacity: Any,
        local_sem: Any,
    ) -> None:
        rank = jax.lax.axis_index("tp")
        barrier = pltpu.get_barrier_semaphore()
        for phase in range(1, 4):
            pl.semaphore_signal(barrier, 1, device_id={"tp": rank ^ phase})
        pl.semaphore_wait(barrier, 3)
        pltpu.async_copy(nr, nv, local_sem).wait()  # type: ignore[no-untyped-call]

        def rs_descriptor(i: Any, phase: int) -> Any:
            slot, peer = i % 2, rank ^ phase
            return pltpu.make_async_remote_copy(
                partial.at[slot, pl.ds(peer * owned, owned), :],
                received.at[slot, phase - 1],
                rs_send.at[slot, phase - 1],
                rs_recv.at[slot, phase - 1],
                device_id={"tp": peer},
            )

        def ag_descriptor(i: Any, phase: int, incoming: bool = False) -> Any:
            slot, peer = i % 2, rank ^ phase
            dest = peer if incoming else rank
            return pltpu.make_async_remote_copy(
                reduced.at[slot], gathered.at[slot, dest], ag_send.at[slot, phase - 1], ag_recv.at[slot, phase - 1], device_id={"tp": peer}
            )

        def compute_tile(i: Any) -> None:
            with jax.named_scope("compute_tile"):
                pltpu.async_copy(xr.at[pl.ds(i * m, m), :], xv, local_sem).wait()  # type: ignore[no-untyped-call]
                compute(xv, nv, gur, dr, partial.at[i % 2], hr, ar)

        def start_rs(i: Any) -> None:
            with jax.named_scope("rs_start"):
                for phase in range(1, 4):

                    @pl.when(i >= 2)
                    def capacity(phase: int = phase) -> None:
                        pl.semaphore_wait(rs_capacity.at[i % 2, phase - 1], 1)

                    rs_descriptor(i, phase).start()

        def finish_rs_start_ag(i: Any) -> None:
            with jax.named_scope("rs_finish_ag_start"):
                slot = i % 2
                for phase in range(1, 4):
                    rs_descriptor(i, phase).wait()

                # Canonical source-rank order, with a single final BF16 cast.
                def add_source(source: Any, acc: Any) -> Any:
                    value = jax.lax.cond(
                        source == rank, lambda: partial[slot, pl.ds(rank * owned, owned), :], lambda: received[slot, (rank ^ source) - 1, :, :]
                    )
                    return acc + value

                reduced[slot] = jax.lax.fori_loop(0, 4, add_source, jnp.zeros((owned, d), jnp.float32)).astype(jnp.bfloat16)
                for phase in range(1, 4):

                    @pl.when(i + 2 < tiles)
                    def credit(phase: int = phase) -> None:
                        pl.semaphore_signal(rs_capacity.at[slot, phase - 1], 1, device_id={"tp": rank ^ phase})

                    @pl.when(i >= 2)
                    def capacity(phase: int = phase) -> None:
                        pl.semaphore_wait(ag_capacity.at[slot, phase - 1], 1)

                gathered[slot, rank] = reduced[slot]
                for phase in range(1, 4):
                    ag_descriptor(i, phase).start()

        def finish_ag(i: Any) -> None:
            with jax.named_scope("ag_finish_store"):
                slot = i % 2
                for phase in range(1, 4):
                    ag_descriptor(i, phase).wait_send()
                    ag_descriptor(i, phase, incoming=True).wait_recv()
                # Reuse local staging after compute; residual is original BF16.
                pltpu.async_copy(xr.at[pl.ds(i * m, m), :], xv, local_sem).wait()  # type: ignore[no-untyped-call]

                def residual(k: Any, _: None) -> None:
                    xv[pl.ds(k * owned, owned), :] = (xv[pl.ds(k * owned, owned), :] + gathered[slot, k]).astype(jnp.bfloat16)

                jax.lax.fori_loop(0, 4, residual, None)
                pltpu.async_copy(xv, yr.at[pl.ds(i * m, m), :], local_sem).wait()  # type: ignore[no-untyped-call]
                for phase in range(1, 4):

                    @pl.when(i + 2 < tiles)
                    def credit(phase: int = phase) -> None:
                        pl.semaphore_signal(ag_capacity.at[slot, phase - 1], 1, device_id={"tp": rank ^ phase})

        def step(i: Any, _: None) -> None:
            if overlap:

                @pl.when(i < tiles)
                def compute_current() -> None:
                    compute_tile(i)

                @pl.when(i >= 2)
                def gather_previous() -> None:
                    finish_ag(i - 2)

                @pl.when((i >= 1) & (i < tiles + 1))
                def reduce_previous() -> None:
                    finish_rs_start_ag(i - 1)

                @pl.when(i < tiles)
                def scatter_current() -> None:
                    start_rs(i)
            else:
                compute_tile(i)
                start_rs(i)
                finish_rs_start_ag(i)
                finish_ag(i)

        jax.lax.fori_loop(0, tiles + (2 if overlap else 0), step, None)

    return pl.pallas_call(  # type: ignore[no-any-return]
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, jnp.bfloat16),  # type: ignore[no-untyped-call]
        grid_spec=pl.GridSpec(
            in_specs=[pl.BlockSpec(memory_space=pl.ANY)] * 4,
            out_specs=pl.BlockSpec(memory_space=pl.ANY),
            scratch_shapes=[
                pltpu.VMEM((m, d), jnp.bfloat16),
                pltpu.VMEM((1, d), jnp.bfloat16),
                pltpu.VMEM((m, d), jnp.bfloat16),
                pltpu.VMEM((m, nt * f), jnp.bfloat16),
                pltpu.VMEM((2, m, d), jnp.float32),
                pltpu.VMEM((2, 3, owned, d), jnp.float32),
                pltpu.VMEM((2, owned, d), jnp.bfloat16),
                pltpu.VMEM((2, 4, owned, d), jnp.bfloat16),
                *[pltpu.SemaphoreType.DMA((2, 3)) for _ in range(4)],
                *[pltpu.SemaphoreType.REGULAR((2, 3)) for _ in range(2)],
                pltpu.SemaphoreType.DMA,
            ],
        ),
        compiler_params=pltpu.CompilerParams(collective_id=0, vmem_limit_bytes=127 * 1024**2),
        name=f"clef_mlp_tp4_{'overlap' if overlap else 'serial'}_m{m}",
    )(x, norm.reshape(1, d), gu, down)[:r]
