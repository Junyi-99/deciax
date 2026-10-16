"""Experimental TP4 MLP with row-tile compute / RS / AG overlap.

Remote transfers stay in VMEM. Capacity credits protect each two-slot receive
buffer against a fast sender overtaking its consumer. No partial HBM output.

For iteration t, before computation begins:
    MXU: compute t     ICI: scatter t-1 and gather t-2 are in flight
After computation: store t-2, reduce t-1 / start its gather, start scatter t.
Two extra iterations drain the pipeline. Tile t always uses slot t % 2.
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
    num_rows, hidden = x.shape
    tile_rows = min(rows, (num_rows + 255) // 256 * 256)
    if tile_rows not in (256, 512) or hidden % 256:
        raise ValueError("requires M256/512 and 256-aligned hidden dimension")
    num_weight_tiles, _, twice_f = gu.shape
    weight_tile_width = twice_f // 2
    x = jnp.pad(x, ((0, -num_rows % tile_rows), (0, 0)))
    num_tiles, shard_rows = x.shape[0] // tile_rows, tile_rows // 4
    compute_local_mlp = make_mlp_tile(tile_rows, hidden, weight_tile_width, num_weight_tiles, eps, 2)

    def kernel(
        input_ref: Any,
        norm_ref: Any,
        gate_up_ref: Any,
        down_ref: Any,
        output_ref: Any,
        input_tile: Any,
        norm_tile: Any,
        normalized_tile: Any,
        activation_tile: Any,
        partials: Any,
        rs_inbox: Any,
        local_shards: Any,
        ag_inbox: Any,
        rs_send_done: Any,
        rs_recv_done: Any,
        ag_send_done: Any,
        ag_recv_done: Any,
        rs_slot_free: Any,
        ag_slot_free: Any,
        local_dma_done: Any,
    ) -> None:
        rank = jax.lax.axis_index("tp")
        barrier = pltpu.get_barrier_semaphore()
        for peer_offset in range(1, 4):
            pl.semaphore_signal(barrier, 1, device_id={"tp": rank ^ peer_offset})
        pl.semaphore_wait(barrier, 3)
        pltpu.async_copy(norm_ref, norm_tile, local_dma_done).wait()  # type: ignore[no-untyped-call]

        # XOR offsets 1/2/3 enumerate the other ranks symmetrically. Scatter
        # inboxes are indexed by peer_offset-1; gather inboxes by source rank.
        def scatter_transfer(tile: Any, peer_offset: int) -> Any:
            slot, peer = tile % 2, rank ^ peer_offset
            return pltpu.make_async_remote_copy(
                partials.at[slot, pl.ds(peer * shard_rows, shard_rows), :],
                rs_inbox.at[slot, peer_offset - 1],
                rs_send_done.at[slot, peer_offset - 1],
                rs_recv_done.at[slot, peer_offset - 1],
                device_id={"tp": peer},
            )

        def gather_transfer(tile: Any, peer_offset: int, incoming: bool = False) -> Any:
            slot, peer = tile % 2, rank ^ peer_offset
            # Sending writes our rank's slot on the peer. Waiting for incoming
            # data instead names the peer's slot in our own receive buffer.
            dest = peer if incoming else rank
            return pltpu.make_async_remote_copy(
                local_shards.at[slot],
                ag_inbox.at[slot, dest],
                ag_send_done.at[slot, peer_offset - 1],
                ag_recv_done.at[slot, peer_offset - 1],
                device_id={"tp": peer},
            )

        def compute_tile(tile: Any) -> None:
            with jax.named_scope("compute_tile"):
                pltpu.async_copy(input_ref.at[pl.ds(tile * tile_rows, tile_rows), :], input_tile, local_dma_done).wait()  # type: ignore[no-untyped-call]
                compute_local_mlp(input_tile, norm_tile, gate_up_ref, down_ref, partials.at[tile % 2], normalized_tile, activation_tile)

        # DMA completion protects the sender's source buffer. Slot-free credits
        # separately protect the receiver until it has consumed that buffer.
        # Slots 0/1 start free; tile t>=2 waits for the credit from tile t-2.
        def start_scatter(tile: Any) -> None:
            with jax.named_scope("rs_start"):
                for peer_offset in range(1, 4):

                    @pl.when(tile >= 2)
                    def wait_for_free_slot(peer_offset: int = peer_offset) -> None:
                        pl.semaphore_wait(rs_slot_free.at[tile % 2, peer_offset - 1], 1)

                    scatter_transfer(tile, peer_offset).start()

        def reduce_and_start_gather(tile: Any) -> None:
            with jax.named_scope("rs_finish_ag_start"):
                slot = tile % 2
                for peer_offset in range(1, 4):
                    scatter_transfer(tile, peer_offset).wait()

                # Canonical source-rank order, with a single final BF16 cast.
                def add_source(source: Any, acc: Any) -> Any:
                    value = jax.lax.cond(
                        source == rank, lambda: partials[slot, pl.ds(rank * shard_rows, shard_rows), :], lambda: rs_inbox[slot, (rank ^ source) - 1, :, :]
                    )
                    return acc + value

                local_shards[slot] = jax.lax.fori_loop(0, 4, add_source, jnp.zeros((shard_rows, hidden), jnp.float32)).astype(jnp.bfloat16)
                # Only release credits that a future tile will consume.
                for peer_offset in range(1, 4):

                    @pl.when(tile + 2 < num_tiles)
                    def release_slot(peer_offset: int = peer_offset) -> None:
                        pl.semaphore_signal(rs_slot_free.at[slot, peer_offset - 1], 1, device_id={"tp": rank ^ peer_offset})

                    @pl.when(tile >= 2)
                    def wait_for_free_slot(peer_offset: int = peer_offset) -> None:
                        pl.semaphore_wait(ag_slot_free.at[slot, peer_offset - 1], 1)

                ag_inbox[slot, rank] = local_shards[slot]
                for peer_offset in range(1, 4):
                    gather_transfer(tile, peer_offset).start()

        def finish_gather_and_store(tile: Any) -> None:
            with jax.named_scope("ag_finish_store"):
                slot = tile % 2
                for peer_offset in range(1, 4):
                    gather_transfer(tile, peer_offset).wait_send()
                    gather_transfer(tile, peer_offset, incoming=True).wait_recv()
                # Reuse local staging after compute; residual is original BF16.
                pltpu.async_copy(input_ref.at[pl.ds(tile * tile_rows, tile_rows), :], input_tile, local_dma_done).wait()  # type: ignore[no-untyped-call]

                def residual(k: Any, _: None) -> None:
                    input_tile[pl.ds(k * shard_rows, shard_rows), :] = (input_tile[pl.ds(k * shard_rows, shard_rows), :] + ag_inbox[slot, k]).astype(
                        jnp.bfloat16
                    )

                jax.lax.fori_loop(0, 4, residual, None)
                pltpu.async_copy(input_tile, output_ref.at[pl.ds(tile * tile_rows, tile_rows), :], local_dma_done).wait()  # type: ignore[no-untyped-call]
                # Only release credits that a future tile will consume.
                for peer_offset in range(1, 4):

                    @pl.when(tile + 2 < num_tiles)
                    def release_slot(peer_offset: int = peer_offset) -> None:
                        pl.semaphore_signal(ag_slot_free.at[slot, peer_offset - 1], 1, device_id={"tp": rank ^ peer_offset})

        # Keep this order: compute t can reuse partials from t-2 (its scatter
        # finished last iteration). Store t-2 frees the gather slot before
        # starting gather t-1. Finally, launch scatter t for the next compute.
        def pipeline_step(tile: Any, _: None) -> None:
            if overlap:

                @pl.when(tile < num_tiles)
                def compute_current() -> None:
                    compute_tile(tile)

                @pl.when(tile >= 2)
                def store_two_tiles_behind() -> None:
                    finish_gather_and_store(tile - 2)

                @pl.when((tile >= 1) & (tile < num_tiles + 1))
                def reduce_one_tile_behind() -> None:
                    reduce_and_start_gather(tile - 1)

                @pl.when(tile < num_tiles)
                def scatter_current() -> None:
                    start_scatter(tile)
            else:
                compute_tile(tile)
                start_scatter(tile)
                reduce_and_start_gather(tile)
                finish_gather_and_store(tile)

        jax.lax.fori_loop(0, num_tiles + (2 if overlap else 0), pipeline_step, None)

    return pl.pallas_call(  # type: ignore[no-any-return]
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, jnp.bfloat16),  # type: ignore[no-untyped-call]
        grid_spec=pl.GridSpec(
            in_specs=[pl.BlockSpec(memory_space=pl.ANY)] * 4,
            out_specs=pl.BlockSpec(memory_space=pl.ANY),
            scratch_shapes=[
                pltpu.VMEM((tile_rows, hidden), jnp.bfloat16),  # input_tile, also residual/output staging
                pltpu.VMEM((1, hidden), jnp.bfloat16),
                pltpu.VMEM((tile_rows, hidden), jnp.bfloat16),
                pltpu.VMEM((tile_rows, num_weight_tiles * weight_tile_width), jnp.bfloat16),
                pltpu.VMEM((2, tile_rows, hidden), jnp.float32),  # partials: [slot, row, hidden]
                pltpu.VMEM((2, 3, shard_rows, hidden), jnp.float32),  # rs_inbox: [slot, peer_offset-1, owned row, hidden]
                pltpu.VMEM((2, shard_rows, hidden), jnp.bfloat16),  # local_shards: FP32 sum, then BF16 cast
                pltpu.VMEM((2, 4, shard_rows, hidden), jnp.bfloat16),  # ag_inbox: [slot, source rank, owned row, hidden]
                *[pltpu.SemaphoreType.DMA((2, 3)) for _ in range(4)],
                *[pltpu.SemaphoreType.REGULAR((2, 3)) for _ in range(2)],
                pltpu.SemaphoreType.DMA,
            ],
        ),
        compiler_params=pltpu.CompilerParams(collective_id=0, vmem_limit_bytes=127 * 1024**2),
        name=f"clef_mlp_tp4_{'overlap' if overlap else 'serial'}_m{tile_rows}",
    )(x, norm.reshape(1, hidden), gu, down)[:num_rows]
