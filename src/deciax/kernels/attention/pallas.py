"""TPU attention via the Pallas flash kernel with an additive mask bias.

Port of the flash path of laya-tpu `scripts/jax_model.py`. The sequence is padded to
a multiple of the block size; padded keys are masked and padded queries dropped.
"""

import jax
import jax.numpy as jnp

from deciax.kernels.attention.reference import MASK_FILL, attention_mask

BLOCK = 128


def attention(
    q: jax.Array, k: jax.Array, v: jax.Array, mask: jax.Array, window: int = 0
) -> jax.Array:
    """[B, S, H, Dh] -> [B, S, H, Dh] (fp32)."""
    from jax.experimental.pallas.ops.tpu.flash_attention import (
        BlockSizes,
        flash_attention,
    )

    batch, seq, heads, dh = q.shape
    padded = -(-seq // BLOCK) * BLOCK
    if padded != seq:
        widths = ((0, 0), (0, padded - seq), (0, 0), (0, 0))
        q, k, v = (jnp.pad(x, widths) for x in (q, k, v))
        mask = jnp.pad(mask, ((0, 0), (0, padded - seq)))
    bias = jnp.where(attention_mask(mask, window), 0.0, MASK_FILL)
    bias = jnp.broadcast_to(bias, (batch, heads, padded, padded)).astype(jnp.float32)
    out: jax.Array = flash_attention(
        *(x.transpose(0, 2, 1, 3) for x in (q, k, v)),
        ab=bias,
        sm_scale=dh**-0.5,
        causal=False,
        block_sizes=BlockSizes(
            block_q=BLOCK, block_k_major=BLOCK, block_k=BLOCK, block_b=1
        ),
    )
    return out.transpose(0, 2, 1, 3)[:, :seq].astype(jnp.float32)
