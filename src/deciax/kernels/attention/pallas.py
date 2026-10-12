"""TPU attention via Pallas: first the flash_attention + bias path that laya-tpu
already validated (scripts/jax_model.py), then splash attention with local masks."""

import jax


def attention(
    q: jax.Array, k: jax.Array, v: jax.Array, mask: jax.Array, window: int = 0
) -> jax.Array:
    raise NotImplementedError
