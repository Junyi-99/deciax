"""`model.safetensors` -> parameter pytree.

The checkpoint's keys match the PyTorch `state_dict` (encoder.*, head.layers.*,
type_emb, scorer, act_head), so the mapping is close to identity. Matrices are
stored in the matmul dtype, vectors (biases, norm weights) in fp32. The
`temperature` buffer is not used: calibration comes from rl_agent_config.json.
"""

from pathlib import Path

import jax
import jax.numpy as jnp


def load_params(path: Path, dtype: jnp.dtype = jnp.bfloat16) -> dict[str, jax.Array]:
    raise NotImplementedError
