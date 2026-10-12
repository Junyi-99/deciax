"""`model.safetensors` -> parameter dict.

The checkpoint's keys match the PyTorch `state_dict` (encoder.*, head.layers.*,
type_emb, scorer, act_head). Matrices are stored in the matmul dtype, vectors
(biases, norm weights) in fp32. The `temperature` buffer is unused: calibration
comes from rl_agent_config.json.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from safetensors import safe_open

SKIP = {"temperature"}


def load_params(path: Path, dtype: jnp.dtype) -> dict[str, jax.Array]:
    params = {}
    with safe_open(str(path), framework="np") as f:
        for key in f.keys():  # noqa: SIM118 -- safe_open is not a mapping
            if key in SKIP:
                continue

            value = np.asarray(f.get_tensor(key), np.float32)
            params[key] = jnp.asarray(value, dtype if value.ndim > 1 else jnp.float32)

    return params
