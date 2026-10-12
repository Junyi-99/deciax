# deciax

High-performance decision models on TPUs, powered by JAX and Pallas.

## Development

Install the development tools and Git hooks:

```sh
uv sync
uv run pre-commit install
```

Run all checks manually with `uv run pre-commit run --all-files`.

Run the tests and type checks:

```sh
uv run pytest            # TPU-only tests are skipped off-TPU
uv run mypy
```

Parity fixtures come from the PyTorch reference, Laya 0.4.4 at commit `682ac67`. Install it with
`uv sync --group reference` and run `scripts/export_reference.py`.

## Layout

| Path | Contents |
|---|---|
| `src/deciax/protocol/` | Model-agnostic question/answer types and the `/v1/systemone` schema. Pure Python. |
| `src/deciax/models/` | Checkpoint registry, the `DecisionModel` interface, the ModernBERT encoder. |
| `src/deciax/models/laya/` | Laya: config, tokenizer, sequence construction, heads, calibration, weights. |
| `src/deciax/nn.py` | Backend-agnostic primitives (linear, layer norm, RoPE). |
| `src/deciax/ops/` | The only place the backend forks (`jax` reference or `pallas`). |
| `src/deciax/kernels/` | Attention: pure-jnp reference and the Pallas TPU kernel. |
| `src/deciax/runtime/` | Shape buckets and the jit engine. |
| `src/deciax/serve/` | `python -m deciax.serve`: stdlib HTTP server. |

## Status

This is a skeleton. Implemented and tested so far:
- Laya temperature calibration
- `nn` primitives
- reference attention
- shape buckets
- checkpoint registry

Every other module defines its interface and raises `NotImplementedError`.

Laya is Apache-2.0 (https://github.com/NandhaKishorM/laya). Code ported from it is attributed in each
module docstring.
