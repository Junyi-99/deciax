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

## Usage

```python
import deciax

decider = deciax.load("laya")  # or "laya-multilingual"; bf16 on TPU, fp32 elsewhere
decider.decide(
    {"user": "I was charged twice this month, please fix it."},
    {
        "intent": {
            "type": "choice",
            "instructions": "What does the user want?",
            "criteria": {"billing": "payments, refunds", "other": "everything else"},
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["not urgent", "soon", "blocking"],
        },
        "angry": {"type": "noul", "instructions": "Is the user angry?"},
    },
)
```

The response has the same shape as Laya's `system_one` response. Install with `uv sync --extra tpu` on a TPU host.

## Status

The Laya inference path works end to end. The runtime needs only JAX and `tokenizers`; it does not need
torch, transformers or laya. Parity against PyTorch Laya 0.4.4 (`tests/laya/`):
- **Inputs**: tokenizer ids and `input_ids` / `marker_pos` match token for token.
- **CPU fp32**: option and act logits are within 1e-3 (act logits within 1e-5 relative). Responses match
  up to the last rounded digit.
- **TPU bf16**: probabilities are within 0.02 and the argmax matches, with both the `jax` and `pallas`
  attention.

`impl="auto"` uses XLA attention, which is faster than the Pallas flash kernel at these shapes.

Not yet supported:
- the parallel option layout
- per-language temperatures and binning maps
- `predict_long` windowing
- tournament selection
- the HTTP server (`deciax.serve`)
- benchmarks

Laya is Apache-2.0 (https://github.com/NandhaKishorM/laya). Code ported from it is attributed in each
module docstring.
