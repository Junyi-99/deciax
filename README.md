# deciax

High-performance decision models on TPUs, powered by JAX and Pallas.

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

### HTTP server

```sh
uv run python -m deciax.serve --model laya --model laya-multilingual --default-model laya-multilingual --port 8000
curl -s localhost:8000/v1/systemone -d '{"state": "I was charged twice", "questions": {"angry": {"type": "noul", "instructions": "Is the user angry?"}}}'
```

The server speaks the same `/v1/systemone` wire protocol as `laya-serve`:
- **Routes**: `POST /v1/systemone`, `POST /v1/systemone/batch` and `GET /health`.
- **Requests and responses**: the same request limits and status codes, `x_jev_confidence` on choice and
  score answers, `--jev-strict` and Bearer auth (`DECIAX_API_KEY`).
- **Request controls**: `model`, `max_len` and `head_max_len`.

There are two differences from `laya-serve`:
- No language routing. A request with no `model`, the bundle id or an unknown id such as `jev-1` gets the
  default model.
- The controls deciax does not implement return 422 instead of being ignored: `task`, `lang`,
  `lang_guess`, `min_confidence`, `batch_size` and `sort_by_length`.

Every flag also reads an environment variable: `DECIAX_HOST`, `DECIAX_PORT`, `DECIAX_MODELS`,
`DECIAX_DEFAULT_MODEL`, `DECIAX_MAX_CONCURRENT`, `DECIAX_JEV_STRICT`, `DECIAX_DTYPE`, `DECIAX_IMPL` and
`DECIAX_LOG_LEVEL`.

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
- language routing between checkpoints
- benchmarks

Laya is Apache-2.0 (https://github.com/NandhaKishorM/laya). Code ported from it is attributed in each
module docstring.
