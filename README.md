# deciax

High-performance decision models on TPUs, powered by JAX and Pallas.

## Install

```bash
pip install deciax
```

On TPU, `pip install "deciax[tpu]"` pulls in `jax[tpu]`.

## Usage

```python
import deciax

decider = deciax.load("laya")  # or "laya-multilingual"; bf16 on TPU, fp32 elsewhere
decider.decide(
    {
        "user": "I was charged twice this month, please fix it.",
    },
    {
        "intent": {
            "type": "choice",
            "instructions": "What does the user want?",
            "criteria": {
                "billing": "payments, refunds",
                "other": "everything else",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": [
                "not urgent",
                "soon",
                "blocking",
            ],
        },
        "angry": {
            "type": "noul",
            "instructions": "Is the user angry?",
        },
    },
)
```

The response has the same shape as Laya's `system_one` response. Install with `uv sync --extra tpu` on a TPU host.

### HTTP server

```sh
uv run python -m deciax.serve --model laya --model laya-multilingual --default-model laya-multilingual --port 8000
curl -s localhost:8000/v1/systemone -d '{"state": "I was charged twice", "questions": {"angry": {"type": "noul", "instructions": "Is the user angry?"}}}'
```

Compared with `laya-serve`:

| | laya-serve | deciax |
|---|---|---|
| Routes `POST /v1/systemone`, `POST /v1/systemone/batch`, `GET /health` | same | same |
| Request limits, status codes, `x_jev_confidence` on choice/score answers | same | same |
| `--jev-strict`, Bearer auth (`DECIAX_API_KEY`) | same | same |
| Controls `model`, `max_len`, `head_max_len` | same | same |
| Language routing | routes by language | none; missing/unknown `model` (e.g. `jev-1`) gets the default model |
| Controls `task`, `lang`, `lang_guess`, `min_confidence`, `batch_size`, `sort_by_length` | accepted | 422 |

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

`impl="auto"` still uses the XLA path; `impl="pallas"` runs the fused TPU kernels (`src/deciax/kernels/`):

| kernel | per encoder layer | what stays in VMEM |
|---|---|---|
| `fused.norm_linear` | fp32 residual -> LayerNorm -> bf16 @ Wqkv (+ bias) -> RoPE -> bf16 | normalized rows, the whole weight |
| `attention.pallas.flash_attention` | per query block x all heads, online softmax over key blocks; key mask and sliding window computed in-kernel | scores, probabilities, running max / sum / accumulator |
| `fused.out_mlp` | att @ Wo + residual -> LayerNorm -> Wi -> gated GELU (ReLU in the heads) -> Wo2 + residual -> fp32 | every MLP intermediate, all three weights |
| `fused.layer` (S <= 512) | the three above in one kernel: a grid step holds whole sequences, attention is a single-pass softmax per sequence and head | everything but x in / x' out and the weights |

On a TPU v6e at laya's 50-question shape (64 rows x 256 tokens) the XLA path moves ~90 GB through HBM per
call (every op HBM-bound) and takes 70 ms; the fused path moves ~5 GB and takes 27 ms (multilingual: 36 -> 14 ms),
1.23x more at 128 x 512 where the one-kernel layer replaces the blocked attention. Parity with the XLA path: argmax identical, max |delta p| 0.006 on the fixture cases;
`tests/laya/test_parity.py::test_bf16_tpu` holds for both.

Not yet supported:
- the parallel option layout
- per-language temperatures and binning maps
- `predict_long` windowing
- tournament selection
- language routing between checkpoints
- benchmarks

Laya is Apache-2.0 (https://github.com/NandhaKishorM/laya). Code ported from it is attributed in each
module docstring.
