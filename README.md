# deciax

[![PyPI Version](https://img.shields.io/pypi/v/deciax?color=3775A9&logo=pypi&logoColor=white)](https://pypi.org/project/deciax/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sY_4BeRq_em4tzePW3poPyl-lGXyxO1g?usp=sharing)

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

## Qwen3.5 decision models: Clef and Clef Flash

`clef` (27B) and `clef-flash` (9B) use the same Qwen3.5 text backbone and Clef
joint schema head. Their HF `architectures` value is
`Qwen3_5ForConditionalGeneration`; deciax runs its language backbone plus the
checkpoint's decision head. Vision weights are skipped during device loading.

```python
import deciax

model = deciax.load("clef")  # "clef-flash" uses the same adapter
answer = model.decide(
    {"review": "The package arrived broken."},
    {
        "refund": {"type": "noul", "instructions": "Should this customer receive a refund?"},
        "urgency": {
            "type": "score",
            "instructions": "How urgent is a response?",
            "criteria": ["Low", "Medium", "High"],
        },
    },
    max_len=2048,
)
```

On a single-host v6e-4, BF16 weights use tensor parallelism across all four local
devices. There is no generation loop or persistent KV cache. Each layer executable
is reused across the model depth; the head evaluates every question jointly.
`lm_head.weight` supplies lexical vectors for option tokens only.

Clef currently supports:

- Text/JSON state with `choice`, `noul`, and `score` questions.
- Up to 16,384 **total** tokens, including the complete schema and prompt.
  State is truncated from the right; a schema that exceeds the budget is rejected.
- Up to 64 questions and 512 total options. Choice options are sorted for encoding,
  and response probabilities follow the request's order.
- `decide_batch` and both HTTP decision routes. Execution chunks contain at most
  two complete requests, with explicit sequence, field, and option masks.
- `impl="auto"` or `impl="jax"`. The current GDN implementation uses a stable
  triangular solve in 64-token chunks; attention limits score memory by query blocks.

Images, videos, `media_kwargs`, `head_max_len`, `labels`, and `option_order` are
unsupported. Multimodal inference and PPLX's noncausal decision adapter are deferred.
Clef uses the checkpoint's probability/confidence semantics, without Laya's act head
or calibration. Runtime dependencies do not include PyTorch or Transformers.

### Run on TPU

From this checkout, install with `uv sync --extra tpu`.

```sh
# Clef 27B, BF16, all four local v6e devices; XLA implementation.
# This cache is temporary and must have room for the approximately 55 GB release.
HF_HOME=/dev/shm/deciax-clef-hf uv run python -m deciax.serve --model clef --dtype bf16 --port 8000
```

```sh
# Clef 27B, text-only HTTP request.
curl -s localhost:8000/v1/systemone -H 'Content-Type: application/json' -d \
  '{"state":"The package arrived broken.","questions":{"refund":{"type":"noul","instructions":"Should this customer receive a refund?"}}}'
```
