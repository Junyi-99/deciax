# deciax

[![PyPI Version](https://img.shields.io/pypi/v/deciax?color=3775A9&logo=pypi&logoColor=white)](https://pypi.org/project/deciax/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sY_4BeRq_em4tzePW3poPyl-lGXyxO1g?usp=sharing)

High-performance decision models on TPUs, powered by JAX and Pallas.

## Install

```sh
pip install deciax          # CPU
pip install "deciax[tpu]"   # TPU
```

## Quick start

```python
import deciax

decider = deciax.load("laya")
result = decider.decide(
    {
        "user": "I was charged twice this month, please fix it.",
    },
    {
        "intent": {
            "type": "choice",
            "instructions": "What does the user want?",
            "criteria": {"billing": "payments, refunds", "other": "everything else"},
        },
    },
)
print(result["answers"]["intent"]["choice"])  # e.g. billing
```

Replace `"laya"` with `"laya-multilingual"`, `"clef-flash"` (9B), or `"clef"` (27B).
Weights download on first use. The default dtype is BF16 on TPU and FP32 elsewhere.

Each question has an `instructions` string and one of these types:

| Type | `criteria` | Answer |
|---|---|---|
| `choice` | `{"billing": "payments, refunds", "other": "everything else"}` | Selected key and probabilities |
| `score` | `["Low", "Medium", "High"]` | Expected level index, starting at 0 |
| `noul` | Optional | Probability of true, from 0 to 1 |

Pass multiple named questions in the same dictionary. To evaluate multiple states,
use `decider.decide_batch(states, questions)`.

## HTTP server

```sh
python -m deciax.serve --model laya
```

In another terminal:

```sh
curl http://localhost:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"I was charged twice","questions":{"refund":{"type":"noul","instructions":"Should this customer receive a refund?"}}}'
```

For options, run `python -m deciax.serve --help`.
See [HTTP deployment and compatibility](docs/serving.md) for authentication, batching, and model routing.

## Clef limits

- Text/JSON only; up to 64 questions and 512 total options per request.
- `max_len` sets the total token budget (default and maximum: 16,384), including
  the prompt and schema. Excess state is truncated from the right; an oversized schema is rejected.
- `head_max_len`, `labels`, `option_order`, images, videos, and `media_kwargs` are unsupported.
- Use the default `impl="auto"` or `impl="jax"`; `impl="pallas"` is unsupported.
