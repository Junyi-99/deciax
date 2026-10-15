# HTTP deployment and compatibility

Start one or more models; `--default-model` selects the model for requests that omit `model`:

```sh
python -m deciax.serve --model laya --model laya-multilingual --default-model laya-multilingual --port 8000
```

## Authentication

Set `DECIAX_API_KEY` on the server and send `Authorization: Bearer <key>` with requests.
Without this variable, the server does not require authentication.

## Routes

- `POST /v1/systemone`: send `state` and `questions`.
- `POST /v1/systemone/batch`: send a nonempty `states` list and shared `questions`.
  Returns `results` in input order and `total_usage`.
- `GET /health`: server health; model details require authentication when an API key is configured.

## Model routing

The request's `model` selects a loaded checkpoint. Omit it to use the server default.
An unknown bare name such as `jev-1` also selects the default; a known but unloaded
model, a path, or an unknown Hub repo ID returns 422. There is no automatic language routing.

## Configuration

Run `python -m deciax.serve --help` for command-line options.
Environment variables provide defaults:

| Flag | Environment variable |
|---|---|
| `--host` | `DECIAX_HOST` |
| `--port` | `DECIAX_PORT` |
| `--model` (repeatable) | `DECIAX_MODELS` (comma-separated) |
| `--default-model` | `DECIAX_DEFAULT_MODEL` |
| `--max-concurrent` | `DECIAX_MAX_CONCURRENT` |
| `--jev-strict` | `DECIAX_JEV_STRICT` |
| `--dtype` | `DECIAX_DTYPE` |
| `--impl` | `DECIAX_IMPL` |
| `--log-level` | `DECIAX_LOG_LEVEL` |

## Laya compatibility

The Laya adapter follows `laya-serve` request limits, status codes, and response shape,
including `x_jev_confidence` on choice/score answers. `--jev-strict` selects the strict
Jev response fields. Requests support `model`, `max_len`, and `head_max_len`.
Clef has its own [limits](../README.md#clef-limits), and does not support `head_max_len`.

Unlike `laya-serve`, deciax rejects `task`, `lang`, `lang_guess`, and `min_confidence`
with 422. Batch requests also reject `batch_size`, `sort_by_length`, and `diagnostics`.
Images, videos, and `media_kwargs` are unsupported. Request hooks are refused.
