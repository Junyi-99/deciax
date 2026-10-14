"""python -m deciax.serve

Every option falls back to an environment variable (DECIAX_HOST, DECIAX_PORT, ...).
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import get_args

from deciax.decider import DtypeChoice
from deciax.models import ModelName
from deciax.ops.backend import ImplChoice
from deciax.serve.app import serve

MODEL_NAMES: tuple[ModelName, ...] = get_args(ModelName.__value__)
DTYPES: tuple[DtypeChoice, ...] = get_args(DtypeChoice.__value__)
IMPLS: tuple[ImplChoice, ...] = get_args(ImplChoice.__value__)


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _choice[T: str](parser: argparse.ArgumentParser, flag: str, value: str, choices: tuple[T, ...]) -> T:
    """argparse checks `choices` on the command line but not on env-var defaults."""
    if value not in choices:
        parser.error(f"{flag}: invalid choice {value!r} (choose from {', '.join(choices)})")
    return value


def main() -> None:
    env = os.environ.get
    parser = argparse.ArgumentParser(prog="python -m deciax.serve", description=__doc__)
    parser.add_argument("--host", default=env("DECIAX_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(env("DECIAX_PORT", "8000")))
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        choices=MODEL_NAMES,
        help="checkpoint to load (repeatable; default DECIAX_MODELS or laya)",
    )
    parser.add_argument("--default-model", default=env("DECIAX_DEFAULT_MODEL"), choices=MODEL_NAMES)
    parser.add_argument("--max-concurrent", type=int, default=int(env("DECIAX_MAX_CONCURRENT", "16")))
    parser.add_argument("--jev-strict", action="store_true", default=_env_bool("DECIAX_JEV_STRICT"))
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--dtype", default=env("DECIAX_DTYPE", "auto"), choices=DTYPES)
    parser.add_argument("--impl", default=env("DECIAX_IMPL", "auto"), choices=IMPLS)
    parser.add_argument("--log-level", default=env("DECIAX_LOG_LEVEL", "info"))
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    models = args.models or [m.strip() for m in env("DECIAX_MODELS", "laya").split(",") if m.strip()]
    models = [_choice(parser, "--model", m, MODEL_NAMES) for m in models]
    default_model = args.default_model and _choice(parser, "--default-model", args.default_model, MODEL_NAMES)

    serve(
        args.host,
        args.port,
        tuple(models),
        default_model=default_model or None,
        api_key=env("DECIAX_API_KEY") or None,
        max_concurrent=args.max_concurrent,
        jev_strict=args.jev_strict,
        warmup=not args.no_warmup,
        dtype=_choice(parser, "--dtype", args.dtype, DTYPES),
        impl=_choice(parser, "--impl", args.impl, IMPLS),
    )


if __name__ == "__main__":
    main()
