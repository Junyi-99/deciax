"""Backend selection. Detection is deferred to first use so importing deciax (and
CPU-only tests) never initialises the TPU."""

import functools
from typing import Literal

import jax

type Impl = Literal["jax", "pallas"]


@functools.cache
def default_impl() -> Impl:
    return "pallas" if jax.default_backend() == "tpu" else "jax"


def resolve(impl: str) -> Impl:
    if impl == "auto":
        return default_impl()
    if impl in ("jax", "pallas"):
        return impl  # type: ignore[return-value]
    raise ValueError(f"impl must be 'auto', 'jax' or 'pallas', got {impl!r}")
