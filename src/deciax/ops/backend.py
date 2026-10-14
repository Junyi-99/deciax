"""Backend selection."""

from __future__ import annotations

from typing import Literal

type Impl = Literal["jax", "pallas"]
type ImplChoice = Literal["auto", "jax", "pallas"]  # what callers pass; `resolve` turns "auto" into an `Impl`


def default_impl() -> Impl:
    import jax

    return "pallas" if jax.default_backend() == "tpu" else "jax"


def resolve(impl: ImplChoice) -> Impl:
    if impl == "auto":
        return default_impl()
    if impl in ("jax", "pallas"):
        return impl
    raise ValueError(f"impl must be 'auto', 'jax' or 'pallas', got {impl!r}")
