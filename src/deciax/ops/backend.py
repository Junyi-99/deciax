"""Backend selection."""

from __future__ import annotations

from typing import Literal

type Impl = Literal["jax", "pallas"]


def default_impl() -> Impl:
    import jax

    return "pallas" if jax.default_backend() == "tpu" else "jax"


def resolve(impl: str) -> Impl:
    if impl == "auto":
        return default_impl()
    if impl in ("jax", "pallas"):
        return impl  # type: ignore[return-value]
    raise ValueError(f"impl must be 'auto', 'jax' or 'pallas', got {impl!r}")
