"""Backend selection."""

from __future__ import annotations

from typing import Literal

import jax

type Impl = Literal["jax", "pallas"]
type ImplChoice = Literal["auto", "jax", "pallas"]  # what callers pass; `resolve` turns "auto" into an `Impl`


def resolve(impl: ImplChoice) -> Impl:
    if impl == "auto":
        return "pallas" if jax.default_backend() == "tpu" else "jax"
    if impl in ("jax", "pallas"):
        return impl
    raise ValueError(f"impl must be 'auto', 'jax' or 'pallas', got {impl!r}")
