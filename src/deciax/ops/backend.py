"""Backend selection."""

from __future__ import annotations

from typing import Literal

type Impl = Literal["jax", "pallas"]


def default_impl() -> Impl:
    """The fused Pallas kernels (`deciax.kernels`) on TPU, the XLA path elsewhere.

    On a v6e the fused path is 2.5x faster than XLA at laya's shapes (64 x 256: 27 vs 70 ms) and
    moves ~17x fewer HBM bytes; the kernels are TPU-only (Mosaic).
    """
    import jax

    return "pallas" if jax.default_backend() == "tpu" else "jax"


def resolve(impl: str) -> Impl:
    if impl == "auto":
        return default_impl()
    if impl in ("jax", "pallas"):
        return impl  # type: ignore[return-value]
    raise ValueError(f"impl must be 'auto', 'jax' or 'pallas', got {impl!r}")
