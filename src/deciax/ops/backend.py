"""Backend selection."""

from __future__ import annotations

from typing import Literal

type Impl = Literal["jax", "pallas"]


def default_impl() -> Impl:
    # XLA's fused attention beats the Pallas flash kernel with a dense bias on TPU at
    # Laya's shapes (laya-tpu fusion_bench, english bf16: 2.0 vs 5.1 ms at S=512), so
    # Pallas is opt-in until a splash kernel with local masks is faster.
    return "jax"


def resolve(impl: str) -> Impl:
    if impl == "auto":
        return default_impl()
    if impl in ("jax", "pallas"):
        return impl  # type: ignore[return-value]
    raise ValueError(f"impl must be 'auto', 'jax' or 'pallas', got {impl!r}")
