# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Modified for functional JAX inference. Licensed under Apache-2.0; see LICENSES/Apache-2.0.txt.

"""Gated delta rule, ported from Transformers 5.10.2 Qwen3.5 (Apache-2.0).

The chunked algorithm uses float32 state and bounded 64-token triangular solves.
The token recurrence remains an independent correctness reference.
"""

import jax
import jax.numpy as jnp


def l2norm(x: jax.Array) -> jax.Array:
    return (x.astype(jnp.float32) * jax.lax.rsqrt(jnp.sum(x.astype(jnp.float32) ** 2, -1, keepdims=True) + 1e-6)).astype(x.dtype)


def recurrent(q: jax.Array, k: jax.Array, v: jax.Array, g: jax.Array, beta: jax.Array) -> jax.Array:
    dtype = q.dtype
    q, k = l2norm(q).astype(jnp.float32) * q.shape[-1] ** -0.5, l2norm(k).astype(jnp.float32)
    state = jnp.zeros((q.shape[0], q.shape[2], q.shape[3], v.shape[3]), jnp.float32)

    def step(state: jax.Array, xs: tuple[jax.Array, ...]) -> tuple[jax.Array, jax.Array]:
        qt, kt, vt, gt, bt = xs
        state = state * jnp.exp(gt)[..., None, None]
        delta = (vt - jnp.einsum("bhkv,bhk->bhv", state, kt)) * bt[..., None]
        state = state + kt[..., None] * delta[..., None, :]
        return state, jnp.einsum("bhkv,bhk->bhv", state, qt)

    _, out = jax.lax.scan(step, state, tuple(x.astype(jnp.float32).swapaxes(0, 1) for x in (q, k, v, g, beta)))
    return jnp.asarray(out).swapaxes(0, 1).astype(dtype)


def chunked(q: jax.Array, k: jax.Array, v: jax.Array, g: jax.Array, beta: jax.Array, chunk_size: int = 64) -> jax.Array:
    dtype, length = q.dtype, q.shape[1]
    q, k = l2norm(q).astype(jnp.float32) * q.shape[-1] ** -0.5, l2norm(k).astype(jnp.float32)
    pad = (-length) % chunk_size

    def chunks(x: jax.Array) -> jax.Array:
        x = x.astype(jnp.float32).swapaxes(1, 2)
        pads = [(0, 0)] * x.ndim
        pads[2] = (0, pad)
        x = jnp.pad(x, pads)
        return x.reshape(*x.shape[:2], -1, chunk_size, *x.shape[3:]).swapaxes(0, 2)

    qs, ks, vs, gs, bs = (chunks(x) for x in (q, k, v, g, beta))
    eye = jnp.eye(chunk_size, dtype=jnp.float32)
    lower = jnp.tril(jnp.ones((chunk_size, chunk_size), bool))
    strict = jnp.tril(lower, -1)
    state = jnp.zeros((q.shape[2], q.shape[0], q.shape[3], v.shape[3]), jnp.float32)

    def mm(a: jax.Array, b: jax.Array) -> jax.Array:
        return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)

    def step(state: jax.Array, xs: tuple[jax.Array, ...]) -> tuple[jax.Array, jax.Array]:
        qi, ki, vi, gi, bi = xs  # [H,B,C,D], [H,B,C]
        gi = jnp.cumsum(gi, axis=-1)
        decay = jnp.exp(jnp.where(lower, gi[..., :, None] - gi[..., None, :], -jnp.inf))
        kb = ki * bi[..., None]
        a = jnp.where(strict, -mm(kb, ki.swapaxes(-1, -2)) * decay, 0)
        # Solve the triangular system directly. A truncated power series can
        # catastrophically cancel when keys align, beta approaches one, and
        # decay approaches zero (observed in Clef layer 44).
        system = eye - a
        inverse = jax.lax.linalg.triangular_solve(
            system,
            jnp.broadcast_to(eye, system.shape),
            left_side=True,
            lower=True,
            unit_diagonal=True,
        )
        value = mm(inverse, vi * bi[..., None])
        cumulative_key = mm(inverse, kb * jnp.exp(gi)[..., None])
        value = value - mm(cumulative_key, state)
        out = mm(qi * jnp.exp(gi)[..., None], state) + mm(mm(qi, ki.swapaxes(-1, -2)) * decay, value)
        next_state = state * jnp.exp(gi[..., -1, None, None]) + mm((ki * jnp.exp(gi[..., -1, None] - gi)[..., None]).swapaxes(-1, -2), value)
        return next_state, out

    _, output = jax.lax.scan(step, state, (qs, ks, vs, gs, bs))
    return jnp.asarray(output).swapaxes(0, 2).reshape(q.shape[0], q.shape[2], -1, v.shape[-1]).swapaxes(1, 2)[:, :length].astype(dtype)
