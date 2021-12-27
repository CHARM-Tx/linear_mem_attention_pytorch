# code from: https://arxiv.org/abs/2112.05682
import functools, jax, math
from jax import numpy as jnp


def _query_chunk_attention(query, key, value, precision, key_chunk_size=4096):
    """Multi-head dot product attention with a limited number of queries."""
    num_kv, num_heads, k_features = key.shape
    v_features = value.shape[-1]
    key_chunk_size = min(key_chunk_size, num_kv)
    query = query / jnp.sqrt(k_features)

    @functools.partial(jax.checkpoint, prevent_cse=False)
    def summarize_chunk(query, key, value):
        attn_weights = jnp.einsum("qhd,khd->qhk", query, key, precision=precision)
        max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
        max_score = jax.lax.stop_gradient(max_score)
        exp_weights = jnp.exp(attn_weights - max_score)
        exp_values = jnp.einsum("vhf,qhv->qhf", value, exp_weights, precision=precision)
        return (
            exp_values,
            exp_weights.sum(axis=-1),
            max_score.reshape((query.shape[0], num_heads)),
        )

    def chunk_scanner(chunk_idx):
        key_chunk = jax.lax.dynamic_slice(
            key, (chunk_idx, 0, 0), slice_sizes=(key_chunk_size, num_heads, k_features)
        )
        value_chunk = jax.lax.dynamic_slice(
            value,
            (chunk_idx, 0, 0),
            slice_sizes=(key_chunk_size, num_heads, v_features),
        )
        return summarize_chunk(query, key_chunk, value_chunk)

    chunk_values, chunk_weights, chunk_max = jax.lax.map(
        chunk_scanner, xs=jnp.arange(0, num_kv, key_chunk_size)
    )

    global_max = jnp.max(chunk_max, axis=0, keepdims=True)
    max_diffs = jnp.exp(chunk_max - global_max)
    chunk_values *= jnp.expand_dims(max_diffs, axis=-1)
    chunk_weights *= max_diffs

    all_values = chunk_values.sum(axis=0)
    all_weights = jnp.expand_dims(chunk_weights, -1).sum(axis=0)
    return all_values / all_weights


def attention(
    query, key, value, precision=jax.lax.Precision.HIGHEST, query_chunk_size=1024
):
    """Memory-efficient multi-head dot product attention."""
    num_q, num_heads, q_features = query.shape

    def chunk_scanner(chunk_idx, _):
        query_chunk = jax.lax.dynamic_slice(
            query,
            (chunk_idx, 0, 0),
            slice_sizes=(min(query_chunk_size, num_q), num_heads, q_features),
        )
        return (
            chunk_idx + query_chunk_size,
            _query_chunk_attention(query_chunk, key, value, precision=precision),
        )

    _, res = jax.lax.scan(
        chunk_scanner, init=0, xs=None, length=math.ceil(num_q / query_chunk_size)
    )
    return res.reshape(num_q, num_heads, value.shape[-1])
