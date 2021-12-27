# adapted to torch from: https://arxiv.org/abs/2112.05682
import math
import torch
from torch.utils import checkpoint
from typing import Tuple, Optional

from .utils import dynamic_slice, torch_map, torch_scan


def query_chunk_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    key_chunk_size: int = 4096,
) -> torch.Tensor:
    """Multi-head dot product attention with a limited number of queries."""
    device, dtype = query.device, query.dtype
    batch, num_kv, num_heads, k_features = key.shape
    v_features = value.shape[-1]
    query_chunk = query.shape[1]  # b n h d
    key_chunk_size = min(key_chunk_size, num_kv)
    query = query / k_features ** 0.5

    # @functools.partial(jax.checkpoint, prevent_cse=False)
    def summarize_chunk(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        attn_weights = torch.einsum("bqhd,bkhd->bqhk", query, key)
        if mask is not None:
            max_neg = -torch.finfo(attn_weights.dtype).max
            attn_weights.masked_fill_(~mask[:, None, None, :], max_neg)

        max_score = torch.amax(attn_weights, dim=-1, keepdim=True).detach()
        exp_weights = torch.exp(attn_weights - max_score)
        exp_values = torch.einsum("bvhf,bqhv->bqhf", value, exp_weights)
        # (b q h f), (b q h), (b q h 1)
        return exp_values, exp_weights.sum(dim=-1), max_score

    def chunk_scanner(
        chunk_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key_chunk = dynamic_slice(
            key, (chunk_idx, 0, 0), slice_sizes=(key_chunk_size, num_heads, k_features)
        )
        value_chunk = dynamic_slice(
            value,
            (chunk_idx, 0, 0),
            slice_sizes=(key_chunk_size, num_heads, v_features),
        )
        mask_chunk = (
            mask[:, chunk_idx : chunk_idx + key_chunk_size]
            if mask is not None
            else None
        )

        return checkpoint.checkpoint(
            summarize_chunk, query, key_chunk, value_chunk, mask_chunk
        )

    chunk_iter = list(range(0, num_kv, key_chunk_size))
    chunk_values = torch.zeros(
        num_kv // key_chunk_size,
        batch,
        query_chunk,
        num_heads,
        v_features,
        dtype=dtype,
        device=device,
    )
    chunk_weights = torch.zeros(
        num_kv // key_chunk_size,
        batch,
        query_chunk,
        num_heads,
        dtype=dtype,
        device=device,
    )
    chunk_max = torch.zeros(
        num_kv // key_chunk_size,
        batch,
        query_chunk,
        num_heads,
        1,
        dtype=dtype,
        device=device,
    )
    for i, xi in enumerate(chunk_iter):
        chunk_values[i], chunk_weights[i], chunk_max[i] = chunk_scanner(xi)

    global_max = torch.amax(chunk_max, dim=0, keepdim=True)
    max_diffs = torch.exp(chunk_max - global_max)

    chunk_values *= max_diffs
    chunk_weights *= max_diffs[..., 0]

    all_values = chunk_values.sum(dim=0)
    all_weights = torch.unsqueeze(chunk_weights, -1).sum(dim=0)
    return all_values / all_weights


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    query_chunk_size: int = 1024,
    key_chunk_size: int = 4096,
) -> torch.Tensor:
    """Memory-efficient multi-head dot product attention.
    Inputs:
    * q, k, v: (b n h d) torch tensors
    * mask: (b n)
    * query_chunk_size: int.
    * key_chunk_size: int.
    Outputs: (b n h d) torch tensor (qk-weighted sum of v)
    """
    batch, num_q, num_heads, q_features = query.shape

    def chunk_scanner(chunk_idx: int, _):
        query_chunk = dynamic_slice(
            query,
            (chunk_idx, 0, 0),
            slice_sizes=(min(query_chunk_size, num_q), num_heads, q_features),
        )
        return (
            chunk_idx + query_chunk_size,
            query_chunk_attention(
                query_chunk, key, value, mask, key_chunk_size=key_chunk_size
            ),
        )

    _, res = torch_scan(
        chunk_scanner, init=0, xs=None, length=math.ceil(num_q / query_chunk_size)
    )
    return res.reshape(batch, num_q, num_heads, value.shape[-1])
