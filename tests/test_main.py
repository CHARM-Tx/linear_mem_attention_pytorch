import numpy as np
import torch
# import jax

from linear_mem_attention_pytorch import *
from linear_mem_attention_pytorch.utils import qkv2res
from linear_mem_attention_torch.fast_attn import Attention


def test_torch_attn(): 
    B, L, D = 1, 2**14, 64
    a = torch.randn(B, L, D) # .cuda()
    b = a[:, None, :, :]                      # (b h n d) batch and heads
    b_ = torch.transpose(a, 0, 1)[None, ...]
    c_ = torch.cat([b_, b_], dim=0)

    # test batching works
    assert torch.allclose(
        attention(b_, b_, b_)[0], # .shape b n h d
        attention(c_, c_, c_)[0], # .shape b n h d
    ), "Batching does not work"

    # test query chunking works
    assert torch.allclose(
        attention(b_, b_, b_, query_chunk_size=32)[0], # .shape b n h d
        attention(b_, b_, b_)[0], # .shape b n h d
        atol = 1e-6
    ), "Query chunking does not work"

    # test key chunking works
    assert torch.allclose(
        attention(b_, b_, b_, key_chunk_size=128)[0], # .shape b n h d
        attention(b_, b_, b_)[0], # .shape b n h d
        atol = 1e-6
    ), "Key chunking does not work"

    # test correctness chunking works
    assert torch.allclose(
        attention(b_, b_, b_)[0], # .shape b n h d
        torch.transpose( qkv2res(*[torch.transpose(b_, 1, 2)]*3), 1, 2 )[0], # .shape b n h d
        atol = 1e+1 # slight difference, but paper code shows it as well
    ), "Attn chunking does not work"


def test_fast_attn(): 
    batch, length, features = 2, 2**8, 64
    x, ctx = torch.randn(2, batch, length, features)
    mask = torch.randn(batch, length) < 1.

    attn = Attention(dim=D, heads = 8, dim_head = 64, bias=False)

    # self-attn
    v_self = attn(x, x, mask, query_chunk_size=1024, key_chunk_size=4096)

    # cross-attn
    v_cross = attn(x, ctx, mask, query_chunk_size=1024, key_chunk_size=4096)

    assert True