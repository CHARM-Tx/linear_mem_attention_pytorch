# Attention for PyTorch with Linear Memory Footprint

Implements https://arxiv.org/abs/2112.05682 to get linear memory cost on attention


### Usage:

#### High Level

```python
from linear_mem_attention_torch.fast_attn import Attention

batch, length, features = 2, 2**8, 64
x, ctx = torch.randn(2, batch, length, features)
mask = torch.randn(batch, length) < 1.

attn = Attention(dim=features, heads = 8, dim_head = 64, bias=False)

# self-attn
v_self = attn(x, x, mask, query_chunk_size=1024, key_chunk_size=4096)

# cross-attn
v_cross = attn(x, ctx, mask, query_chunk_size=1024, key_chunk_size=4096)
```

#### Low level

```python
from linear_mem_attention_torch import attention

batch, length, heads, features = 2, 2**8, 8, 64
mask = torch.randn(batch, length) < 1.
q, k, v = torch.randn(3, batch, length, heads, features)

v_ = attention(q, k, v, mask, query_chunk_size=1024, key_chunk_size=4096)
```


## Citations:

```bibtex
@misc{rabe2021selfattention,
      title={Self-attention Does Not Need $O(n^2)$ Memory}, 
      author={Markus N. Rabe and Charles Staats},
      year={2021},
      eprint={2112.05682},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
