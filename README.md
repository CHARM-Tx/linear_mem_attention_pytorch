# Attention for PyTorch with Linear Memory Footprint

Unofficially implements https://arxiv.org/abs/2112.05682 to get **Linear Memory Cost on Attention** (+ some sidekick speedup on the GPU when compared to reference implementation in `JAX`)

### Usage: 

```
git clone https://github.com/CHARM-Tx/linear_mem_attention_pytorch
cd linear_mem_attention_pytorch
python setup.py install 
```

## Usage:

### High Level

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

### Low level

```python
from linear_mem_attention_torch import attention

batch, length, heads, features = 2, 2**8, 8, 64
mask = torch.randn(batch, length) < 1.
q, k, v = torch.randn(3, batch, length, heads, features)

v_ = attention(q, k, v, mask, query_chunk_size=1024, key_chunk_size=4096)
```


## Benchmarks

* soon: provide `seq_len`, `time_cpu_jax`, `time_cpu_torch`, `time_cpu_base`, `time_gpu_jax`, `time_gpu_torch`, `time_gpu_base`
	* uses `torch.einsum` for `bihd,bjhd->bihj` and `bihj,bjhd->bihd` as a reference baseline

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
