import torch
from einops import rearrange
from .linear_mem_attn_torch import attention


class Attention(torch.nn.Module):
    """ Simple PyTorch Multihead Attention. """
    def __init__(self, dim, heads = 8, dim_head = 64, bias=False):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = torch.nn.Linear(dim, inner_dim, bias = bias)
        self.to_kv = torch.nn.Linear(dim, inner_dim * 2, bias = bias)
        self.to_out = torch.nn.Linear(inner_dim, dim)


    def forward(self, x, context, mask = None, query_chunk_size = 1024, key_chunk_size = 4096):
        """ Inputs: 
            * x, context: (b n d)
            * mask: b, n
            * query_chunk_size, key_chunk_size: see `linear_mem_attn_torch.py`
        """
        h = self.heads

        q = self.to_q(x)
        kv = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h = h), (q, *kv))

        out = attention(q, k, v, mask, query_chunk_size, key_chunk_size)

        out = rearrange(out, 'b n h d -> b n (h d)', h = h)
        return self.to_out(out)