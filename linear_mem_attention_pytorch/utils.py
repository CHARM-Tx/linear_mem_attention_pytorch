import torch
from typing import Optional, Tuple, Any, List
from types import FunctionType


@torch.jit.script
def qkv2res(q, k, v): 
    """ Inputs must be in (b n h d) format. """  
    # return (q @ torch.transpose(k, -1, -2)).softmax(dim=-1) @ v
    qk = torch.einsum('b i h d, b j h d -> b i h j', q, k).softmax(dim=-1)
    return torch.einsum('b i h j, b j h d -> b i h d', qk, v)


@torch.jit.script
def dynamic_slice(
    x: torch.Tensor, 
    slices: Tuple[int, int, int], 
    slice_sizes: Tuple[int, int, int],
) -> torch.Tensor:
    """ approx like jax.lax.dynamic_slice.
        * NOTE: assumes we dont work on first dim
        Ex: 
        dynamic_slice(
            x, 
            slices=(0, 0, 0),
            slice_sizes=(16, 64, 64)
        )
    """
    return x[
        :,
        slices[0]: slices[0] + slice_sizes[0],
        slices[1]: slices[1] + slice_sizes[1],
        slices[2]: slices[2] + slice_sizes[2],
    ]


def torch_map(fn, xs) -> Tuple[torch.Tensor, torch.Tensor,torch.Tensor]:
    """ approx like jax.lax.map """
    return


def torch_scan(
        f: FunctionType,
        init: int = 0,
        xs: Optional[List] = None,
        length: int = 0
) -> Tuple[Any, torch.Tensor]:
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, torch.stack(ys, dim=0)

