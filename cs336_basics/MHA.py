# implement Multi-Head Attention using pyTorch.
# start with softmax function then scaled dot-product attention, then multi-head attention etc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

def softmax(x, dim):
    # numerically stable softmax
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_exp_sum

def scaled_dot_product_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute the scaled dot-product attention.
    """

    scores = einsum("... q d, ... k d -> ... q k", queries, keys) / torch.sqrt(torch.tensor(queries.size(-1), dtype=torch.float32))

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn = softmax(scores, dim=-1)
    output = einsum("... q k, ... k d -> ... q d", attn, values)
    return output

def test_scaled_dot_product_attention():
    # simple test
    queries = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # shape: (1, 2, 2)
    keys = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])     # shape: (1, 2, 2)
    values = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])   # shape: (1, 2, 2)
    mask = torch.tensor([[[1, 0], [1, 1]]])      # shape: (1, 2, 2)

    output = scaled_dot_product_attention(queries, keys, values, mask)
    print("Output:", output)
    expected_shape = (1, 2, 2)
    assert output.shape == expected_shape, f"Output shape {output.shape} does not match expected shape {expected_shape}"
    print("Scaled dot-product attention test passed!")

if __name__ == "__main__":
    # simple test
    test_scaled_dot_product_attention()
