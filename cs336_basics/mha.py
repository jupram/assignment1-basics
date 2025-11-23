# implement Multi-Head Attention using pyTorch.
# start with softmax function then scaled dot-product attention, then multi-head attention etc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
from linear import LinearLayer
from rope import RoPE


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

class MultiHeadAttention(nn.Module):
    def __init__(self, dmodel: int, num_heads: int, device=None, dtype=None):
        super(MultiHeadAttention, self).__init__()
        assert dmodel % num_heads == 0, "dmodel must be divisible by num_heads"
        self.dmodel = dmodel
        self.num_heads = num_heads
        self.d_k = dmodel // num_heads

        self.linear_q = LinearLayer(dmodel, dmodel, device=device, dtype=dtype)
        self.linear_k = LinearLayer(dmodel, dmodel, device=device, dtype=dtype)
        self.linear_v = LinearLayer(dmodel, dmodel, device=device, dtype=dtype)
        self.linear_out = LinearLayer(dmodel, dmodel, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:

        # Linear projections
        Q = self.linear_q(x)  # (batch_size, seq_len, dmodel)
        K = self.linear_k(x)     # (batch_size, seq_len, dmodel)
        V = self.linear_v(x)   # (batch_size, seq_len, dmodel)
        # Split into multiple heads
        Q = rearrange(Q, '... seq (h d) -> ... h seq d', h=self.num_heads)
        K = rearrange(K, '... seq (h d) -> ... h seq d', h=self.num_heads)
        V = rearrange(V, '... seq (h d) -> ... h seq d', h=self.num_heads)
        # Create mask if not provided (causal mask)
        if mask is None:
            mask = torch.tril(torch.ones(x.size(-2), x.size(-2), device=x.device)).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

        print(mask)
        # Scaled dot-product attention
        attn_output = scaled_dot_product_attention(Q, K, V, mask)  # (batch_size, num_heads, seq_len, d_k)

        # Concatenate heads
        attn_output = rearrange(attn_output, 'b h seq d -> b seq (h d)')

        # Final linear layer
        output = self.linear_out(attn_output)  # (batch_size, seq_len, dmodel)

        return output


class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, dmodel: int, num_heads: int, max_seq_len: int, theta: int, device=None, dtype=None):
        super(MultiHeadAttentionWithRoPE, self).__init__()
        assert dmodel % num_heads == 0, "dmodel must be divisible by num_heads"
        self.dmodel = dmodel
        self.num_heads = num_heads
        self.d_k = dmodel // num_heads

        self.linear_q = LinearLayer(dmodel, dmodel, device=device, dtype=dtype)
        self.linear_k = LinearLayer(dmodel, dmodel, device=device, dtype=dtype)
        self.linear_v = LinearLayer(dmodel, dmodel, device=device, dtype=dtype)
        self.linear_out = LinearLayer(dmodel, dmodel, device=device, dtype=dtype)
        self.theta = theta

        self.rope = RoPE(theta=self.theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:

        # Linear projections
        Q = self.linear_q(x)  # (batch_size, seq_len, dmodel)
        K = self.linear_k(x)     # (batch_size, seq_len, dmodel)
        V = self.linear_v(x)   # (batch_size, seq_len, dmodel)
        # Split into multiple heads
        Q = rearrange(Q, '... seq (h d) -> ... h seq d', h=self.num_heads)
        K = rearrange(K, '... seq (h d) -> ... h seq d', h=self.num_heads)
        V = rearrange(V, '... seq (h d) -> ... h seq d', h=self.num_heads)

        if token_positions is None:
            seq_len = x.size(-2)
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        # Add head axis so RoPE broadcast works with (batch, heads, seq, d_k)
        token_positions = token_positions.unsqueeze(1)
        # Apply RoPE
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        # Create mask if not provided (causal mask)
        if mask is None:
            mask = torch.tril(torch.ones(x.size(-2), x.size(-2), device=x.device)).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

        # Scaled dot-product attention
        attn_output = scaled_dot_product_attention(Q, K, V, mask)  # (batch_size, num_heads, seq_len, d_k)
        # Concatenate heads
        attn_output = rearrange(attn_output, 'b h seq d -> b seq (h d)')
        # Final linear layer
        output = self.linear_out(attn_output)  # (batch_size, seq_len, dmodel)

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

def test_multi_head_attention():
    layer = MultiHeadAttention(dmodel=4, num_heads=2)
    input = torch.randn(1, 3, 4)  # (batch=1, seq=3, dmodel=4)
    output = layer(input)
    print("Output:", output)
    expected_shape = (1, 3, 4)
    assert output.shape == expected_shape, f"Output shape {output.shape} does not match expected shape {expected_shape}"
    print("Multi-Head Attention test passed!")

def test_multi_head_attention_with_rope():
    layer = MultiHeadAttentionWithRoPE(dmodel=4, num_heads=2, max_seq_len=10, theta=10000)
    input = torch.randn(1, 5, 4)  # (batch=1, seq=5, dmodel=4)
    token_positions = torch.tensor([[0, 1, 2, 3, 4]])  # shape: (1, 5)
    output = layer(input, token_positions)
    print("Output:", output)
    expected_shape = (1, 5, 4)
    assert output.shape == expected_shape, f"Output shape {output.shape} does not match expected shape {expected_shape}"
    print("Multi-Head Attention with RoPE test passed!")

if __name__ == "__main__":
    # simple test
    #test_scaled_dot_product_attention()
    test_multi_head_attention()
    test_multi_head_attention_with_rope()
