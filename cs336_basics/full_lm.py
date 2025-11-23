import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
from linear import LinearLayer
from rope import RoPE
from mha import MultiHeadAttentionWithRoPE, softmax, scaled_dot_product_attention
from swiglu import SwiGLU
from embedding import EmbeddingLayer
from rmsnorm import RMSNorm


class LMBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: int, device=None, dtype=None):
        super(LMBlock, self).__init__()
        self.mha = MultiHeadAttentionWithRoPE(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.swiglu = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.rms_norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.rms_norm2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        x1 = self.rms_norm1(x)
        # Multi-head attention with RoPE
        attn_output = self.mha(x1, token_positions, mask)
        x = x + attn_output  # Residual connection

        # SwiGLU feed-forward network
        x2 = self.rms_norm2(x)
        ff_output = self.swiglu(x2)
        x = x + ff_output  # Residual connection

        return x


class FullLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int, num_layers: int, max_seq_len: int, theta: int, device=None, dtype=None):
        super(FullLM, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            LMBlock(d_model, num_heads, d_ff, max_seq_len, theta, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.rms_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.output_linear = LinearLayer(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)  # (batch_size, seq_len, d_model)
        for layer in self.layers:
            x = layer(x)
        x = self.rms_norm(x)
        logits = self.output_linear(x)  # (batch_size, seq_len, vocab_size)
        return logits

def test_lm_block():
    d_model = 8
    num_heads = 2
    d_ff = 32
    max_seq_len = 16
    theta = 10000

    lm_block = LMBlock(d_model, num_heads, d_ff, max_seq_len, theta)
    input = torch.randn(2, 10, d_model)  # (batch=2, seq=10, d_model=8)
    output = lm_block(input)
    print("Output:", output)
    assert output.shape == (2, 10, d_model), "Output shape is incorrect"
    print("LMBlock test passed!")

if __name__ == "__main__":
    test_lm_block()