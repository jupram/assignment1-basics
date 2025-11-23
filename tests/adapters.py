from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import torch.nn as nn

import regex as re
import multiprocessing as mp
from multiprocessing import cpu_count

from einops import einsum, rearrange
import math

class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(LinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
        self.init_parameters()

    def init_parameters(self):
        bound = 3
        variance = 2/(self.in_features + self.out_features) 
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=variance**0.5, a=-bound, b=bound)

    def forward(self, input):
        return einsum(input, self.weight,"... d_in, d_out d_in -> ... d_out")

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """

    linear = LinearLayer(d_in, d_out)
    linear.weight.data = weights
    return linear(in_features)

class EmbeddingLayer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super(EmbeddingLayer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device))
        self.init_parameters()

    def init_parameters(self):
        bound = 3
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1, a=-bound, b=bound)

    def forward(self, input):
        return self.weight[input]

    def extra_repr(self):
        return 'num_embeddings={}, embedding_dim={}'.format(
            self.num_embeddings, self.embedding_dim
        )

def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    embedding = EmbeddingLayer(vocab_size, d_model)
    embedding.weight.data = weights
    return embedding(token_ids)

class SwiGLU(nn.Module):
    def __init__(self, dmodel : int, d_ff: int = None, device=None, dtype=None):
        super(SwiGLU, self).__init__()
        self.d_model = dmodel

        if d_ff is None:
            # make sure d_ff is a multiple of 64
            target = int(math.ceil((8/3) * dmodel))
            self.d_ff = ((target + 63) // 64) * 64
        else:
            self.d_ff = d_ff

        self.W1 = nn.Parameter(torch.empty(self.d_ff, self.d_model, dtype=dtype, device=device))
        self.W2 = nn.Parameter(torch.empty(self.d_model, self.d_ff, dtype=dtype, device=device))
        self.W3 = nn.Parameter(torch.empty(self.d_ff, self.d_model, dtype=dtype, device=device))
        self.init_parameters()

    def silu(self, x):
        return x * torch.sigmoid(x)

    def init_parameters(self):
        bound = 3
        variance = 2/(self.W1.size(1) + self.W1.size(0)) 
        torch.nn.init.trunc_normal_(self.W1, mean=0.0, std=variance**0.5, a=-bound, b=bound)
        variance = 2/(self.W2.size(1) + self.W2.size(0)) 
        torch.nn.init.trunc_normal_(self.W2, mean=0.0, std=variance**0.5, a=-bound, b=bound)
        variance = 2/(self.W3.size(1) + self.W3.size(0))
        torch.nn.init.trunc_normal_(self.W3, mean=0.0, std=variance**0.5, a=-bound, b=bound)

    def forward(self, x):
        # Compute the SwiGLU activation
        x1 = einsum(x, self.W1, "... d_model, d_ff d_model -> ... d_ff")
        silu_x1 = self.silu(x1)
        x3 = einsum(x, self.W3, "... d_model, d_ff d_model -> ... d_ff")
        y = silu_x1 * x3 # element-wise multiplication
        result = einsum(y, self.W2, "... d_ff, d_model d_ff -> ... d_model")
        return result

    def extra_repr(self):
        return 'dmodel={}, eps={}'.format(
            self.d_model, self.eps
        )

def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    swiglu = SwiGLU(d_model)
    swiglu.W1.data = w1_weight
    swiglu.W2.data = w2_weight
    swiglu.W3.data = w3_weight
    swiglu.d_ff = d_ff  # Ensure d_ff is set correctly
    return swiglu(in_features)

def scaled_dot_product_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute the scaled dot-product attention.
    """

    scores = einsum(queries, keys,"... q d, ... k d -> ... q k") / torch.sqrt(torch.tensor(queries.size(-1), dtype=torch.float32))

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn = softmax(scores, dim=-1)
    output = einsum(attn, values, "... q k, ... k d -> ... q d")
    return output

def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    return scaled_dot_product_attention(Q, K, V, mask)

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

        # Scaled dot-product attention
        attn_output = scaled_dot_product_attention(Q, K, V, mask)  # (batch_size, num_heads, seq_len, d_k)

        # Concatenate heads
        attn_output = rearrange(attn_output, 'b h seq d -> b seq (h d)')

        # Final linear layer
        output = self.linear_out(attn_output)  # (batch_size, seq_len, dmodel)

        return output

def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    #print(q_proj_weight.shape, k_proj_weight.shape, v_proj_weight.shape, o_proj_weight.shape, in_features.shape)
    mha = MultiHeadAttention(d_model, num_heads)
    mha.linear_q.weight.data = q_proj_weight
    mha.linear_k.weight.data = k_proj_weight
    mha.linear_v.weight.data = v_proj_weight
    mha.linear_out.weight.data = o_proj_weight
    return mha(in_features)

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

def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """

    mha = MultiHeadAttentionWithRoPE(d_model, num_heads, max_seq_len, theta)
    mha.linear_q.weight.data = q_proj_weight
    mha.linear_k.weight.data = k_proj_weight
    mha.linear_v.weight.data = v_proj_weight
    mha.linear_out.weight.data = o_proj_weight
    
    return mha(in_features, token_positions)

class RoPE(nn.Module):
    def __init__(self, theta : float, d_k : int, max_seq_len : int, dtype=torch.float32, device=None,):
        super(RoPE, self).__init__()
        self.theta = theta
        self.device = device
        self.dtype = dtype
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # This mapping is needed to generate the rotary angles, following the formula:
        # angle = position / (theta ** (2 * k / d_k)), where k is the block index. each block has size 2.
        self.inv_freq = {(i, k): i / (self.theta ** ((2 * k) / float(self.d_k))) 
                                           for i in range(self.max_seq_len) for k in range(self.d_k // 2)}
        
        # Calculate the rotary positional matrix for each position and block pair.
        # rope_matrix shape: [max_seq_len, d_k // 2, 4] where 4 = [cos, -sin, sin, cos]
        rope_matrix = torch.zeros(self.max_seq_len, self.d_k // 2, 4, device=self.device, dtype=self.dtype)

        for i in range(self.max_seq_len):
            for k in range(self.d_k // 2):
                angle = self.inv_freq[(i, k)]
                rope_matrix[i, k, 0] = math.cos(angle)      # cos(angle)
                rope_matrix[i, k, 1] = -math.sin(angle)     # -sin(angle)
                rope_matrix[i, k, 2] = math.sin(angle)      # sin(angle)
                rope_matrix[i, k, 3] = math.cos(angle)      # cos(angle)

        self.register_buffer('rope_matrix_buffer', rope_matrix, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        # x can be any shape with (..., seq_len, d_k)
        # token_positions shape: (..., seq_len)
        seq_len = x.size(-2)
        d_k = x.size(-1)
        assert d_k == self.d_k, f"Input feature dimension {d_k} does not match RoPE d_k {self.d_k}"
        assert seq_len <= self.max_seq_len, f"Input sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"

        # Get the relevant rotary matrix for the token positions
        rope_matrix = self.rope_matrix_buffer[token_positions] # shape: (..., seq_len, d_k // 2, 4)

        # split the last dimension of x into pairs for rotation
        x_reshaped = rearrange(x, '... seq (block pair) -> ... seq block pair', block=(self.d_k // 2), pair=2)  # shape: (..., seq_len, d_k // 2, 2)
        x1 = x_reshaped[..., 0]  # shape: (..., seq_len, d_k // 2)
        x2 = x_reshaped[..., 1]  # shape: (..., seq_len, d_k // 2)
        
        # apply the rotation
        # rope_matrix[..., 0]: cos(angle), rope_matrix[..., 1]: -sin(angle), rope_matrix[..., 2]: sin(angle), rope_matrix[..., 3]: cos(angle)
        rotated_x1 = x1 * rope_matrix[..., 0] + x2 * rope_matrix[..., 1]  # cos(angle) * x1 - sin(angle) * x2
        rotated_x2 = x1 * rope_matrix[..., 2] + x2 * rope_matrix[..., 3]  # sin(angle) * x1 + cos(angle) * x2

        # combine back the rotated parts
        rotated_x = rearrange(torch.stack([rotated_x1, rotated_x2], dim=-1), '... seq block pair -> ... seq (block pair)')  # shape: (..., seq_len, d_k)
        return rotated_x

def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    rope = RoPE(theta, d_k, max_seq_len, device=in_query_or_key.device)
    return rope(in_query_or_key, token_positions)

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

def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """

    lm_block = LMBlock(d_model, num_heads, d_ff, max_seq_len, theta)
    lm_block.mha.linear_q.weight.data = weights['attn.q_proj.weight']
    lm_block.mha.linear_k.weight.data = weights['attn.k_proj.weight']
    lm_block.mha.linear_v.weight.data = weights['attn.v_proj.weight']
    lm_block.mha.linear_out.weight.data = weights['attn.output_proj.weight']
    lm_block.rms_norm1.gains.data = weights['ln1.weight']
    lm_block.swiglu.W1.data = weights['ffn.w1.weight']
    lm_block.swiglu.W2.data = weights['ffn.w2.weight']
    lm_block.swiglu.W3.data = weights['ffn.w3.weight']
    lm_block.rms_norm2.gains.data = weights['ln2.weight']
    return lm_block(in_features)

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

def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    lm = FullLM(vocab_size, d_model, num_heads, d_ff, num_layers, context_length, rope_theta)
    lm.embedding.weight.data = weights['token_embeddings.weight']
    for layer_idx in range(num_layers):
        lm.layers[layer_idx].mha.linear_q.weight.data = weights[f'layers.{layer_idx}.attn.q_proj.weight']
        lm.layers[layer_idx].mha.linear_k.weight.data = weights[f'layers.{layer_idx}.attn.k_proj.weight']
        lm.layers[layer_idx].mha.linear_v.weight.data = weights[f'layers.{layer_idx}.attn.v_proj.weight']
        lm.layers[layer_idx].mha.linear_out.weight.data = weights[f'layers.{layer_idx}.attn.output_proj.weight']
        lm.layers[layer_idx].rms_norm1.gains.data = weights[f'layers.{layer_idx}.ln1.weight']
        lm.layers[layer_idx].swiglu.W1.data = weights[f'layers.{layer_idx}.ffn.w1.weight']
        lm.layers[layer_idx].swiglu.W2.data = weights[f'layers.{layer_idx}.ffn.w2.weight']
        lm.layers[layer_idx].swiglu.W3.data = weights[f'layers.{layer_idx}.ffn.w3.weight']
        lm.layers[layer_idx].rms_norm2.gains.data = weights[f'layers.{layer_idx}.ln2.weight']
    lm.rms_norm.gains.data = weights['ln_final.weight']
    lm.output_linear.weight.data = weights['lm_head.weight']
    return lm(in_indices)

class RMSNorm(nn.Module):
    def __init__(self, dmodel : int, eps : float = 1e-5 , device=None, dtype=None):
        super(RMSNorm, self).__init__()
        self.dmodel = dmodel
        self.eps = eps
        self.gains = nn.Parameter(torch.ones(dmodel, dtype=dtype, device=device))

    def forward(self, x):
        # Compute the RMS normalization
        in_dtype = x.dtype
        x = x.to(torch.float32)
        norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x = x / norm
        result = x * self.gains
        return result.to(in_dtype)

    def extra_repr(self):
        return 'dmodel={}, eps={}'.format(
            self.dmodel, self.eps
        )

def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    rms_norm = RMSNorm(d_model, eps)
    rms_norm.gains.data = weights
    return rms_norm(in_features)

def silu(x):
        return x * torch.sigmoid(x)

def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    return silu(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError

def softmax (x, dim):
    # numerically stable softmax
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_exp_sum

def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    return softmax(in_features, dim=dim)

def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError


class Tokenizer:
    """A BPE tokenizer that uses a provided vocabulary, merges, and special tokens."""
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.inv_vocab = {t: id for id, t in self.vocab.items()}
        #assign index to UNK token if not present
        self.unk_id = max(self.vocab.keys()) + 1
        vocab[self.unk_id] = b"<|unk|>"
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self._pretokenize_regex = re.compile(PAT)
    
    @classmethod
    def from_files(cls,
        vocab_path: str,
        merges_path: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """Create a Tokenizer instance from vocab and merges files."""
        # Load vocab
        vocab = {}
        #parse json file with int keys and string values
        import json
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_json = json.load(f)
            for token_id, token in vocab_json.items():
                vocab[int(token_id)] = token.encode("utf-8")
        
        # Load merges
        merges = []
        with open(merges_path, "r", encoding="utf-8") as f:
            #parse text json array of arrays with two string elements each
            merge_json = json.load(f)
            for m in merge_json:
                token1, token2 = m
                merges.append((token1.encode("utf-8"), token2.encode("utf-8")))
        
        return cls(vocab, merges, special_tokens)

    def _do_single_merge(self, tokens: list[bytes]) -> list[bytes]:
        """Perform a single BPE merge on the list of tokens."""
        # Create a dictionary of pairs of adjacent tokens
        pairs = {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}

        # Find the first mergeable pair according to the merges list
        for merge in self.merges:
            if merge in pairs:
                # merge all occurrences of the pair
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == merge:
                        new_tokens.append(tokens[i] + tokens[i + 1])
                        i += 2  # Skip the next token as it's merged
                    else:
                        new_tokens.append(tokens[i])
                        i += 1 # Move to the next token
                return new_tokens

        # If no merges were performed, return the original tokens
        return tokens

    def encode(self, text: str) -> list[int]:
        """Encode the input text into a list of token IDs."""
        # Pre-tokenize the input text
        tokens = self._pretokenize(text)
        token_ids = []
        #print(tokens)
        for t in tokens:
            if t in [k.encode("utf-8") for k in self.special_tokens]:
                token_id = self.inv_vocab.get(t, self.unk_id)
                if token_id is not None:
                    token_ids.append(token_id)
                continue
            else:
                # turn token into list of bytes
                subtokens = [bytes([b]) for b in t]
                # Repeatedly apply merges until no more merges can be applied
                while True:
                    new_subtokens = self._do_single_merge(subtokens)
                    if new_subtokens == subtokens:
                        # Convert subtokens to token IDs and add to token_ids
                        for subtoken in new_subtokens:
                            token_id = self.inv_vocab.get(subtoken, self.unk_id)
                            if token_id is not None:
                                token_ids.append(token_id)
                        break
                    subtokens = new_subtokens
        #print(token_ids)
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode an iterable of strings into an iterator of token IDs."""
        for text in iterable:
            yield from self.encode(text)
    
    def decode(self, token_ids: list[int]) -> str:
        """Decode a list of token IDs back into a string."""
        #print(token_ids)
        tokens = [self.vocab.get(tid) for tid in token_ids]
        # if any token is b<|unk|>, replace it with U+FFFD
        for i, token in enumerate(tokens):
            if token == b"<|unk|>":
                tokens[i] = b"\xef\xbf\xbd"  # U+FFFD in UTF-8
        #turn list of bytes into list of strings
        tokens = b"".join(tokens)
        return tokens.decode("utf-8", errors='replace')

    def _pretokenize(self, text: str) -> list[bytes]:
        """Pre-tokenize the input text into a list of byte tokens."""
        # create a regex pattern that matches either any of the special tokens or the pretokenize regex
              
        special_tokens = "|".join(sorted(map(re.escape, self.special_tokens), key=len, reverse=True))
        tokens = []
        parts = re.split(f"({special_tokens})", text)

        for part in parts:
            if part in self.special_tokens:
                tokens.append(part.encode("utf-8"))
            else:
            # Tokenize each part using regex
                for match in self._pretokenize_regex.finditer(part):
                    token = match.group(0)
                    tokens.append(token.encode("utf-8"))
        return tokens

def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    if not special_tokens:
        special_tokens = ["<endoftext>"]
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    return tokenizer

# def run_train_bpe(
#     input_path: str | os.PathLike,
#     vocab_size: int,
#     special_tokens: list[str],
#     **kwargs,
# ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
#     """Given the path to an input corpus, run train a BPE tokenizer and
#     output its vocabulary and merges.

#     Args:
#         input_path (str | os.PathLike): Path to BPE tokenizer training data.
#         vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
#         special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
#             These strings will never be split into multiple tokens, and will always be
#             kept as a single token. If these special tokens occur in the `input_path`,
#             they are treated as any other string.

#     Returns:
#         tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
#             vocab:
#                 The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
#                 to bytes (token bytes)
#             merges:
#                 BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
#                 representing that <token1> was merged with <token2>.
#                 Merges are ordered by order of creation.
#     """
#     raise NotImplementedError


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def process_chunk(
        input_path: str, start: int, end: int, special_token: bytes
    ) -> dict[bytes, int]:
        """
        Process the specified chunk of the input file.
        Splits the chunk by the decoded special token, tokenizes each part using regex,
        counts tokens per part, merges them, and prints the results.
        Returns the merged tokens count.
        """
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # Compile regex pattern
        pattern = re.compile(PAT)

        # Read and process the specified chunk
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            
            # Split chunk using the special token (decoded)
            parts = chunk.split(special_token.decode("utf-8"))
            # Array to hold token counts for each part
            tokens_count = [{} for _ in range(len(parts))]
            
            for i, part in enumerate(parts):
                # Tokenize each part using regex
                counter = {}
                for match in pattern.finditer(part):
                    token = match.group(0)
                    counter[token] = counter.get(token, 0) + 1
                tokens_count[i] = counter
            
            # Merge all token count dictionaries into one
            merged_tokens_count = merge_token_counts(tokens_count)
            return merged_tokens_count

def merge_token_counts(counts_list: list[dict[bytes, int]]) -> dict[bytes, int]:
    """
    Merge a list of token count dictionaries into a single dictionary.
    """
    merged_counts = {}
    for counts in counts_list:
        for token, count in counts.items():
            merged_counts[token] = merged_counts.get(token, 0) + count
    return merged_counts

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer from scratch on the given input file.
    Returns the vocabulary and merges.
    """
     # check if input_path exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file {input_path} does not exist.")
    
    # first intialize empty dictionary for tokens_count for each chunk
    tokens_count = []
    # create a process thread for each chunk
    num_processes = kwargs.get("num_processes", cpu_count())
    special_tokens = [token.encode("utf-8") for token in special_tokens]
    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0])
    #print(f"Chunk boundaries: {chunk_boundaries}")

    # Prepare arguments for each process
    args = [
        (input_path, start, end, special_tokens[0])
        for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:])
    ]

    with mp.Pool(processes=num_processes) as pool:
        tokens_count = pool.starmap(process_chunk, args)

    final_tokens_count = merge_token_counts(tokens_count)
    del tokens_count  # free memory
    
    # change the keys from bytes to tuple of bytes
    final_tokens_tuple_count = {tuple(bytes([b]) for b in token.encode("utf-8")): count for token, count in final_tokens_count.items()}
    del final_tokens_count
    # #print count for "the"
    # the = tuple(bytes([b]) for b in b" the")
    # print(f" Count for token {the}: {final_tokens_tuple_count.get(the, 0)}")

    # iterate over the items of final_tokens_tuple_count and read every tuple([bytes]) and for every consecutive pair of bytes, 
    # add to a new dictionary with count as value 
    pair_counts = {}
    for token_tuple, count in final_tokens_tuple_count.items():
        for i in range(len(token_tuple) - 1):
            pair = (token_tuple[i], token_tuple[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + count
    
    #print(pair_counts[(b' ', b't')], pair_counts[(b't', b'h')], pair_counts[(b'h', b'e')])

    # sort the pair_counts by value in descending order and key in descending order
    if pair_counts:
        most_frequent_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))
        most_frequent_pair = most_frequent_pair[0]
    else:
        most_frequent_pair = None

    # Initialize vocabulary with specail tokens and single-character tokens
    # add sepcial tokens to vocab
    vocab = {}
    vocab_size_current = 0
    for token in special_tokens:
        vocab[vocab_size_current] = token
        vocab_size_current += 1
    # add single character tokens from ascii range 0 to 255
    for i in range(256):
        vocab[vocab_size_current] = bytes([i])
        vocab_size_current += 1
    


    merges = []
    # Perform BPE merges until reaching the desired vocabulary size
    while vocab_size_current < vocab_size and most_frequent_pair: 
        # Get the most frequent pair
        merges.append(most_frequent_pair)

        # Create new token by merging the most frequent pair
        new_token = most_frequent_pair[0] + most_frequent_pair[1]
        vocab[vocab_size_current] = new_token
        vocab_size_current += 1

        # Update token counts with the new merged token
        keys_changed = {}
        
        for token_tuple, count in final_tokens_tuple_count.items():
            # Replace occurrences of the most frequent pair with the new token
            new_token_tuple = []
            i = 0
            merged = False
            while i < len(token_tuple):
                if i < len(token_tuple) - 1 and (token_tuple[i], token_tuple[i + 1]) == most_frequent_pair:
                    new_token_tuple.append(new_token)
                    i += 2
                    merged = True
                else:
                    new_token_tuple.append(token_tuple[i])
                    i += 1
            if merged:
                keys_changed[token_tuple] = new_token_tuple
                
        # create a new dictionary with the items from final_tokens_tuple_count that were changed
        for key in keys_changed.items():
            count = final_tokens_tuple_count.pop(key[0])
            final_tokens_tuple_count[tuple(key[1])] = final_tokens_tuple_count.get(tuple(key[1]), 0) + count
        # Recalculate pair counts
        pair_counts = {}
        for token_tuple, count in final_tokens_tuple_count.items():
            for i in range(len(token_tuple) - 1):
                pair = (token_tuple[i], token_tuple[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + count    

        # Sort pairs by frequency and lexicographically
        if pair_counts:
            most_frequent_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))
            most_frequent_pair = most_frequent_pair[0]
        else:
            most_frequent_pair = None
        #print(f"Vocab size: {vocab_size_current}, Most common pair: {sorted_pairs[0] if sorted_pairs else 'N/A'}") 

    return vocab, merges
