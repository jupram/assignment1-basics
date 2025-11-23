# implement RoPE layer using pyTorch
import torch
import torch.nn as nn
from einops import rearrange
import math

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
        
def test_rope_layer():
    layer = RoPE(theta=10000, d_k=6, max_seq_len=10)
    input = torch.randn(2, 5, 6)  # (batch=2, seq=5, d_k=6)
    token_positions = torch.tensor([[0, 1, 2, 3, 4],
                                    [0, 2, 1, 3, 4]])
    print("Input:", input)
    output = layer(input, token_positions)
    print("Output:", output)
    assert output.shape == (2, 5, 6), "Output shape is incorrect"
    print("RoPE test passed!")


if __name__ == "__main__":
    test_rope_layer()
