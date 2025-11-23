# implement RMSNorm layer using pyTorch
import torch
import torch.nn as nn
from einops import einsum
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

def test_rmsnorm_layer():
    layer = RMSNorm(6)
    input = torch.randn(2, 3, 6)
    print("Input:", input)
    output = layer(input)
    print("Output:", output)
    assert output.shape == (2, 3, 6), "Output shape is incorrect"
    print("RMSNorm test passed!")

if __name__ == "__main__":
    test_rmsnorm_layer()