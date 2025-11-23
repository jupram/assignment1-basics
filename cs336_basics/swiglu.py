# implement SwiGLU layer using pyTorch
import torch
import torch.nn as nn
from einops import einsum
import math
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
            self.dmodel, self.eps
        )

def test_swiglu_layer():
    layer = SwiGLU(6)
    input = torch.randn(2, 3, 6)
    print("Input:", input)
    output = layer(input)
    print("Output:", output)
    assert output.shape == (2, 3, 6), "Output shape is incorrect"
    print("SwiGLU test passed!")

if __name__ == "__main__":
    test_swiglu_layer()