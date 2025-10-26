# linear layer implementation using pyTorch
import torch
import torch.nn as nn
from einops import einsum, rearrange

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
        return einsum(input, self.weight,"... d_in, d_out d_in -> ... d_out", )

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


def test_linear_layer():
    layer = LinearLayer(4, 3)
    input = torch.randn(2, 2, 4)
    output = layer(input)
    print("Output:", output)
    assert output.shape == (2, 2, 3), "Output shape is incorrect"
    print("LinearLayer test passed!")

    print(layer)


if __name__ == "__main__":
    test_linear_layer()
