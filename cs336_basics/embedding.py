# implement embedding layer using pyTorch
import torch
import torch.nn as nn
from einops import einsum
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

def test_embedding_layer():
    layer = EmbeddingLayer(10, 6)
    input = torch.randint(0, 10, (2, 3))
    print("Input:", input)
    output = layer(input)
    print("Output:", output)
    assert output.shape == (2, 3, 6), "Output shape is incorrect"
    print("EmbeddingLayer test passed!")

if __name__ == "__main__":
    test_embedding_layer()