import torch 
from torch import nn, Tensor

class GLU(nn.Module):
    """
    Gated Linear Unit (GLU).
    Applies the gated linear unit on the last dimension for tensor of shape (*, dim).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim * 2)
        self.glu = nn.GLU(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        x_proj = self.fc(x)  # (*, 2 * dim)
        return self.glu(x_proj)  # (*, dim)

class GateAddNorm(nn.Module):
    """
    Gated Add & Norm layer.
    Applies a residual connection followed by layer normalization.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.glu = GLU(dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, residual: Tensor) -> Tensor:
        return self.norm(self.glu(x) + residual)
