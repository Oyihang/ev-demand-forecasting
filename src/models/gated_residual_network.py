from typing import Optional
import torch 
from torch import nn, Tensor
from .gating import GLU

class GatedResidualNetwork(nn.Module):
    """
    The Gated Residual Network (GRN) is the core nonlinear processing block,
    used throughout the Temporal Fusion Transformer (TFT) architecture.

    GRNs are designed to provide:
        - flexible non-linear transformation with gating
        - contextual conditioning (optional)
        - residual learning

    Args:
        input_dim : int
            Dimensionality of the input tensor.
        hidden_dim : int
            Hidden dimension used in the intermediate nonlinear transformation.
        output_dim : int, optional
            Output feature dimension. If None, defaults to `input_dim`.
        context_dim : int, optional
            Dimensionality of the optional context vector.
            When provided, the context is linearly projected and added
            before the ELU activation.
        dropout : float, default=0.1
            Dropout probability applied before gating.
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: Optional[int] = None, 
        context_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.output_dim = output_dim or input_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        self.context_fc = nn.Linear(context_dim, hidden_dim) if context_dim is not None else None
        
        self.elu = nn.ELU()
        
        self.fc2 = nn.Linear(hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.glu = GLU(self.output_dim)
        self.layer_norm = nn.LayerNorm(self.output_dim)
        
        self.skip = (
            nn.Linear(input_dim, self.output_dim) if input_dim != self.output_dim else nn.Identity()
        )

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        # Apply first linear layer and activation
        residual = x 
        
        x = self.fc1(x)     # (*, hidden_dim)
        if self.context_fc is not None and context is not None:
            x = x + self.context_fc(context)    # (*, hidden_dim)

        x = self.elu(x)     # (*, hidden_dim)

        x = self.fc2(x)     # (*, output_dim)   
        x = self.dropout(x)  # (*, output_dim)

        x = self.glu(x)
        
        if self.skip is not None:
            residual = self.skip(residual)
        
        return self.layer_norm(x + residual)    # GRN是返回经过 GLU、残差连接和 LayerNorm 处理后的张量