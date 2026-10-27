from typing import Optional
import torch    
from torch import nn, Tensor
from .gated_residual_network import GatedResidualNetwork as GRN

class VSN(nn.Module):
    """
    Variable Selection Network (VSN) as defined in the TFT paper: https://arxiv.org/pdf/1912.09363
    
    Allows to select the most relevant variables at each timestep for making the prediction.
    In the Temporal Fusion Transformer, VSNs are used to suppress unnecessary or noisy variables
    and improve interpretability.

    Each input variable is first independently transformed using its own Gated Residual Network (GRN). 
    The transformed representations are then combined using learned selection weights produced by a separate
    GRN-based gating network.

    Args:
        num_vars: int
            Number of input variables.
        input_dim: int
            Dimensionality of each individual variable embedding.
        hidden_dim: int
            Hidden dimensionality for transformed variable representations.
        context_dim: int
            Dimensionality of the optional context vector used to condition variable selection.
            In TFT, this context is typically derived from static covariates.
        dropout: float
            Dropout probability used inside the GRNs.
    """
    def __init__(
        self,
        num_vars: int,
        input_dim: int,
        hidden_dim: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_vars = num_vars

        # One specific GRN per input variables
        self.var_grns = nn.ModuleList(
            [
                GRN(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_vars)
            ]
        )

        # For variable selection weights
        # Takes all variables (flattened vector) as inputs and outputs one weight to each variable
        # The weights are then passed to a softmax
        self.weight_grn = GRN(
            input_dim=input_dim * num_vars,
            hidden_dim=hidden_dim,
            output_dim=num_vars,
            context_dim=context_dim,
            dropout=dropout,
        )

    def forward(self, x: Tensor, context: Optional[Tensor] = None):
        """
        Args:
            x: (B, T, N, C) or (B, N, C)
            context: if provided, shape = (B, 1, C) (in TFT, static covariate embeddings are given as context)
            B: batch, T: time dimension, N: number of variables, C: dimension for each variable
        
        Outputs:
            output: 
                tensor of aggregated variable representation after variable selection.
                shape (B, T, C) if x.shape == (B, T, N, C) or (B, C) if x.shape == (B, N, C)
            weights: 
                tensor of softmax-normalized variable importance weights. 
                shape (B, T, N)
        """
        if x.ndim == 4:
            has_time_dimension = True
            B, T, N, C = x.shape
        elif x.ndim == 3:
            has_time_dimension = False
            B, N, C = x.shape
            # We treat this tensor as having a time dimension with only one timestep
            # This way, we can consider that when entering the forward,
            # x.shape is always (B, T, N, C), with T>=1
            # This makes it easier to read the shapes throughout the forward function 
            # because we have the same function behaviour than when x.ndim == 4
            x = x[:, None, :, :]
        else:
            raise ValueError(
                "Input tensor shape is expected to be either "
                f"(B, T, N, C) or (B, N, C), got {x.shape}"
            )
        if context is not None:
            assert context.ndim == 3, (
                "context tensor shape is expected to be (B, 1, C), "
                f"but got {context.shape}"
            )

        var_outputs = []
        # 对每个变量的嵌入进行独立的GRN变换
        for i in range(self.num_vars):
            var_i = x[..., i, :]  # (B, T, C)
            var_out = self.var_grns[i](var_i)  # (B, T, H) with H = hidden_dim
            var_outputs.append(var_out)

        var_outputs = torch.stack(var_outputs, dim=-2)  # (B, T, N, H)

        # 计算变量重要性权重
        flat_x = x.view(B, -1, N * C)  # flatten the last two dimension (B, T, N*C)，注意这里的N*C是每个时间步的所有变量的嵌入拼接在一起
        weights = self.weight_grn(flat_x, context)  # (B, T, N)
        weights = torch.softmax(weights, dim=-1)  # (B, T, N)

        output = (var_outputs * weights[..., None]).sum(dim=-2)  # (B, T, H)

        if not has_time_dimension:
            # In case where input.ndim was 3, we artificially created a time dimension 
            # with one timestep for the sake of readability.
            # We remove this dimension before exiting the function
            output = output.squeeze(1)

        return output, weights