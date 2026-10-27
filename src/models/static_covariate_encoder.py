from torch import Tensor, nn

from .gated_residual_network import GatedResidualNetwork as GRN

class StaticCovariateEncoder(nn.Module):
    """
    Static covariate encoder as defined in the TFT paper: https://arxiv.org/pdf/1912.09363.

    The encoder transforms a static embedding into four distinct context
    vectors using independent Gated Residual Networks:
        - context_selection: conditioning vector for Variable Selection Networks (VSNs)
        - context_enrichment: context vector used for static enrichment after temporal
            processing
        - context_state_h: initial hidden state for the local LSTM encoder
        - context_state_c: initial cell state for the local LSTM encoder/decoder

    Each context vector is produced independently to allow specialized
    conditioning for different parts of the TFT architecture.

    Args:
        hidden_dim: int
            Dimensionality of the static embedding and all generatedcontext vectors.
        dropout: float
            Dropout probability used inside each Gated Residual Network.
    """
    def __init__(
        self, 
        hidden_dim: int, 
        dropout: float = 0.0
    ):
        super().__init__()
        self.context_selection = GRN(hidden_dim, hidden_dim, dropout=dropout)
        self.context_enrichment = GRN(hidden_dim, hidden_dim, dropout=dropout)
        self.context_state_h = GRN(hidden_dim, hidden_dim, dropout=dropout)
        self.context_state_c = GRN(hidden_dim, hidden_dim, dropout=dropout)

    def forward(self, static_embedding: Tensor):
        """
        static_embedding: (B, H), with H = hidden_dim
        Each returned context vector has shape (B, H)
        """
        c_selection = self.context_selection(static_embedding)  # (B, H)，控制变量选择网络的条件向量
        c_enrichment = self.context_enrichment(static_embedding)  # (B, H)，控制静态丰富的条件向量
        c_h = self.context_state_h(static_embedding)  # (B, H)，LSTM初始隐藏状态
        c_c = self.context_state_c(static_embedding)  # (B, H)，LSTM初始单元状态
        return c_selection, c_enrichment, c_h, c_c