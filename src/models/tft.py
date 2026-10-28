from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .attention import InterpretableMultiHeadSelfAttention
from .embeddings import TFTEmbeddings
from .gated_residual_network import GatedResidualNetwork as GRN
from .gating import GateAddNorm
from .static_covariate_encoder import StaticCovariateEncoder
from .variable_selection_network import VSN


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer (TFT) for multi-horizon time series forecasting.

    This implementation follows the architecture proposed in the paper (https://arxiv.org/pdf/1912.09363).

    It includes:
        - Per-variable embeddings (via `TFTEmbeddings`)
        - Variable Selection Networks (VSNs)
        - LSTM encoder/decoder backbone
        - Static covariate conditioning
        - Temporal interpretable self-attention layer
        - Gated residual connections
        - Quantile regression output

    Args:
        horizon_length : int
            Number of future time steps to forecast (decoder length).
        hidden_dim : int
            Dimensionality of all internal representations in the model.
        known_inputs : List[str]
            Names of time-varying inputs known for all timesteps (past and future),
            e.g. calendar features, planned events, or potential exogenous variables.
        observed_inputs : List[str]
            Names of time-varying inputs only observed historically (not known in advance),
            e.g. target variable and potential exogenous covariates, for which we don't know future values.
        static_covariates : List[str], optional
            Names of time-invariant features, e.g. entity identifiers.
        known_inputs_categorical_cardinalities : List[int], optional
            Cardinalities of categorical variables among `known_inputs`.
            If all variables in known_inputs are continuous, known_inputs_categorical_cardinalities is an empty list.
            The order must match the categorical variables appearing at the end of the corresponding input tensor.
        observed_inputs_categorical_cardinalities : List[int], optional
            Cardinalities of categorical variables among `observed_inputs`.
        static_covariates_categorical_cardinalities : List[int], optional
            Cardinalities of categorical variables among `static_covariates`.
        num_heads : int, default=1
            Number of attention heads in the temporal self-attention layer.
        dropout : float, default=0.0
            Dropout rate applied in GRNs, LSTM, and attention layers.
        lstm_layers : int, default=1
            Number of stacked LSTM layers in encoder and decoder.
        quantiles : Tuple[float], optional, default=(0.1, 0.5, 0.9)
            Quantiles to predict for probabilistic forecasting.

    Input Format:
        The forward pass expects three tensors:
            - observed_inputs: (B, context_length, N_observed)
            - known_inputs: (B, context_length + horizon_length, N_known)
            - static_inputs: (B, N_static)
        where:
            - Continuous variables appear first
            - Categorical variables are concatenated at the end
            - Variable ordering in provided tensors strictly matches the provided name lists

        All tensors are expected to be of dtype `torch.float32`.
        Categorical variables must contain integer-valued entries and will be
        internally cast to `torch.long` before embedding.

    Notes:
        - Each variable is embedded independently (no shared projection)
        - Variable Selection Networks (VSNs) operate on top of embeddings to
            dynamically weight input variables at each time step.
        - Static covariates are used to condition:
            * LSTM initial states
            * variable selection
            * static enrichment layer
        - This implementation assumes that the target variable is included in `observed_inputs`.

    Warnings:
        - Incorrect ordering of variables can lead to silent errors.
    """
    def __init__(
        self,
        horizon_length: int,
        hidden_dim: int,
        known_inputs: List[str],
        observed_inputs: List[str],
        static_covariates: List[str] = [],
        known_inputs_categorical_cardinalities: List[int] = [],
        observed_inputs_categorical_cardinalities: List[int] = [],
        static_covariates_categorical_cardinalities: List[int] = [],
        num_heads: int = 1,
        dropout: float = 0.0,
        lstm_layers: int = 1,
        quantiles: Optional[Tuple[float]] = (0.1, 0.5, 0.9),
    ):
        super().__init__()
        self.horizon_length = horizon_length
        self.hidden_dim = hidden_dim
        self.known_inputs = known_inputs
        self.known_inputs_categorical_cardinalities = known_inputs_categorical_cardinalities
        self.observed_inputs = observed_inputs
        self.observed_inputs_categorical_cardinalities = observed_inputs_categorical_cardinalities
        self.static_covariates = static_covariates
        self.static_covariates_categorical_cardinalities = static_covariates_categorical_cardinalities
        self.num_heads = num_heads
        self.dropout = dropout
        self.lstm_layers = lstm_layers
        self.quantiles = quantiles

        # Input embedding and transformations
        self.embeddings = TFTEmbeddings(
            hidden_dim=hidden_dim,
            known_inputs=known_inputs,
            known_inputs_categorical_cardinalities=known_inputs_categorical_cardinalities,
            observed_inputs=observed_inputs,
            observed_inputs_categorical_cardinalities=observed_inputs_categorical_cardinalities,
            static_covariates=static_covariates,
            static_covariates_categorical_cardinalities=static_covariates_categorical_cardinalities,
        )

        self.static_vsn = None
        self.static_covariate_encoder = None
        self.context_dim = None
        if len(static_covariates):
            self.context_dim = hidden_dim
            # Variable selection network for static covariates
            self.static_vsn = VSN(
                num_vars=len(static_covariates),
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            # Static covariate encoder
            self.static_covariate_encoder = StaticCovariateEncoder(hidden_dim, dropout)

        # Variable selection networks
        self.past_vsn = VSN(
            num_vars=len(known_inputs) + len(observed_inputs),
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            context_dim=self.context_dim,
            dropout=dropout,
        )
        self.future_vsn = VSN(
            num_vars=len(known_inputs),
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            context_dim=self.context_dim,
            dropout=dropout,
        )

        # LSTM encoder + decoder for local processing
        self.encoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.lstm_dropout = nn.Dropout(dropout)
        self.lstm_gate = GateAddNorm(hidden_dim)

        # Static enrichment
        self.enrichment = GRN(hidden_dim, hidden_dim, context_dim=self.context_dim, dropout=dropout)

        # Interpretable multi-head Attention
        self.attention = InterpretableMultiHeadSelfAttention(hidden_dim, num_heads, dropout=dropout)
        self.attn_gate = GateAddNorm(hidden_dim)

        # Position-wise Feed Forward
        self.grn_post_attn = GRN(input_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout)

        # Final gate
        self.final_gate = GateAddNorm(hidden_dim)

        # Quantile forecasts
        output_dim = len(quantiles) if quantiles is not None else 1
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        observed_inputs: Tensor,
        known_inputs: Tensor,
        static_covariates: Optional[Tensor] = None,
    ):
        """
        Run the TFT forward pass.
        For the input format of tensors, please refer to the TFT class docstring above.

        Args:
            observed_inputs: Tensor of shape (B, context_length, N_observed).
            known_inputs: Tensor of shape (B, context_length + horizon_length, N_known).
            static_inputs: Tensor of shape (B, N_static).

        Outputs:
            Forecast tensor of shape (B, horizon_length, output_dim) where output_dim = len(quantiles)
        """
        context_length = observed_inputs.shape[1]
        horizon_length = known_inputs.shape[1] - context_length
        assert self.horizon_length == horizon_length, (
            f"The temporal dim of known_inputs {known_inputs.shape[1]} and observed_inputs "
            f"{observed_inputs.shape[1]} don't match the specified model horizon_length {self.horizon_length}"
        )
        # step 1: 对所有输入进行embedding
        observed_inputs_embedded, known_inputs_embedded, static_covariates_embedded = (
            self.embeddings(
                observed_inputs=observed_inputs,
                known_inputs=known_inputs,
                static_covariates=static_covariates,
            )
        )  
        # known_inputs_embedded shape = (B, context_length, N_observed, hidden_dim), 
        # observed_inputs_embedded shape = (B, context_length + horizon_length, N_known, hidden_dim)
        # static_covariates_embedded shape = (B, N_static, hidden_dim)

        # Separate historical and future known inputs
        past_known_embedded, future_known_embedded = (
            known_inputs_embedded[:, : -self.horizon_length, :, :],
            known_inputs_embedded[:, -self.horizon_length :, :, :],
        )  # (B, T, N_know, hidden_dim)

        # Concatenate past variables (past targets + past observed and known inputs)
        past_inputs_embedded = torch.cat(
            [observed_inputs_embedded, past_known_embedded], dim=-2
        )  # (B, context_length, N_observed + N_known, hidden_dim)

        # step 2: 初始化LSTM隐藏状态和静态变量上下文
        lstm_hidden_t0 = None
        c_selection = None
        c_enrichment = None
        if static_covariates_embedded is not None:
            static_in, _ = self.static_vsn(
                static_covariates_embedded
            )  # (B, hidden_dim)

            # Static covariate encoders
            c_selection, c_enrichment, c_h, c_c = self.static_covariate_encoder(static_in)  # each is (B, hidden_dim)
            c_selection = c_selection.unsqueeze(1)  # (B, 1, hidden_dim)
            c_enrichment = c_enrichment.unsqueeze(1)  # (B, 1, hidden_dim)
            lstm_hidden_t0 = (
                c_h.unsqueeze(0).expand(self.lstm_layers, -1, -1).contiguous(),
                c_c.unsqueeze(0).expand(self.lstm_layers, -1, -1).contiguous(),
            )  # h0, c0

        # Variable selection networks for past and future variables
        # vsn 每个时间步只会输出hidden_dim维度的向量，表示每个时间步的所有变量经过加权后的表示
        encoder_in, _ = self.past_vsn(
            past_inputs_embedded, c_selection
        )  # (B, context_length, hidden_dim)
        decoder_in, _ = self.future_vsn(
            future_known_embedded, c_selection
        )  # (B, horizon_length, hidden_dim)

        # step 3: LSTM
        # LSTM encoder for past variables and LSTM decoder for future variables
        enc_out, (h, c) = self.encoder_lstm(
            encoder_in, lstm_hidden_t0
        )  # (B, context_length, hidden_dim)
        dec_out, _ = self.decoder_lstm(
            decoder_in, (h, c)
        )  # (B, horizon_length, hidden_dim)
        
        # step 4: 拼接encoder和decoder的输出，dropout后进行门控残差连接
        seq2seq_out = torch.cat(
            [enc_out, dec_out], dim=1
        )  # (B, T, hidden_dim) with T = context_length + horizon_length
        seq2seq_out = self.lstm_dropout(seq2seq_out)

        # 门控残差连接，没有静态上下文向量c_enrichment参与
        temp_fusion_dec_in = self.lstm_gate(
            seq2seq_out, residual=torch.cat([encoder_in, decoder_in], dim=1)
        )  # (B, T, hidden_dim)

        # step 5: 对从LSTM输出的序列进行静态丰富处理，使用静态上下文向量c_enrichment
        # Static enrichment
        x = self.enrichment(
            temp_fusion_dec_in, context=c_enrichment
        )  # (B, T, hidden_dim)

        # step 6: 进行时间自注意力机制
        # temporal self attention
        attn_out, _ = self.attention(
            query=x[:, -self.horizon_length :],
            key=x,
            value=x,
            is_causal=True,
        )  # (B, horizon_length, hidden_dim)
        
        # 门控残差连接
        x = self.attn_gate(attn_out, residual=x[:, -self.horizon_length :, :])

        # step 7: GRN position-wise feed forward
        # Position wise ff
        x = self.grn_post_attn(x)  # (B, horizon_length, hidden_dim)

        # step 8: 最终门控残差连接，完全跳过Temporal Fusion Decoder，如果数据集不需要这种复杂性
        # Final gate that completely skips the Temporal Fusion Decoder if this complexity is not required by the dataset
        # Yields a direct path from the LSTM decoder to the output
        x = self.final_gate(
            x, residual=temp_fusion_dec_in[:, -self.horizon_length :, :]
        )  # (B, horizon_length, hidden_dim)

        out = self.output(x)  # Dense layer for generating quantile forecasts

        return out
