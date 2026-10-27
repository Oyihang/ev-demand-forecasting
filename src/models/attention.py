from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class InterpretableMultiHeadSelfAttention(nn.Module):
    """
    Interpretable Temporal Self Attention Block as defined in the TFT paper: https://arxiv.org/pdf/1912.09363.

    This class assumes that keys, querys and values share the same embedding space (have the same dimension d_model).
    Also assumes input shape = (B, T, C) (equivalent to setting batch_first=True in nn.MultiheadAttention).

    Args:
        d_model: int
            Model dimension.
        num_heads: int
            Number of parallel attention heads. 
            Note that ``d_model`` will be split across ``num_heads``
        dropout: float
            Dropout probability on output.
        bias: bool
            Use bias in projections.
        device: str or torch.device, optional
            Device on which to initialize.
        dtype: torch.dtype, optional)
            dtype to use.

    Note: TFT interpretable attention differs from standard Transformer multi-head attention:
        - values are shared across heads
        - heads are averaged rather than concatenated
        - interpretability is prioritized over maximal expressiveness
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        bias: bool = True,
        dropout: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        assert d_model % num_heads == 0, f"num_heads ({num_heads}) must divide d_model ({d_model})"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model
        self.bias = bias

        # key, query projections for all heads, in a batch
        self.qk_proj = nn.Linear(d_model, 2 * d_model, **factory_kwargs)
        if bias:
            self.qk_proj.bias.data.zero_()  # Following Pytorch convention
        # Value projection: values are shared across heads
        self.v_proj = nn.Linear(d_model, self.head_dim)
        if bias:
            self.v_proj.bias.data.zero_()  # Following Pytorch convention
        self.dropout = nn.Dropout(dropout)
        # output projection
        self.out_proj = nn.Linear(self.head_dim, d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        is_causal: bool = True,
    ):
        """
        query: (B, T_query, d_model)
        key: (B, T_key, d_model)
        value: (B, T_key, d_model)
        """
        T_query, T_key = query.size(-2), key.size(-2)

        if is_causal:
            # Queries may represent only the final horizon steps while keys contain
            # the full encoder+decoder sequence. Offset query positions accordingly:
            # the first query can attend through key index T_key - T_query.
            query_positions = torch.arange(T_query, device=query.device) + (T_key - T_query)
            key_positions = torch.arange(T_key, device=query.device)
            attn_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)

        # 这里的query和key是输入
        if query is key:
            # query and key are the same input
            # Calculate query, key for all heads
            result_packed_proj = self.qk_proj(query)
            query, key = torch.chunk(result_packed_proj, 2, dim=-1)  # (B, T, d_model)，经过变换矩阵后变成Q和K矩阵
        else:
            # key, query are different (e.g., cross attention). In this case we have to manually split the weights and biases.
            q_weight, k_weight = torch.chunk(self.qk_proj.weight, 2, dim=0)  # (d_model, d_model)
            if self.bias:
                q_bias, k_bias = torch.chunk(self.qk_proj.bias, 2, dim=0)
            else:
                q_bias, k_bias = None, None
            query, key = (
                F.linear(query, q_weight, q_bias),
                F.linear(key, k_weight, k_bias),
            )
        value = self.v_proj(value)  # (B, T, head_dim)

        key = key.view(-1, T_key, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T_key, head_dim)
        query = query.view(-1, T_query, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T_query, head_dim)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim**0.5)  # (B, num_heads, T_query, T_key)
        if is_causal:
            attn_scores = attn_scores.masked_fill(~attn_mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, num_heads, T, T)

        # Average attention maps acoross heads (for interpretability)
        attn_weights = attn_weights.mean(dim=1)  # (B, T_query, T_key)
        # Apply attention map to shared values
        out = torch.matmul(attn_weights, value)  # (B, T_query, head_dim)
        out = self.out_proj(out)

        return self.dropout(out), attn_weights