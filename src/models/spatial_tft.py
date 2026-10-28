"""Spatially enhanced Temporal Fusion Transformer (ST-TFT).

A graph convolution encodes cross-zone relationships at every observed timestep.
The resulting time-varying spatial features are passed to the TFT alongside the
historical target, where its temporal layers model their evolution.
"""
from __future__ import annotations
from typing import Sequence
import torch
from torch import Tensor, nn

from .tft import TemporalFusionTransformer


class GraphConvolution(nn.Module):
    """Simple normalized graph convolution: H' = sigma(A_norm X W)."""
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        # x: (B,T,N,F), adjacency: (N,N)
        aggregated = torch.einsum("ij,btjf->btif", adjacency, x)
        return self.dropout(torch.relu(self.norm(self.linear(aggregated))))


class SpatialEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gcn1 = GraphConvolution(input_dim, hidden_dim, dropout)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim, dropout)
        self.output_dim = hidden_dim

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        return self.gcn2(self.gcn1(x, adjacency), adjacency)


class SpatialTFT(nn.Module):
    """TFT + GNN for all zones in a city.

    Inputs:
      observed: (B, context, N, 1), normalized historical target per zone
      graph_features: (B, context, N, 5), historical duration, prices,
        occupancy, and volume
      known: (B, context+horizon, N, 3), integer day-of-week, hour, and weekend
      static: (N, 3), charge count, area, and perimeter
      adjacency: (N,N), normalized adjacency matrix
    Output:
      (B, N, horizon, num_quantiles), in normalized target units
    """
    def __init__(self, context_length: int, horizon_length: int,
                 graph_input_dim: int, known_feature_names: Sequence[str],
                 static_feature_names: Sequence[str], hidden_dim: int = 32,
                 gnn_hidden_dim: int = 16, num_heads: int = 4,
                 dropout: float = 0.1, lstm_layers: int = 1,
                 quantiles=(0.1, 0.5, 0.9), use_spatial: bool = True,
                 known_input_cardinalities: Sequence[int] = (7, 24, 2)):
        super().__init__()
        self.context_length = context_length
        self.horizon_length = horizon_length
        self.quantiles = tuple(quantiles)
        self.use_spatial = bool(use_spatial)
        if len(known_feature_names) != len(known_input_cardinalities):
            raise ValueError("Each known categorical feature must have a cardinality")
        self.spatial_encoder = SpatialEncoder(graph_input_dim, gnn_hidden_dim, dropout)
        self.spatial_projection = nn.Sequential(
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim), nn.ReLU(), nn.LayerNorm(gnn_hidden_dim)
        )
        self.tft = TemporalFusionTransformer(
            horizon_length=horizon_length,
            hidden_dim=hidden_dim,
            known_inputs=list(known_feature_names),
            known_inputs_categorical_cardinalities=list(known_input_cardinalities),
            observed_inputs=["target_volume"] + [f"spatial_feature_{i}" for i in range(gnn_hidden_dim)],
            static_covariates=list(static_feature_names),
            num_heads=num_heads,
            dropout=dropout,
            lstm_layers=lstm_layers,
            quantiles=self.quantiles,
        )

    def forward(self, observed: Tensor, graph_features: Tensor, known: Tensor,
                static: Tensor, adjacency: Tensor) -> Tensor:
        if observed.ndim != 4 or graph_features.ndim != 4 or known.ndim != 4:
            raise ValueError("observed, graph_features and known must be rank-4 tensors")
        b, t, n, _ = observed.shape
        if t != self.context_length:
            raise ValueError(f"Expected context length {self.context_length}, got {t}")
        if graph_features.shape[:3] != (b, t, n):
            raise ValueError("graph_features must align with observed (B,T,N)")
        if graph_features.shape[-1] != self.spatial_encoder.gcn1.linear.in_features:
            raise ValueError("graph_features do not match the configured GNN input features")
        if known.shape[:3] != (b, t + self.horizon_length, n):
            raise ValueError("known must have shape (B, context+horizon, N, K)")
        if known.shape[-1] != len(self.tft.known_inputs):
            raise ValueError("known must contain all configured calendar features")
        if static.shape[0] != n or adjacency.shape != (n, n):
            raise ValueError("static features / adjacency do not align with zones")
        if static.shape[-1] != len(self.tft.static_covariates):
            raise ValueError("static must contain all configured static features")

        if self.use_spatial:
            spatial_seq = self.spatial_encoder(graph_features, adjacency)  # B,T,N,G
            spatial_features = self.spatial_projection(spatial_seq)  # B,T,N,G
        else:
            # Controlled temporal-only ablation: no information from the graph branch.
            spatial_features = observed.new_zeros(
                (b, t, n, self.spatial_projection[0].out_features)
            )

        # Flatten each (window, zone) into an independent TFT series.
        observed_with_spatial = torch.cat([observed, spatial_features], dim=-1)
        observed_flat = observed_with_spatial.permute(0, 2, 1, 3).reshape(b * n, t, -1)
        known_flat = known.permute(0, 2, 1, 3).reshape(b * n, t + self.horizon_length, -1)
        static_batch = static.unsqueeze(0).expand(b, -1, -1)
        static_flat = static_batch.reshape(b * n, -1)
        forecast = self.tft(observed_inputs=observed_flat, known_inputs=known_flat,
                            static_covariates=static_flat)
        return forecast.reshape(b, n, self.horizon_length, len(self.quantiles))
