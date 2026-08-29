from __future__ import annotations

from typing import Dict, Mapping

import torch
from torch import nn
from torch.nn import functional as F


class WeightedGCNConv(nn.Module):
    """
    Self-contained weighted GCN layer implementing symmetric normalization:

        H' = D_hat^{-1/2} A_hat D_hat^{-1/2} H W

    Self-loops are added internally. The implementation uses torch.index_add_
    and does not require a graph-library-specific runtime.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.linear = nn.Linear(self.in_channels, self.out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.out_channels))

    def forward(self, x, edge_index, edge_weight):
        n = x.shape[0]
        device = x.device
        dtype = x.dtype

        src, dst = edge_index[0], edge_index[1]
        w = edge_weight.to(device=device, dtype=dtype)

        loops = torch.arange(n, device=device, dtype=torch.long)
        src = torch.cat([src.to(device), loops], dim=0)
        dst = torch.cat([dst.to(device), loops], dim=0)
        w = torch.cat([w, torch.ones(n, device=device, dtype=dtype)], dim=0)

        deg = torch.zeros(n, device=device, dtype=dtype)
        deg.index_add_(0, dst, w)
        inv_sqrt = deg.clamp_min(1e-12).pow(-0.5)
        norm = inv_sqrt[src] * w * inv_sqrt[dst]

        h = self.linear(x)
        msg = h[src] * norm.unsqueeze(-1)
        out = torch.zeros((n, self.out_channels), device=device, dtype=dtype)
        out.index_add_(0, dst, msg)
        return out + self.bias


class ModalityFusion(nn.Module):
    def __init__(self, modality_slices: Mapping[str, slice], learnable: bool = True):
        super().__init__()
        self.names = list(modality_slices.keys())
        self.slices = dict(modality_slices)
        if not self.names:
            raise ValueError("At least one modality slice is required.")
        self.learnable = bool(learnable)
        if self.learnable:
            self.logits = nn.Parameter(torch.zeros(len(self.names), dtype=torch.float32))
        else:
            self.register_buffer("logits", torch.zeros(len(self.names), dtype=torch.float32))

    def weights(self) -> torch.Tensor:
        if self.learnable:
            return torch.softmax(self.logits, dim=0)
        return torch.ones_like(self.logits) / max(1, len(self.names))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weights()
        return torch.cat(
            [x[:, self.slices[name]] * w[idx] for idx, name in enumerate(self.names)],
            dim=1,
        )


class IMHGN(nn.Module):
    """
    Two-layer modality-aware GCN.

    Default reported configuration:
      input -> GCN(64) -> GCN(32) -> binary classification head
    """

    def __init__(
        self,
        input_dim: int,
        modality_slices: Mapping[str, slice],
        hidden1: int = 64,
        hidden2: int = 32,
        dropout: float = 0.2,
        learnable_fusion: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden1 = int(hidden1)
        self.hidden2 = int(hidden2)
        self.dropout = float(dropout)
        self.fusion = ModalityFusion(modality_slices, learnable=learnable_fusion)
        self.gcn1 = WeightedGCNConv(self.input_dim, self.hidden1)
        self.gcn2 = WeightedGCNConv(self.hidden1, self.hidden2)
        self.classifier = nn.Linear(self.hidden2, 1)

    def forward(self, data):
        x = self.fusion(data.x)
        x = self.gcn1(x, data.edge_index, data.edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, data.edge_index, data.edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, data) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self(data))

    @torch.no_grad()
    def modality_weights(self) -> Dict[str, float]:
        vals = self.fusion.weights().detach().cpu().numpy().tolist()
        return {name: float(v) for name, v in zip(self.fusion.names, vals)}
