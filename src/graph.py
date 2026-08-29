from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


@dataclass
class GraphData:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    y: Optional[torch.Tensor] = None

    def to(self, device):
        return GraphData(
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            edge_weight=self.edge_weight.to(device),
            y=None if self.y is None else self.y.to(device),
        )

    def clone(self):
        return GraphData(
            x=self.x.clone(),
            edge_index=self.edge_index.clone(),
            edge_weight=self.edge_weight.clone(),
            y=None if self.y is None else self.y.clone(),
        )


def _similarity_matrix(x: np.ndarray, metric: str = "cosine") -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if metric == "cosine":
        s = cosine_similarity(x)
    elif metric == "euclidean":
        d = euclidean_distances(x)
        s = 1.0 / (1.0 + d)
    else:
        raise ValueError(f"Unsupported graph metric: {metric}")
    np.fill_diagonal(s, -np.inf)
    return s


def knn_edge_index(
    x: np.ndarray,
    *,
    k: int = 5,
    metric: str = "cosine",
    weighted: bool = True,
    undirected: bool = True,
):
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    if n < 2:
        idx = torch.arange(n, dtype=torch.long)
        return torch.stack([idx, idx], dim=0), torch.ones(n, dtype=torch.float32)

    k_eff = max(1, min(int(k), n - 1))
    s = _similarity_matrix(x, metric=metric)

    directed = {}
    for i in range(n):
        nbrs = np.argpartition(-s[i], kth=k_eff - 1)[:k_eff]
        nbrs = nbrs[np.argsort(-s[i, nbrs])]
        for j in nbrs:
            w = float(s[i, j])
            if np.isfinite(w):
                directed[(i, int(j))] = w

    if undirected:
        undirected_edges = {}
        for (i, j), w in directed.items():
            a, b = min(i, j), max(i, j)
            undirected_edges[(a, b)] = max(undirected_edges.get((a, b), -np.inf), w)
        edges = {}
        for (i, j), w in undirected_edges.items():
            edges[(i, j)] = w
            edges[(j, i)] = w
    else:
        edges = directed

    if not edges:
        raise RuntimeError("No graph edges were created.")

    pairs = sorted(edges.keys())
    edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(
        [edges[p] if weighted else 1.0 for p in pairs], dtype=torch.float32
    )
    return edge_index, edge_weight


def identity_graph(x: np.ndarray):
    n = len(x)
    idx = torch.arange(n, dtype=torch.long)
    return torch.stack([idx, idx], dim=0), torch.ones(n, dtype=torch.float32)


def build_graph(
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
    *,
    k: int = 5,
    metric: str = "cosine",
    weighted: bool = True,
    undirected: bool = True,
    no_graph: bool = False,
) -> GraphData:
    x = np.asarray(x, dtype=np.float32)
    if no_graph:
        edge_index, edge_weight = identity_graph(x)
    else:
        edge_index, edge_weight = knn_edge_index(
            x, k=k, metric=metric, weighted=weighted, undirected=undirected
        )

    return GraphData(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=None if y is None else torch.tensor(np.asarray(y).astype(np.float32)),
    )
