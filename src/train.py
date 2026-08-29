from __future__ import annotations

import copy
import random
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, balanced_accuracy_score


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(data, device):
    return data.to(device)


def train_model(
    model,
    train_graph,
    val_graph=None,
    *,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 150,
    patience: int = 20,
    device: Optional[str] = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model = model.to(device)
    train_graph = _to_device(train_graph, device)
    val_graph = _to_device(val_graph, device) if val_graph is not None else None

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_state = copy.deepcopy(model.state_dict())
    best_score = -np.inf
    no_improve = 0

    for _ in range(int(epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_graph)
        loss = loss_fn(logits, train_graph.y)
        loss.backward()
        optimizer.step()

        if val_graph is None:
            score = -float(loss.detach().cpu())
        else:
            score = validation_score(model, val_graph)

        if score > best_score + 1e-8:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= int(patience):
                break

    model.load_state_dict(best_state)
    return model, float(best_score)


@torch.no_grad()
def validation_score(model, graph) -> float:
    model.eval()
    probs = torch.sigmoid(model(graph)).detach().cpu().numpy()
    y = graph.y.detach().cpu().numpy().astype(int)
    if len(np.unique(y)) == 2:
        return float(roc_auc_score(y, probs))
    pred = (probs >= 0.5).astype(int)
    return float(balanced_accuracy_score(y, pred))


@torch.no_grad()
def predict_proba(model, graph, device: Optional[str] = None) -> np.ndarray:
    if device is None:
        device = next(model.parameters()).device
    graph = graph.to(device)
    model.eval()
    return torch.sigmoid(model(graph)).detach().cpu().numpy()
