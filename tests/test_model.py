import numpy as np
import torch

from src.graph import build_graph
from src.model import IMHGN


def test_two_layer_model_and_fusion():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(20, 6)).astype(np.float32)
    y = rng.integers(0, 2, size=20)
    graph = build_graph(x, y, k=3)

    model = IMHGN(
        input_dim=6,
        modality_slices={
            "clinical": slice(0, 3),
            "demographic": slice(3, 5),
            "lifestyle": slice(5, 6),
        },
        hidden1=64,
        hidden2=32,
        dropout=0.2,
    )

    logits = model(graph)
    assert logits.shape == (20,)
    assert model.gcn1.out_channels == 64
    assert model.gcn2.out_channels == 32
    weights = model.fusion.weights().detach()
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
