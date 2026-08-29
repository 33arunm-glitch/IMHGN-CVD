import numpy as np
from src.graph import knn_edge_index


def test_knn_graph_is_symmetric():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(10, 5)).astype(np.float32)
    edge_index, edge_weight = knn_edge_index(x, k=3, metric="cosine", undirected=True)

    edges = set(map(tuple, edge_index.t().numpy().tolist()))
    for i, j in edges:
        assert (j, i) in edges
    assert len(edge_weight) == edge_index.shape[1]
    assert np.isfinite(edge_weight.numpy()).all()
