from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import torch


def modality_weights(model) -> Dict[str, float]:
    return model.modality_weights()


def gradient_feature_importance(model, graph) -> np.ndarray:
    """
    Lightweight feature-level attribution for the transformed feature space.

    Returns mean absolute input gradient per transformed feature.
    This does not replace clinical SHAP analysis on the original feature space,
    but provides a dependency-light diagnostic for reproducibility.
    """
    model.eval()
    x = graph.x.detach().clone().requires_grad_(True)
    local = graph.clone()
    local.x = x
    logits = model(local)
    score = torch.sigmoid(logits).mean()
    score.backward()
    grad = x.grad.detach().abs().mean(dim=0).cpu().numpy()
    return grad


def try_shap_kernel(predict_fn, background: np.ndarray, samples: np.ndarray):
    """
    Optional generic SHAP helper.

    The caller is responsible for providing a prediction function that accepts
    a 2-D NumPy array and returns positive-class probabilities.
    """
    try:
        import shap
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("SHAP is not available in the current environment.") from exc

    explainer = shap.KernelExplainer(predict_fn, background)
    return explainer.shap_values(samples)
