import numpy as np
import pandas as pd

from src.preprocessing import LeakageSafePreprocessor
from src.graph import build_graph
from src.model import IMHGN
from src.train import train_model, predict_proba
from src.evaluation import classification_metrics


def test_end_to_end_smoke():
    rng = np.random.default_rng(4)
    n = 50
    X = pd.DataFrame(
        {
            "age": rng.normal(55, 9, n),
            "sex": rng.integers(0, 2, n),
            "resting_bp": rng.normal(130, 15, n),
            "cholesterol": rng.normal(230, 35, n),
            "max_hr": rng.normal(145, 20, n),
            "exercise_angina": rng.integers(0, 2, n),
        }
    )
    score = (
        0.03 * (X["age"] - 50)
        + 0.01 * (X["cholesterol"] - 210)
        - 0.02 * (X["max_hr"] - 140)
        + 0.8 * X["exercise_angina"]
    )
    y = (score > np.median(score)).astype(int).to_numpy()

    train_idx = np.arange(0, 40)
    test_idx = np.arange(40, 50)

    pp = LeakageSafePreprocessor(
        {
            "clinical": ["resting_bp", "cholesterol", "max_hr"],
            "demographic": ["age", "sex"],
            "lifestyle": ["exercise_angina"],
        },
        random_state=42,
    )

    Xtr, ytr = pp.fit_resample_train(X.iloc[train_idx], y[train_idx])
    Xte = pp.transform(X.iloc[test_idx])

    gtr = build_graph(Xtr, ytr, k=3)
    gte = build_graph(Xte, y[test_idx], k=3)

    model = IMHGN(
        input_dim=pp.output_dim_,
        modality_slices=pp.modality_slices,
        hidden1=16,
        hidden2=8,
        dropout=0.1,
    )
    model, _ = train_model(model, gtr, None, epochs=5, patience=3, device="cpu")
    p = predict_proba(model, gte, device="cpu")

    assert len(p) == len(test_idx)
    assert np.all((p >= 0.0) & (p <= 1.0))
    m = classification_metrics(y[test_idx], p)
    assert 0.0 <= m["accuracy"] <= 1.0
