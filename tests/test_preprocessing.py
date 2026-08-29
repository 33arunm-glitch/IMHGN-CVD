import numpy as np
import pandas as pd

from src.preprocessing import LeakageSafePreprocessor


def test_preprocessing_and_smote():
    rng = np.random.default_rng(7)
    n = 40
    X = pd.DataFrame(
        {
            "age": rng.normal(55, 8, n),
            "sex": rng.integers(0, 2, n),
            "cholesterol": rng.normal(220, 30, n),
            "exercise_angina": rng.integers(0, 2, n),
        }
    )
    X.loc[0, "cholesterol"] = np.nan
    y = np.array([0] * 28 + [1] * 12)

    pp = LeakageSafePreprocessor(
        {
            "clinical": ["cholesterol"],
            "demographic": ["age", "sex"],
            "lifestyle": ["exercise_angina"],
        },
        random_state=42,
    )

    z, yr = pp.fit_resample_train(X, y, smote_k_neighbors=5)
    assert np.isfinite(z).all()
    assert z.shape[1] == 4
    assert np.bincount(yr)[0] == np.bincount(yr)[1]

    z2 = pp.transform(X.iloc[:5])
    assert z2.shape == (5, 4)
    assert np.isfinite(z2).all()
