from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE


@dataclass
class ModalityState:
    columns: List[str]
    category_maps: Dict[str, Dict[str, int]]
    imputer: IterativeImputer
    scaler: MinMaxScaler
    selector: Optional[RFE]
    pca: Optional[PCA]
    output_dim: int


class LeakageSafePreprocessor:
    """
    Fits all transformations on training data only and keeps modality blocks separate.

    Categorical columns are integer-coded using mappings learned from the training data.
    Unknown validation/test categories are encoded as NaN and subsequently imputed.
    """

    def __init__(
        self,
        modality_columns: Mapping[str, Sequence[str]],
        *,
        imputer_max_iter: int = 10,
        feature_range: Tuple[float, float] = (0.0, 1.0),
        use_rfe: bool = False,
        rfe_fraction: float = 0.8,
        use_pca: bool = False,
        pca_variance: float = 0.95,
        random_state: int = 42,
    ):
        self.modality_columns = {m: list(cols) for m, cols in modality_columns.items()}
        self.imputer_max_iter = int(imputer_max_iter)
        self.feature_range = tuple(feature_range)
        self.use_rfe = bool(use_rfe)
        self.rfe_fraction = float(rfe_fraction)
        self.use_pca = bool(use_pca)
        self.pca_variance = float(pca_variance)
        self.random_state = int(random_state)
        self.states_: Dict[str, ModalityState] = {}
        self.modality_slices_: Dict[str, slice] = {}
        self.output_dim_: int = 0

    @staticmethod
    def _fit_category_maps(df: pd.DataFrame, columns: Sequence[str]) -> Dict[str, Dict[str, int]]:
        maps = {}
        for c in columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                vals = pd.Series(df[c].dropna().astype(str).unique()).sort_values().tolist()
                maps[c] = {v: i for i, v in enumerate(vals)}
        return maps

    @staticmethod
    def _apply_category_maps(
        df: pd.DataFrame,
        columns: Sequence[str],
        maps: Dict[str, Dict[str, int]],
    ) -> np.ndarray:
        out = pd.DataFrame(index=df.index)
        for c in columns:
            s = df[c]
            if c in maps:
                out[c] = s.astype(str).map(maps[c]).where(s.notna(), np.nan)
            else:
                out[c] = pd.to_numeric(s, errors="coerce")
        return out.astype(float).to_numpy()

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LeakageSafePreprocessor":
        y = np.asarray(y).astype(int)
        offset = 0
        self.states_.clear()
        self.modality_slices_.clear()

        for modality, columns in self.modality_columns.items():
            cols = [c for c in columns if c in X.columns]
            if not cols:
                continue

            maps = self._fit_category_maps(X, cols)
            raw = self._apply_category_maps(X, cols, maps)

            imputer = IterativeImputer(
                max_iter=self.imputer_max_iter,
                random_state=self.random_state,
                sample_posterior=False,
                initial_strategy="median",
                skip_complete=True,
            )
            z = imputer.fit_transform(raw)

            scaler = MinMaxScaler(feature_range=self.feature_range)
            z = scaler.fit_transform(z)

            selector = None
            if self.use_rfe and z.shape[1] > 1 and len(np.unique(y)) > 1:
                n_select = max(1, int(round(z.shape[1] * self.rfe_fraction)))
                n_select = min(n_select, z.shape[1])
                estimator = LogisticRegression(max_iter=1000, solver="liblinear", random_state=self.random_state)
                selector = RFE(estimator=estimator, n_features_to_select=n_select, step=1)
                z = selector.fit_transform(z, y)

            pca = None
            if self.use_pca and z.shape[1] > 1:
                pca = PCA(n_components=self.pca_variance, svd_solver="full", random_state=self.random_state)
                z = pca.fit_transform(z)

            dim = int(z.shape[1])
            self.states_[modality] = ModalityState(
                columns=cols,
                category_maps=maps,
                imputer=imputer,
                scaler=scaler,
                selector=selector,
                pca=pca,
                output_dim=dim,
            )
            self.modality_slices_[modality] = slice(offset, offset + dim)
            offset += dim

        if offset == 0:
            raise ValueError("Preprocessor produced no modality features.")

        self.output_dim_ = offset
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.states_:
            raise RuntimeError("Preprocessor must be fitted before transform().")
        blocks = []
        for modality, state in self.states_.items():
            raw = self._apply_category_maps(X, state.columns, state.category_maps)
            z = state.imputer.transform(raw)
            z = state.scaler.transform(z)
            if state.selector is not None:
                z = state.selector.transform(z)
            if state.pca is not None:
                z = state.pca.transform(z)
            blocks.append(np.asarray(z, dtype=np.float32))
        return np.concatenate(blocks, axis=1).astype(np.float32)

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def fit_resample_train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        *,
        smote_k_neighbors: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        z = self.fit_transform(X, y)
        y = np.asarray(y).astype(int)
        counts = np.bincount(y)
        present = counts[counts > 0]
        if len(present) < 2:
            return z, y

        minority_count = int(present.min())
        if minority_count <= 1:
            return z, y

        k = min(int(smote_k_neighbors), minority_count - 1)
        smote = SMOTE(k_neighbors=max(1, k), random_state=self.random_state)
        zr, yr = smote.fit_resample(z, y)
        return np.asarray(zr, dtype=np.float32), np.asarray(yr, dtype=int)

    @property
    def modality_slices(self) -> Dict[str, slice]:
        return dict(self.modality_slices_)
