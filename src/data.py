from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _canonicalize_name(name: str) -> str:
    name = str(name).strip().lower()
    for ch in [" ", "-", "/", "(", ")", ".", ":"]:
        name = name.replace(ch, "_")
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_")


def _normalize_aliases(aliases: Mapping[str, str]) -> Dict[str, str]:
    return {_canonicalize_name(k): _canonicalize_name(v) for k, v in aliases.items()}


@dataclass
class DatasetBundle:
    frame: pd.DataFrame
    feature_columns: List[str]
    target_column: str
    modality_columns: Dict[str, List[str]]
    source_column: str = "_source"


def load_csv(path: str | Path, source_name: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"Dataset is empty: {p}")
    df = df.copy()
    df.columns = [_canonicalize_name(c) for c in df.columns]
    df["_source"] = source_name
    return df


def apply_aliases(df: pd.DataFrame, aliases: Mapping[str, str]) -> pd.DataFrame:
    aliases = _normalize_aliases(aliases)
    rename_map = {}
    for c in df.columns:
        if c in aliases:
            rename_map[c] = aliases[c]
    return df.rename(columns=rename_map)


def _coerce_binary_target(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)

    if pd.api.types.is_numeric_dtype(s):
        vals = sorted(pd.Series(s.dropna().unique()).astype(float).tolist())
        if set(vals).issubset({0.0, 1.0}):
            return s.astype(float).round().astype("Int64")
        if len(vals) == 2:
            mapping = {vals[0]: 0, vals[1]: 1}
            return s.astype(float).map(mapping).astype("Int64")

    lowered = s.astype(str).str.strip().str.lower()
    negative = {"0", "normal", "no", "negative", "non-cvd", "non_cvd", "healthy", "false"}
    positive = {"1", "cvd", "heart disease", "heart_disease", "disease", "yes", "positive", "true"}
    mapped = lowered.map(lambda x: 0 if x in negative else (1 if x in positive else np.nan))
    if mapped.notna().all():
        return mapped.astype("Int64")
    raise ValueError("Target column cannot be deterministically mapped to binary labels 0/1.")


def harmonize_frames(
    frames: Sequence[pd.DataFrame],
    aliases: Mapping[str, str],
    target_column: str,
    modalities: Mapping[str, Sequence[str]],
) -> DatasetBundle:
    if not frames:
        raise ValueError("At least one dataframe is required.")

    target = _canonicalize_name(target_column)
    aliases = _normalize_aliases(aliases)
    normalized_modalities = {
        m: [_canonicalize_name(c) for c in cols] for m, cols in modalities.items()
    }

    prepared = []
    for df in frames:
        x = apply_aliases(df, aliases).copy()
        if target not in x.columns:
            raise KeyError(f"Target column '{target}' not found after alias normalization.")
        x[target] = _coerce_binary_target(x[target])
        prepared.append(x)

    modality_columns: Dict[str, List[str]] = {}
    requested_features: List[str] = []
    for modality, cols in normalized_modalities.items():
        available = [c for c in cols if any(c in df.columns for df in prepared)]
        if available:
            modality_columns[modality] = available
            requested_features.extend(available)

    requested_features = list(dict.fromkeys(requested_features))
    if not requested_features:
        raise ValueError("None of the configured modality features were found in the datasets.")

    aligned = []
    for df in prepared:
        keep = requested_features + [target, "_source"]
        for c in requested_features:
            if c not in df.columns:
                df[c] = np.nan
        aligned.append(df[keep])

    merged = pd.concat(aligned, axis=0, ignore_index=True)
    merged = merged.dropna(subset=[target]).reset_index(drop=True)
    merged[target] = merged[target].astype(int)

    return DatasetBundle(
        frame=merged,
        feature_columns=requested_features,
        target_column=target,
        modality_columns=modality_columns,
    )


def load_from_config(cfg: Mapping) -> DatasetBundle:
    data_cfg = cfg["data"]
    frames = []
    df1_path = str(data_cfg.get("df1_path", "") or "").strip()
    df2_path = str(data_cfg.get("df2_path", "") or "").strip()

    if df1_path:
        frames.append(load_csv(df1_path, "DF1"))
    if df2_path:
        frames.append(load_csv(df2_path, "DF2"))
    if not frames:
        raise ValueError("No dataset path is configured. Set data.df1_path and/or data.df2_path.")

    return harmonize_frames(
        frames=frames,
        aliases=data_cfg.get("aliases", {}),
        target_column=data_cfg.get("target_column", "target"),
        modalities=data_cfg.get("modalities", {}),
    )
