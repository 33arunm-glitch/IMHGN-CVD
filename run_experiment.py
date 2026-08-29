from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.data import load_from_config
from src.preprocessing import LeakageSafePreprocessor
from src.graph import build_graph
from src.model import IMHGN
from src.optimization import AdaptiveBatOptimizer
from src.train import set_global_seed, train_model, predict_proba
from src.evaluation import (
    classification_metrics,
    calibration_table,
    bootstrap_ci,
    paired_wilcoxon,
)


def make_preprocessor(cfg, modality_columns, seed):
    p = cfg["preprocessing"]
    return LeakageSafePreprocessor(
        modality_columns=modality_columns,
        imputer_max_iter=p.get("iterative_imputer_max_iter", 10),
        feature_range=tuple(p.get("minmax_range", [0.0, 1.0])),
        use_rfe=p.get("use_rfe", False),
        rfe_fraction=p.get("rfe_fraction", 0.8),
        use_pca=p.get("use_pca", False),
        pca_variance=p.get("pca_variance", 0.95),
        random_state=seed,
    )


def params_from_cfg(cfg):
    m = cfg["model"]
    t = cfg["training"]
    return {
        "hidden1": int(m["hidden_dims"][0]),
        "hidden2": int(m["hidden_dims"][1]),
        "dropout": float(m["dropout"]),
        "learning_rate": float(t["learning_rate"]),
        "weight_decay": float(t["weight_decay"]),
        "epochs": int(t["epochs"]),
    }


def build_model(preprocessor, params, learnable_fusion=True):
    return IMHGN(
        input_dim=preprocessor.output_dim_,
        modality_slices=preprocessor.modality_slices,
        hidden1=int(params["hidden1"]),
        hidden2=int(params["hidden2"]),
        dropout=float(params["dropout"]),
        learnable_fusion=learnable_fusion,
    )


def evaluate_candidate_inner_cv(
    candidate,
    X,
    y,
    modality_columns,
    cfg,
    seed,
):
    n_splits = int(cfg["cross_validation"].get("inner_folds", 5))
    n_splits = min(n_splits, int(np.bincount(y).min()))
    if n_splits < 2:
        return 0.0
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []

    for inner_id, (tr, va) in enumerate(skf.split(X, y)):
        fold_seed = seed + inner_id
        set_global_seed(fold_seed)
        pp = make_preprocessor(cfg, modality_columns, fold_seed)
        Xtr, ytr = pp.fit_resample_train(
            X.iloc[tr],
            y[tr],
            smote_k_neighbors=cfg["preprocessing"].get("smote_k_neighbors", 5),
        )
        Xva = pp.transform(X.iloc[va])

        graph_cfg = cfg["graph"]
        gtr = build_graph(
            Xtr,
            ytr,
            k=graph_cfg.get("k", 5),
            metric=graph_cfg.get("metric", "cosine"),
            weighted=graph_cfg.get("weighted", True),
            undirected=graph_cfg.get("undirected", True),
        )
        gva = build_graph(
            Xva,
            y[va],
            k=graph_cfg.get("k", 5),
            metric=graph_cfg.get("metric", "cosine"),
            weighted=graph_cfg.get("weighted", True),
            undirected=graph_cfg.get("undirected", True),
        )
        model = build_model(pp, candidate, learnable_fusion=True)
        model, _ = train_model(
            model,
            gtr,
            gva,
            learning_rate=float(candidate["learning_rate"]),
            weight_decay=float(candidate["weight_decay"]),
            epochs=int(candidate["epochs"]),
            patience=int(cfg["training"].get("patience", 20)),
        )
        p = predict_proba(model, gva)
        if len(np.unique(y[va])) == 2:
            scores.append(roc_auc_score(y[va], p))
        else:
            pred = (p >= 0.5).astype(int)
            scores.append(float(np.mean(pred == y[va])))

    return float(np.mean(scores))


def optimize_params(X, y, modality_columns, cfg, seed):
    if not cfg["abo"].get("enabled", True):
        return params_from_cfg(cfg), [], None

    a = cfg["abo"]
    optimizer = AdaptiveBatOptimizer(
        a["search_space"],
        population_size=a.get("population_size", 8),
        iterations=a.get("iterations", 8),
        loudness=a.get("loudness", 0.9),
        pulse_rate=a.get("pulse_rate", 0.1),
        alpha=a.get("alpha", 0.95),
        gamma=a.get("gamma", 0.9),
        frequency_min=a.get("frequency_min", 0.0),
        frequency_max=a.get("frequency_max", 2.0),
        seed=seed,
    )

    def objective(candidate):
        return evaluate_candidate_inner_cv(candidate, X, y, modality_columns, cfg, seed)

    best, fitness, history = optimizer.optimize(objective)
    # Keep architecture at two GCN layers; candidate controls dimensions, not layer count.
    return best, history, fitness


def logistic_baseline(train_x, train_y, test_x):
    model = LogisticRegression(max_iter=2000, solver="liblinear", random_state=42)
    model.fit(train_x, train_y)
    return model.predict_proba(test_x)[:, 1]


def evaluate_outer_variant(
    variant,
    X_train,
    y_train,
    X_test,
    y_test,
    modality_columns,
    cfg,
    seed,
    tuned_params=None,
):
    local_cfg = json.loads(json.dumps(cfg))
    graph_metric = local_cfg["graph"].get("metric", "cosine")
    graph_k = int(local_cfg["graph"].get("k", 5))
    no_graph = False
    learnable_fusion = True

    if variant == "no_graph":
        no_graph = True
    elif variant == "no_fusion":
        learnable_fusion = False
    elif variant == "euclidean_graph":
        graph_metric = "euclidean"
    elif variant == "k3":
        graph_k = 3
    elif variant == "k5":
        graph_k = 5
    elif variant == "k7":
        graph_k = 7

    if variant == "no_abo" or tuned_params is None:
        params = params_from_cfg(local_cfg)
    else:
        params = tuned_params

    pp = make_preprocessor(local_cfg, modality_columns, seed)
    Xtr, ytr = pp.fit_resample_train(
        X_train,
        y_train,
        smote_k_neighbors=local_cfg["preprocessing"].get("smote_k_neighbors", 5),
    )
    Xte = pp.transform(X_test)

    gtr = build_graph(
        Xtr,
        ytr,
        k=graph_k,
        metric=graph_metric,
        weighted=local_cfg["graph"].get("weighted", True),
        undirected=local_cfg["graph"].get("undirected", True),
        no_graph=no_graph,
    )
    gte = build_graph(
        Xte,
        y_test,
        k=graph_k,
        metric=graph_metric,
        weighted=local_cfg["graph"].get("weighted", True),
        undirected=local_cfg["graph"].get("undirected", True),
        no_graph=no_graph,
    )

    model = build_model(pp, params, learnable_fusion=learnable_fusion)
    model, _ = train_model(
        model,
        gtr,
        None,
        learning_rate=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
        epochs=int(params["epochs"]),
        patience=int(local_cfg["training"].get("patience", 20)),
    )
    p = predict_proba(model, gte)
    return p, pp, model, Xtr, ytr, Xte


def run(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)
    bundle = load_from_config(cfg)

    X = bundle.frame[bundle.feature_columns].copy()
    y = bundle.frame[bundle.target_column].to_numpy(dtype=int)

    out_dir = Path(cfg["output"].get("directory", "./outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    n_outer = int(cfg["cross_validation"].get("outer_folds", 5))
    n_outer = min(n_outer, int(np.bincount(y).min()))
    if n_outer < 2:
        raise ValueError("Not enough observations per class for stratified cross-validation.")

    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)

    rows = []
    predictions = []
    baseline_scores = []
    full_scores = []
    ablation_rows = []

    for fold, (tr, te) in enumerate(outer.split(X, y), start=1):
        fold_seed = seed + 1000 * fold
        Xtr_raw = X.iloc[tr].reset_index(drop=True)
        Xte_raw = X.iloc[te].reset_index(drop=True)
        ytr_raw, yte = y[tr], y[te]

        best_params, history, best_inner = optimize_params(
            Xtr_raw, ytr_raw, bundle.modality_columns, cfg, fold_seed
        )

        p, pp, model, Xtr, ytr, Xte = evaluate_outer_variant(
            "full",
            Xtr_raw,
            ytr_raw,
            Xte_raw,
            yte,
            bundle.modality_columns,
            cfg,
            fold_seed,
            tuned_params=best_params,
        )

        metrics = classification_metrics(yte, p)
        metrics["fold"] = fold
        metrics["inner_objective"] = float(best_inner) if best_inner is not None else float("nan")
        metrics["fusion_weights"] = json.dumps(model.modality_weights())
        rows.append(metrics)
        full_scores.append(metrics["accuracy"])

        bp = logistic_baseline(Xtr, ytr, Xte)
        bmetrics = classification_metrics(yte, bp)
        baseline_scores.append(bmetrics["accuracy"])

        for idx, prob in zip(te, p):
            predictions.append(
                {
                    "row_index": int(idx),
                    "fold": fold,
                    "y_true": int(y[idx]),
                    "y_prob": float(prob),
                    "y_pred": int(prob >= 0.5),
                }
            )

        if cfg.get("ablation", {}).get("enabled", True):
            for variant in cfg["ablation"].get("variants", []):
                if variant == "full":
                    vm = metrics
                else:
                    vp, *_ = evaluate_outer_variant(
                        variant,
                        Xtr_raw,
                        ytr_raw,
                        Xte_raw,
                        yte,
                        bundle.modality_columns,
                        cfg,
                        fold_seed + 77,
                        tuned_params=best_params,
                    )
                    vm = classification_metrics(yte, vp)
                ablation_rows.append(
                    {
                        "fold": fold,
                        "variant": variant,
                        **{k: v for k, v in vm.items() if k not in {"fold", "fusion_weights", "inner_objective"}},
                    }
                )

        hist_path = out_dir / f"abo_history_fold_{fold}.json"
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "outer_fold_metrics.csv", index=False)

    pred_df = pd.DataFrame(predictions).sort_values("row_index")
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    cal = calibration_table(pred_df["y_true"], pred_df["y_prob"], n_bins=10)
    cal.to_csv(out_dir / "calibration.csv", index=False)

    stat = {
        "wilcoxon_accuracy_vs_logistic_regression": paired_wilcoxon(
            full_scores,
            baseline_scores,
            alpha=cfg["statistics"].get("alpha", 0.05),
        ),
        "accuracy_bootstrap_ci": bootstrap_ci(
            full_scores,
            n_boot=cfg["statistics"].get("bootstrap_samples", 1000),
            alpha=cfg["statistics"].get("alpha", 0.05),
            seed=seed,
        ),
    }
    with open(out_dir / "statistical_tests.json", "w", encoding="utf-8") as f:
        json.dump(stat, f, indent=2)

    if ablation_rows:
        abd = pd.DataFrame(ablation_rows)
        full_mean = float(abd.loc[abd["variant"] == "full", "accuracy"].mean())
        summary_ab = (
            abd.groupby("variant", as_index=False)["accuracy"].mean()
            .rename(columns={"accuracy": "mean_accuracy"})
        )
        summary_ab["delta_vs_full"] = summary_ab["mean_accuracy"] - full_mean
        summary_ab.to_csv(out_dir / "ablation.csv", index=False)

    summary = {
        "n_samples": int(len(bundle.frame)),
        "class_counts": {str(k): int(v) for k, v in bundle.frame[bundle.target_column].value_counts().sort_index().items()},
        "modalities": bundle.modality_columns,
        "outer_folds": n_outer,
        "mean_metrics": {
            c: float(metrics_df[c].mean())
            for c in ["accuracy", "precision", "recall", "f1", "roc_auc", "kappa", "brier"]
        },
        "std_metrics": {
            c: float(metrics_df[c].std(ddof=1))
            for c in ["accuracy", "precision", "recall", "f1", "roc_auc", "kappa", "brier"]
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nResults written to: {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IMHGN CVD reproducibility experiment.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration.")
    args = parser.parse_args()
    run(args.config)
