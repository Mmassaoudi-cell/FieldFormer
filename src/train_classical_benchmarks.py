"""Classical + boosting benchmark suite (Section 14 categories: Classical,
Tree/boosting) trained on the hand-crafted window_features.parquet.
Runs on CPU/GPU-light so it can proceed alongside BART training.

5 seeds per stochastic model; multiclass macro-F1 as primary metric plus
balanced accuracy, weighted F1, per-class P/R/F1, confusion matrix.
Outputs: results/raw_seeds/classical_<model>.json (all seeds),
         results/aggregate/classical_benchmarks_summary.csv
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

ROOT = Path(__file__).resolve().parent.parent
FEAT_PARQUET = ROOT / "data_audit" / "window_features.parquet"
RAW_DIR = ROOT / "results" / "raw_seeds"
AGG_DIR = ROOT / "results" / "aggregate"
RAW_DIR.mkdir(parents=True, exist_ok=True)
AGG_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
FEATURE_COLS = None  # filled at runtime


def load_split():
    df = pd.read_parquet(FEAT_PARQUET)
    global FEATURE_COLS
    FEATURE_COLS = [c for c in df.columns if c not in ("window_id", "class", "split")]
    le = LabelEncoder()
    df["y"] = le.fit_transform(df["class"])
    splits = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    return splits, le, FEATURE_COLS


def eval_preds(y_true, y_pred, label_names):
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1c, support = precision_recall_fscore_support(y_true, y_pred, labels=range(len(label_names)), zero_division=0)
    per_class = {label_names[i]: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1c[i]), "support": int(support[i])} for i in range(len(label_names))}
    cm = confusion_matrix(y_true, y_pred, labels=range(len(label_names))).tolist()
    return {
        "macro_f1": float(macro_f1), "weighted_f1": float(weighted_f1),
        "balanced_accuracy": float(bal_acc), "per_class": per_class, "confusion_matrix": cm,
    }


def make_model(name, seed):
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    if name == "SVM_RBF":
        return SVC(kernel="rbf", C=10.0, gamma="scale", random_state=seed)
    if name == "kNN":
        return KNeighborsClassifier(n_neighbors=7)
    if name == "DecisionTree":
        return DecisionTreeClassifier(max_depth=12, random_state=seed)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=seed)
    if name == "ExtraTrees":
        return ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(max_iter=300, random_state=seed)
    if name == "XGBoost":
        return xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=-1,
                                  eval_metric="mlogloss", random_state=seed)
    if name == "LightGBM":
        return lgb.LGBMClassifier(n_estimators=300, max_depth=-1, learning_rate=0.1, n_jobs=-1,
                                   random_state=seed, verbose=-1)
    if name == "CatBoost":
        return cb.CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, random_state=seed,
                                      verbose=False, thread_count=-1)
    raise ValueError(name)


NEEDS_SCALING = {"LogisticRegression", "SVM_RBF", "kNN"}
MODEL_NAMES = ["LogisticRegression", "SVM_RBF", "kNN", "DecisionTree", "RandomForest", "ExtraTrees",
               "HistGradientBoosting", "XGBoost", "LightGBM", "CatBoost"]


def main():
    splits, le, feat_cols = load_split()
    label_names = list(le.classes_)
    print("Classes:", label_names)
    print({k: len(v) for k, v in splits.items()})

    Xtr, ytr = splits["train"][feat_cols].values, splits["train"]["y"].values
    Xva, yva = splits["val"][feat_cols].values, splits["val"]["y"].values
    Xte, yte = splits["test"][feat_cols].values, splits["test"]["y"].values

    all_summaries = []
    for name in MODEL_NAMES:
        seed_results = []
        for seed in SEEDS:
            scaler = None
            Xtr_use, Xte_use = Xtr, Xte
            if name in NEEDS_SCALING:
                scaler = StandardScaler().fit(Xtr)
                Xtr_use = scaler.transform(Xtr)
                Xte_use = scaler.transform(Xte)
            model = make_model(name, seed)
            t0 = time.time()
            model.fit(Xtr_use, ytr)
            train_time = time.time() - t0
            t0 = time.time()
            y_pred = model.predict(Xte_use)
            infer_time = (time.time() - t0) / max(1, len(Xte_use))
            metrics = eval_preds(yte, y_pred, label_names)
            metrics["train_seconds"] = train_time
            metrics["inference_seconds_per_window"] = infer_time
            metrics["seed"] = seed
            seed_results.append(metrics)
            print(f"{name} seed={seed}: macro_f1={metrics['macro_f1']:.4f} bal_acc={metrics['balanced_accuracy']:.4f} ({train_time:.2f}s)")

        with open(RAW_DIR / f"classical_{name}.json", "w") as f:
            json.dump(seed_results, f, indent=2)

        macro_f1s = [r["macro_f1"] for r in seed_results]
        bal_accs = [r["balanced_accuracy"] for r in seed_results]
        all_summaries.append({
            "model": name,
            "macro_f1_mean": np.mean(macro_f1s), "macro_f1_std": np.std(macro_f1s),
            "balanced_accuracy_mean": np.mean(bal_accs), "balanced_accuracy_std": np.std(bal_accs),
            "train_seconds_mean": np.mean([r["train_seconds"] for r in seed_results]),
            "inference_seconds_per_window_mean": np.mean([r["inference_seconds_per_window"] for r in seed_results]),
        })

    summary_df = pd.DataFrame(all_summaries).sort_values("macro_f1_mean", ascending=False)
    summary_df.to_csv(AGG_DIR / "classical_benchmarks_summary.csv", index=False)
    print("\n=== Summary (sorted by macro F1) ===")
    print(summary_df.to_string(index=False))
    print(f"\nWrote {AGG_DIR / 'classical_benchmarks_summary.csv'}")


if __name__ == "__main__":
    main()
