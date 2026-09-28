"""Anomaly-detection baselines matching the source paper's Table VI
comparison (Isolation Forest, One-Class SVM), trained benign-only on the
hand-crafted window_features.parquet, with the same calibration-threshold
protocol as our LSTM-autoencoder reproduction (train/calibrate on benign
train+val, threshold chosen on calibration by max-F1, evaluated on test).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parent.parent
FEAT_PARQUET = ROOT / "data_audit" / "window_features.parquet"
RAW_DIR = ROOT / "results" / "raw_seeds"
AGG_DIR = ROOT / "results" / "aggregate"
RAW_DIR.mkdir(parents=True, exist_ok=True)
AGG_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]


def load():
    df = pd.read_parquet(FEAT_PARQUET)
    feat_cols = [c for c in df.columns if c not in ("window_id", "class", "split")]
    return df, feat_cols


def run_model(name, df, feat_cols, seed):
    benign_train = df[(df["class"] == "benign") & (df["split"] == "train")]
    benign_calib = df[(df["class"] == "benign") & (df["split"] == "val")]
    attack_calib = df[(df["class"] != "benign") & (df["split"] == "val")]
    benign_test = df[(df["class"] == "benign") & (df["split"] == "test")]
    attack_test = df[(df["class"] != "benign") & (df["split"] == "test")]

    scaler = StandardScaler().fit(benign_train[feat_cols].values)
    Xtr = scaler.transform(benign_train[feat_cols].values)

    if name == "IsolationForest":
        model = IsolationForest(n_estimators=200, contamination="auto", random_state=seed)
        model.fit(Xtr)
        score_fn = lambda X: -model.score_samples(X)  # higher = more anomalous
    elif name == "OneClassSVM":
        model = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
        model.fit(Xtr)
        score_fn = lambda X: -model.decision_function(X)
    else:
        raise ValueError(name)

    def scores(sub):
        if len(sub) == 0:
            return np.array([])
        return score_fn(scaler.transform(sub[feat_cols].values))

    s_benign_calib = scores(benign_calib)
    s_attack_calib = scores(attack_calib)
    s_benign_test = scores(benign_test)
    s_attack_test = scores(attack_test)

    all_calib = np.concatenate([s_benign_calib, s_attack_calib])
    calib_labels = np.concatenate([np.zeros(len(s_benign_calib)), np.ones(len(s_attack_calib))])
    candidates = np.quantile(all_calib, np.linspace(0.01, 0.99, 199))
    best_thr, best_f1 = None, -1
    for thr in candidates:
        pred = (all_calib > thr).astype(int)
        f1 = f1_score(calib_labels, pred)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    y_true = np.concatenate([np.zeros(len(s_benign_test)), np.ones(len(s_attack_test))])
    y_score = np.concatenate([s_benign_test, s_attack_test])
    y_pred = (y_score > best_thr).astype(int)

    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    roc_auc = roc_auc_score(y_true, y_score) if len(set(y_true.tolist())) > 1 else float("nan")
    pr_auc = average_precision_score(y_true, y_score) if len(set(y_true.tolist())) > 1 else float("nan")

    return {
        "test_f1": float(f1), "test_precision": float(precision), "test_recall": float(recall),
        "test_fpr": float(fpr), "test_fnr": float(fnr), "threshold": float(best_thr),
        "calib_f1": float(best_f1), "tn": tn, "fp": fp, "fn": fn, "tp": tp, "seed": seed,
        "roc_auc": float(roc_auc), "pr_auc": float(pr_auc),
    }


def main():
    df, feat_cols = load()
    summaries = []
    for name in ["IsolationForest", "OneClassSVM"]:
        seed_results = []
        for seed in SEEDS:
            t0 = time.time()
            r = run_model(name, df, feat_cols, seed)
            r["train_seconds"] = time.time() - t0
            seed_results.append(r)
            print(f"{name} seed={seed}: f1={r['test_f1']:.4f} recall={r['test_recall']:.4f} fpr={r['test_fpr']:.4f} roc_auc={r['roc_auc']:.4f} pr_auc={r['pr_auc']:.4f}")
        with open(RAW_DIR / f"anomaly_{name}.json", "w") as f:
            json.dump(seed_results, f, indent=2)
        f1s = [r["test_f1"] for r in seed_results]
        summaries.append({
            "model": name, "test_f1_mean": np.mean(f1s), "test_f1_std": np.std(f1s),
            "test_recall_mean": np.mean([r["test_recall"] for r in seed_results]),
            "test_fpr_mean": np.mean([r["test_fpr"] for r in seed_results]),
            "roc_auc_mean": np.mean([r["roc_auc"] for r in seed_results]),
            "pr_auc_mean": np.mean([r["pr_auc"] for r in seed_results]),
        })
    pd.DataFrame(summaries).to_csv(AGG_DIR / "anomaly_benchmarks_summary.csv", index=False)
    print(pd.DataFrame(summaries))


if __name__ == "__main__":
    main()
