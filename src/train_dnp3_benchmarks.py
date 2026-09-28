"""Cross-protocol DNP3 classical/boosting benchmark run (see
build_dnp3_features.py docstring for scope note). Mirrors
train_classical_benchmarks.py's model set and protocol (5 seeds, macro-F1
primary) on the DNP3 flow-feature dataset instead of the Modbus window
dataset, to give a genuine (if feature-level-scoped) cross-protocol data
point for REPRODUCTION_REPORT.md Section 5.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix
import xgboost as xgb
import catboost as cb

ROOT = Path(__file__).resolve().parent.parent
FEAT_PARQUET = ROOT / "data_audit" / "dnp3_flow_features.parquet"
RAW_DIR = ROOT / "results" / "raw_seeds"
AGG_DIR = ROOT / "results" / "aggregate"
RAW_DIR.mkdir(parents=True, exist_ok=True)
AGG_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]


def main():
    df = pd.read_parquet(FEAT_PARQUET)
    feat_cols = [c for c in df.columns if c not in ("label", "source_file", "split")]

    le = LabelEncoder().fit(df["label"])
    df["y"] = le.transform(df["label"])
    classes_present_test = sorted(df.loc[df["split"] == "test", "y"].unique().tolist())
    label_names = list(le.classes_)

    Xtr, ytr = df[df.split == "train"][feat_cols].values, df[df.split == "train"]["y"].values
    Xte, yte = df[df.split == "test"][feat_cols].values, df[df.split == "test"]["y"].values
    print("train:", Xtr.shape, "test:", Xte.shape)
    print("classes with test support:", [label_names[i] for i in classes_present_test])

    models = {
        "LogisticRegression": lambda seed: LogisticRegression(max_iter=2000, random_state=seed),
        "RandomForest": lambda seed: RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed),
        "HistGradientBoosting": lambda seed: HistGradientBoostingClassifier(max_iter=300, random_state=seed),
        "XGBoost": lambda seed: xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=-1,
                                                   eval_metric="mlogloss", random_state=seed),
        "CatBoost": lambda seed: cb.CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, random_state=seed,
                                                        verbose=False, thread_count=-1),
    }
    needs_scaling = {"LogisticRegression"}

    summaries = []
    for name, factory in models.items():
        seed_results = []
        for seed in SEEDS:
            Xtr_use, Xte_use = Xtr, Xte
            if name in needs_scaling:
                scaler = StandardScaler().fit(Xtr)
                Xtr_use, Xte_use = scaler.transform(Xtr), scaler.transform(Xte)
            model = factory(seed)
            model.fit(Xtr_use, ytr)
            y_pred = model.predict(Xte_use)
            macro_f1 = f1_score(yte, y_pred, average="macro", labels=classes_present_test)
            bal_acc = balanced_accuracy_score(yte, y_pred)
            prec, rec, f1c, support = precision_recall_fscore_support(
                yte, y_pred, labels=classes_present_test, zero_division=0)
            per_class = {label_names[c]: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
                         for c, p, r, f, s in zip(classes_present_test, prec, rec, f1c, support)}
            seed_results.append({"macro_f1": float(macro_f1), "balanced_accuracy": float(bal_acc),
                                  "per_class": per_class, "seed": seed})
            print(f"{name} seed={seed}: macro_f1={macro_f1:.4f} bal_acc={bal_acc:.4f}")
        with open(RAW_DIR / f"dnp3_{name}.json", "w") as f:
            json.dump(seed_results, f, indent=2)
        f1s = [r["macro_f1"] for r in seed_results]
        summaries.append({"model": name, "macro_f1_mean": np.mean(f1s), "macro_f1_std": np.std(f1s)})

    summary_df = pd.DataFrame(summaries).sort_values("macro_f1_mean", ascending=False)
    summary_df.to_csv(AGG_DIR / "dnp3_benchmarks_summary.csv", index=False)
    print(summary_df)


if __name__ == "__main__":
    main()
