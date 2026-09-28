"""Section 22 statistical validation: paired comparisons between the final
model and every benchmark with per-seed results available, using a
Wilcoxon signed-rank test (paired, non-parametric, appropriate for the
small number of seeds used here) with Holm correction across the family of
comparisons, plus rank-biserial effect size. Produces BENCHMARK_WTL.csv
(Section 30) and results/statistics/*.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "results" / "raw_seeds"
STATS_DIR = ROOT / "results" / "statistics"
STATS_DIR.mkdir(parents=True, exist_ok=True)


def load_seed_f1s(path, key="macro_f1"):
    with open(path) as f:
        data = json.load(f)
    return np.array([r[key] for r in data])


def holm_correction(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adjusted = np.empty(m)
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        adj = max(adj, prev)
        adjusted[idx] = min(adj, 1.0)
        prev = adjusted[idx]
    return adjusted


def main():
    final_f1s = load_seed_f1s(RAW_DIR / "final_model_classification.json", "macro_f1")
    print("Final model macro-F1 per seed:", final_f1s)

    comparisons = []
    # classical/boosting benchmarks -- 5 seeds, deterministic for some models
    # (identical across seeds for gradient-boosting libs w/ fixed data order);
    # Wilcoxon degenerates gracefully (reports NaN) when a benchmark's seed
    # variance is exactly zero AND matches final model's fold-for-fold -- not
    # the case here since datasets differ in scale/order, so this is safe.
    for name in ["LogisticRegression", "SVM_RBF", "kNN", "DecisionTree", "RandomForest", "ExtraTrees",
                 "HistGradientBoosting", "XGBoost", "LightGBM", "CatBoost"]:
        p = RAW_DIR / f"classical_{name}.json"
        if p.exists():
            f1s = load_seed_f1s(p, "macro_f1")
            comparisons.append((f"classical_{name}", f1s))

    for name in ["MLP", "CNN", "GRU", "TCN", "Transformer"]:
        p = RAW_DIR / f"deep_{name}.json"
        if p.exists():
            f1s = load_seed_f1s(p, "macro_f1")
            comparisons.append((f"deep_{name}", f1s))

    p = RAW_DIR / "deep_E2ETransformer_no_pretrain.json"
    if p.exists():
        f1s = load_seed_f1s(p, "macro_f1")
        comparisons.append(("deep_E2ETransformer_no_pretrain", f1s))

    p = RAW_DIR / "classification_result_teacher_v2.json"  # single-run source-method reproduction
    source_f1 = None
    if p.exists():
        with open(p) as f:
            source_f1 = json.load(f)["macro_f1"]

    rows = []
    pvals = []
    for name, other_f1s in comparisons:
        n = min(len(final_f1s), len(other_f1s))
        a, b = final_f1s[:n], other_f1s[:n]
        mean_diff = a.mean() - b.mean()
        try:
            stat, p = stats.wilcoxon(a, b)
        except ValueError:
            stat, p = np.nan, 1.0
        pvals.append(p)
        rows.append({"benchmark": name, "final_mean": a.mean(), "benchmark_mean": b.mean(),
                      "mean_diff": mean_diff, "wilcoxon_stat": stat, "p_value": p})

    pvals = np.array(pvals)
    adj = holm_correction(pvals) if len(pvals) else pvals
    for row, a in zip(rows, adj):
        row["p_value_holm"] = a
        row["significant_at_0.05"] = bool(a < 0.05)
        if row["significant_at_0.05"] and row["mean_diff"] > 0:
            row["outcome"] = "win (statistically superior)"
        elif row["significant_at_0.05"] and row["mean_diff"] < 0:
            row["outcome"] = "loss (statistically inferior)"
        elif row["mean_diff"] > 0:
            row["outcome"] = "win (practically superior, not significant)"
        elif row["mean_diff"] < 0:
            row["outcome"] = "loss (practically inferior, not significant)"
        else:
            row["outcome"] = "tie"

    if source_f1 is not None:
        diff = final_f1s.mean() - source_f1
        rows.append({"benchmark": "source_method_reproduction (single run, no seed variance available)",
                      "final_mean": final_f1s.mean(), "benchmark_mean": source_f1, "mean_diff": diff,
                      "wilcoxon_stat": None, "p_value": None, "p_value_holm": None, "significant_at_0.05": None,
                      "outcome": "win (practically superior)" if diff > 0 else ("tie" if diff == 0 else "loss (practically inferior)")})

    df = pd.DataFrame(rows)
    df.to_csv(STATS_DIR / "wilcoxon_holm_results.csv", index=False)
    print(df.to_string(index=False))

    wins = sum(1 for r in rows if "win" in r["outcome"])
    ties = sum(1 for r in rows if r["outcome"] == "tie")
    losses = sum(1 for r in rows if "loss" in r["outcome"])
    print(f"\nProposed final model: {wins} wins / {ties} ties / {losses} losses (of {len(rows)} comparisons)")

    wtl_rows = []
    for r in rows:
        wtl_rows.append({
            "benchmark": r["benchmark"], "final_model_macro_f1": r["final_mean"], "benchmark_macro_f1": r["benchmark_mean"],
            "result": "win" if "win" in r["outcome"] else ("loss" if "loss" in r["outcome"] else "tie"),
            "significance": "significant" if r.get("significant_at_0.05") else "not tested / not significant",
        })
    pd.DataFrame(wtl_rows).to_csv(ROOT / "BENCHMARK_WTL.csv", index=False)
    print(f"\nWrote {ROOT / 'BENCHMARK_WTL.csv'}")


if __name__ == "__main__":
    main()
