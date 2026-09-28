"""FINAL, FROZEN model (FINAL_MODEL_CONFIG.yaml), multi-seed headline
evaluation. This is Step 15 of the research protocol: the first genuine
test-set evaluation after freezing -- hyperparameters below are copied
verbatim from FINAL_MODEL_CONFIG.yaml and must not be changed based on the
results produced here.
"""
import json
from pathlib import Path

import numpy as np
import torch

from train_candidate_c import run as run_final_model

ROOT = Path(__file__).resolve().parent.parent
EMB_TAG = "fieldformer"
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

FROZEN_KWARGS = dict(
    epochs=15, d_model=192, n_layers=2, lr=0.00087941, batch_size=64, lam_recon=0.3537354196960241,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_dir = ROOT / "results" / "raw_seeds"
    agg_dir = ROOT / "results" / "aggregate"
    raw_dir.mkdir(parents=True, exist_ok=True)
    agg_dir.mkdir(parents=True, exist_ok=True)

    cls_results, anomaly_results = [], []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        cls_result, anomaly_result = run_final_model(EMB_TAG, device=device, use_focal=True, **FROZEN_KWARGS)
        cls_result["seed"] = seed
        anomaly_result["seed"] = seed
        cls_results.append(cls_result)
        anomaly_results.append(anomaly_result)
        print(f"seed={seed}: cls_macro_f1={cls_result['macro_f1']:.4f} "
              f"anomaly_f1={anomaly_result['test_f1']:.4f} anomaly_fpr={anomaly_result['test_fpr']:.4f} "
              f"anomaly_roc_auc={anomaly_result['roc_auc']:.4f}")

    with open(raw_dir / "final_model_classification.json", "w") as f:
        json.dump(cls_results, f, indent=2)
    with open(raw_dir / "final_model_anomaly.json", "w") as f:
        json.dump(anomaly_results, f, indent=2)

    cls_f1s = [r["macro_f1"] for r in cls_results]
    an_f1s = [r["test_f1"] for r in anomaly_results]
    an_fprs = [r["test_fpr"] for r in anomaly_results]
    an_aucs = [r["roc_auc"] for r in anomaly_results]
    summary = {
        "classification_macro_f1_mean": float(np.mean(cls_f1s)), "classification_macro_f1_std": float(np.std(cls_f1s)),
        "classification_macro_f1_ci95": float(1.96 * np.std(cls_f1s) / np.sqrt(len(cls_f1s))),
        "anomaly_f1_mean": float(np.mean(an_f1s)), "anomaly_f1_std": float(np.std(an_f1s)),
        "anomaly_fpr_mean": float(np.mean(an_fprs)), "anomaly_fpr_std": float(np.std(an_fprs)),
        "anomaly_roc_auc_mean": float(np.mean(an_aucs)), "anomaly_roc_auc_std": float(np.std(an_aucs)),
        "n_seeds": len(SEEDS),
    }
    with open(agg_dir / "final_model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
