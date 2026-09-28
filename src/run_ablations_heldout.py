"""Extends the ablation study (run_ablations.py) with held-out-class ROC-AUC
for each variant, addressing peer-review Required Revision R4: the original
ablation table only reported classification macro-F1, so the paper's causal
claim about joint training's effect on generalization was argued using a
between-candidate comparison (Candidate A vs C) rather than a properly
controlled within-ablation comparison. This isolates the effect by holding
the backbone fixed and toggling only one mechanism at a time, measured on
the confirmatory held-out class (tcp_syn_flood, never used in selection).
"""
import json
from pathlib import Path

import numpy as np
import torch

from eval_heldout_class import eval_candidate_c

ROOT = Path(__file__).resolve().parent.parent
EMB_TAG = "fieldformer"
HOLDOUT_CLASS = "tcp_syn_flood"  # confirmatory class, matches run_heldout_multiseed.py
SEEDS = [0, 1, 2]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    variants = {
        "full_model": dict(use_attention_pool=True, use_recon=True, use_focal=True),
        "no_attention_pool": dict(use_attention_pool=False, use_recon=True, use_focal=True),
        "no_focal_loss": dict(use_attention_pool=True, use_recon=True, use_focal=False),
        "no_joint_recon": dict(use_attention_pool=True, use_recon=False, use_focal=True),
    }

    rows = []
    for name, kwargs in variants.items():
        aucs, drs = [], []
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            r = eval_candidate_c(EMB_TAG, HOLDOUT_CLASS, device, **kwargs)
            if r is None:
                print(f"{name} seed={seed}: no anomaly head -> held-out ROC-AUC undefined")
                continue
            aucs.append(r["roc_auc"])
            drs.append(r["holdout_detection_rate_pct"])
            print(f"{name} seed={seed}: held_out_roc_auc={r['roc_auc']:.4f} detect_rate={r['holdout_detection_rate_pct']:.2f}%")
        rows.append({
            "variant": name,
            "heldout_roc_auc_mean": float(np.mean(aucs)) if aucs else None,
            "heldout_roc_auc_std": float(np.std(aucs)) if aucs else None,
            "heldout_detect_rate_mean": float(np.mean(drs)) if drs else None,
            "n_seeds": len(aucs),
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    out_dir = ROOT / "results" / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "ablation_heldout_results.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
