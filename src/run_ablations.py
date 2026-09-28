"""Section 19 ablations for the selected final model (Candidate C /
BalancedFusion on the FieldFormer backbone): each named module removed one
at a time, plus a backbone-swap ablation against Candidate A's plain SSM
(no masked-field pretraining) at matched downstream parameter count.
"""
import json
from pathlib import Path

import numpy as np
import torch

from train_candidate_c import run as run_candidate_c
from train_candidate_a import train_classifier as run_candidate_a_classifier

ROOT = Path(__file__).resolve().parent.parent
EMB_TAG = "fieldformer"
SEEDS = [0, 1, 2]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = ROOT / "results" / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        "full_model": dict(use_attention_pool=True, use_recon=True, use_focal=True),
        "no_attention_pool": dict(use_attention_pool=False, use_recon=True, use_focal=True),
        "no_focal_loss": dict(use_attention_pool=True, use_recon=True, use_focal=False),
        "no_joint_recon": dict(use_attention_pool=True, use_recon=False, use_focal=True),
    }

    rows = []
    for name, kwargs in variants.items():
        f1s = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            cls_result, anomaly_result = run_candidate_c(EMB_TAG, device=device, **kwargs)
            f1s.append(cls_result["macro_f1"])
            print(f"{name} seed={seed}: macro_f1={cls_result['macro_f1']:.4f}"
                  + (f" anomaly_f1={anomaly_result['test_f1']:.4f} anomaly_fpr={anomaly_result['test_fpr']:.4f}" if anomaly_result else " (no anomaly head)"))
        rows.append({"variant": name, "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s))})

    # backbone-swap ablation: Candidate A's plain (non-masked-pretrained) SSM
    # backbone at the same downstream scale, isolating FieldFormer's own
    # contribution from BalancedFusion's head/training-objective contribution.
    f1s = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        r = run_candidate_a_classifier(EMB_TAG, device=device)
        f1s.append(r["macro_f1"])
        print(f"backbone_swap_candidateA seed={seed}: macro_f1={r['macro_f1']:.4f}")
    rows.append({"variant": "backbone_swap_candidateA_plain_ssm", "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s))})

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ablation_results.csv", index=False)
    print(df)


if __name__ == "__main__":
    main()
