"""Multi-seed held-out-class generalization sweep, addressing peer-review
Required Revisions R1+R2 (peer_review/06_editorial_decision.md): (a) a fresh
attack class never used during candidate selection or Optuna tuning
(tcp_syn_flood), evaluated post-freeze as a confirmatory result, reported
separately from (b) the two classes used during development (mitm,
icmp_flood, here re-run with proper seed repetition instead of the
single-run point estimates originally reported).
"""
import json
from pathlib import Path

import numpy as np
import torch

from eval_heldout_class import eval_source_method, eval_candidate_a, eval_candidate_b, eval_candidate_c

ROOT = Path(__file__).resolve().parent.parent
EMB_TAG = "fieldformer"
SEEDS = [0, 1, 2, 3, 4]

# mitm, icmp_flood: used during candidate screening/selection (Stage 2-3) -> "selection"
# tcp_syn_flood: never touched during screening, tuning, or the original Table III -> "confirmatory"
CLASSES = {
    "mitm": "selection",
    "icmp_flood": "selection",
    "tcp_syn_flood": "confirmatory",
}

METHODS = {
    "source_method_lstm_ae": eval_source_method,
    "candidate_a": eval_candidate_a,
    "candidate_b": eval_candidate_b,
    "candidate_c": eval_candidate_c,
}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = ROOT / "results" / "raw_seeds"
    out_dir.mkdir(parents=True, exist_ok=True)
    agg_dir = ROOT / "results" / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for cls, tag in CLASSES.items():
        for method_name, fn in METHODS.items():
            per_seed = []
            for seed in SEEDS:
                torch.manual_seed(seed)
                np.random.seed(seed)
                r = fn(EMB_TAG, cls, device)
                r["seed"] = seed
                per_seed.append(r)
                print(f"[{tag}] class={cls} method={method_name} seed={seed}: "
                      f"roc_auc={r['roc_auc']}, detect_rate={r['holdout_detection_rate_pct']:.2f}%, f1={r['test_f1']:.4f}")
            with open(out_dir / f"heldout_{cls}_{method_name}_{tag}_multiseed.json", "w") as f:
                json.dump(per_seed, f, indent=2)

            aucs = [r["roc_auc"] for r in per_seed if r["roc_auc"] is not None]
            f1s = [r["test_f1"] for r in per_seed]
            drs = [r["holdout_detection_rate_pct"] for r in per_seed]
            all_rows.append({
                "class": cls, "class_role": tag, "method": method_name,
                "roc_auc_mean": float(np.mean(aucs)) if aucs else None,
                "roc_auc_std": float(np.std(aucs)) if aucs else None,
                "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
                "detect_rate_mean": float(np.mean(drs)), "detect_rate_std": float(np.std(drs)),
                "n_seeds": len(per_seed),
            })

    import pandas as pd
    df = pd.DataFrame(all_rows)
    df.to_csv(agg_dir / "heldout_multiseed_summary.csv", index=False)
    print("\n=== Summary ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
