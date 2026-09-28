"""Cross-protocol DNP3 generalization check using the genuine DNP3 intrusion
dataset's own pre-computed, per-flow labeled CIC-FlowMeter features (9
classes: NORMAL + 8 attack types across the 9 real attack-scenario
captures). This is a SCOPE REDUCTION from the full raw-byte LM pipeline used
for Modbus (see REPRODUCTION_REPORT.md): given session time constraints, we
use the dataset's own labeled flow-feature CSVs directly with the classical/
boosting benchmark family, rather than re-deriving raw-byte windows and
retraining a second BART instance for DNP3. This still gives a genuine,
real cross-protocol data point -- just at the engineered-feature level
(paper's Table IX style comparison, restricted to the non-LM representation
family) rather than the full LM-embedding level.

Leakage control: split is by SOURCE FILE (each capture file goes entirely to
one split), not by row, since flows from the same capture are correlated.
"""
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DNP3_ROOT = r"C:/rd/dnp3"
OUT_PARQUET = ROOT / "data_audit" / "dnp3_flow_features.parquet"

DROP_COLS = {"Unnamed: 0", "Flow ID", "Src IP", "Dst IP", "Timestamp", "Label"}


def main():
    files = glob.glob(f"{DNP3_ROOT}/*/CSV Files/120_timeout/**/*.csv", recursive=True)
    print(f"Found {len(files)} candidate CSVs")

    rng = np.random.RandomState(1337)
    file_split = {}

    all_rows = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception as e:
            continue
        if "Label" not in df.columns:
            continue
        fname = os.path.basename(f)
        if fname not in file_split:
            r = rng.rand()
            file_split[fname] = "train" if r < 0.70 else ("val" if r < 0.85 else "test")
        split = file_split[fname]

        feat_cols = [c for c in df.columns if c not in DROP_COLS]
        sub = df[feat_cols].apply(pd.to_numeric, errors="coerce")
        sub = sub.replace([np.inf, -np.inf], np.nan)
        sub["label"] = df["Label"].str.upper()
        sub["source_file"] = fname
        sub["split"] = split
        all_rows.append(sub)

    full = pd.concat(all_rows, ignore_index=True)
    full = full.dropna(axis=1, how="all")
    numeric_cols = [c for c in full.columns if c not in ("label", "source_file", "split")]
    full[numeric_cols] = full[numeric_cols].fillna(0.0)

    full.to_parquet(OUT_PARQUET, index=False)
    print(full.groupby(["label", "split"]).size())
    print(f"\nWrote {len(full)} DNP3 flow rows, {len(numeric_cols)} features -> {OUT_PARQUET}")


if __name__ == "__main__":
    main()
