"""Targeted patch: append additional benign ('clean') packets/windows to the
existing dataset so benign reaches a comparable total to the attack classes
(fixing the anomaly-calibration imbalance found in REPRODUCTION_REPORT.md),
without re-scanning the (already-correct, already onset-fixed) attack
classes. Safe to run while BART trains in the background -- it only touches
the parquet files on disk; any already-running training job has its own
in-memory dataset snapshot.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import csv
import pandas as pd
from dissect import dissect_to_text, raw_bytes
from scapy.utils import PcapReader

ROOT = Path(__file__).resolve().parent.parent
FRAZAO_ROOT = ROOT / "external_datasets" / "Frazao2019_ModbusTCP_ICS_PCAPS" / "extracted"
OUT_DIR = ROOT / "data_audit"

WINDOW_SIZE = 32
MAX_BYTES = 300
TARGET_TOTAL = 48000  # match attack-class totals
SPLIT_FRACS = (0.70, 0.15, 0.15)

BENIGN_FILES = sorted((FRAZAO_ROOT / "captures1_v2" / "clean").glob("*.pcap"))


def read_file_packets(path, skip, cap):
    out = []
    with PcapReader(str(path)) as reader:
        for i, pkt in enumerate(reader):
            if i < skip:
                continue
            if i >= skip + cap:
                break
            try:
                b = raw_bytes(pkt)[:MAX_BYTES]
                t = dissect_to_text(pkt)
                ts = float(pkt.time)
            except Exception:
                continue
            out.append((b, t, ts))
    return out


def main():
    packets = pd.read_parquet(OUT_DIR / "modbus_packets.parquet")
    windows = pd.read_parquet(OUT_DIR / "modbus_windows.parquet")
    manifest = pd.read_csv(ROOT / "DATA_SPLIT_MANIFEST.csv")

    existing_benign = packets[packets["class"] == "benign"]
    have = len(existing_benign)
    need = max(0, TARGET_TOTAL - have)
    print(f"Existing benign packets: {have}, need {need} more (target {TARGET_TOTAL})")
    if need == 0:
        print("Nothing to do.")
        return

    already_read_per_file = existing_benign.groupby("source_file")["local_idx"].max() + 1

    pkt_id_counter = int(packets["pkt_id"].max()) + 1
    window_id_counter = int(windows["window_id"].max()) + 1

    new_packet_rows, new_window_rows, new_manifest_rows = [], [], []
    per_file_need = need // len(BENIGN_FILES) + 1

    for fpath in BENIGN_FILES:
        rel = str(fpath.relative_to(FRAZAO_ROOT))
        skip = int(already_read_per_file.get(rel, 0))
        pkts = read_file_packets(fpath, skip=skip, cap=per_file_need)
        if not pkts:
            continue
        n = len(pkts)
        n_train = int(n * SPLIT_FRACS[0])
        n_val = int(n * SPLIT_FRACS[1])
        blocks = [("train", 0, n_train), ("val", n_train, n_train + n_val), ("test", n_train + n_val, n)]
        file_pkt_ids = [None] * n
        for (b, t, ts), local_idx in zip(pkts, range(n)):
            pid = pkt_id_counter
            pkt_id_counter += 1
            file_pkt_ids[local_idx] = pid
            new_packet_rows.append({
                "pkt_id": pid, "class": "benign", "source_file": rel,
                "local_idx": skip + local_idx, "raw_bytes_hex": b.hex(), "n_bytes": len(b),
                "dissect_text": t, "timestamp": ts, "onset_skip": 0,
            })
        for split_name, start, end in blocks:
            n_windows = (end - start) // WINDOW_SIZE
            for w in range(n_windows):
                ws, we = start + w * WINDOW_SIZE, start + (w + 1) * WINDOW_SIZE
                ids = file_pkt_ids[ws:we]
                wid = window_id_counter
                window_id_counter += 1
                new_window_rows.append({"window_id": wid, "class": "benign", "split": split_name,
                                         "source_file": rel, "pkt_ids": ",".join(map(str, ids))})
                new_manifest_rows.append({"window_id": wid, "class": "benign", "split": split_name,
                                           "source_file": rel, "start_local_idx": skip + ws, "end_local_idx": skip + we})
        print(f"  {fpath.name}: +{n} packets (skip={skip})")

    packets = pd.concat([packets, pd.DataFrame(new_packet_rows)], ignore_index=True)
    windows = pd.concat([windows, pd.DataFrame(new_window_rows)], ignore_index=True)
    packets.to_parquet(OUT_DIR / "modbus_packets.parquet", index=False)
    windows.to_parquet(OUT_DIR / "modbus_windows.parquet", index=False)

    manifest = pd.concat([manifest, pd.DataFrame(new_manifest_rows)], ignore_index=True)
    manifest.to_csv(ROOT / "DATA_SPLIT_MANIFEST.csv", index=False)

    print(f"\nNew benign packet total: {len(packets[packets['class']=='benign'])}")
    print(windows.groupby(["class", "split"]).size())


if __name__ == "__main__":
    main()
