"""Build the reproduction dataset from the REAL Frazao et al. 2019 Modbus/TCP
process-automation PCAPs (source-paper primary dataset).

Leakage control (see DATA_AUDIT.md / Section 13 of the research protocol):
  - Splitting is done WITHIN each capture file by contiguous, non-overlapping
    time blocks (70% train / 15% val / 15% test, in packet order), never by
    randomly shuffling rows or windows. Sliding windows never straddle a
    split boundary.
  - A compute budget is applied (documented, not hidden): each class is
    capped at CLASS_PACKET_CAP packets, drawn proportionally across that
    class's available capture files (varying attack duration/interval),
    to keep this reproducible within realistic session compute/time.

Outputs:
  data_audit/modbus_packets.parquet        (one row per packet: bytes+text+meta)
  data_audit/modbus_windows.parquet        (one row per window: packet-id list + label)
  DATA_SPLIT_MANIFEST.csv                  (window-level split assignment ledger)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import csv
import random
from collections import defaultdict

import pandas as pd
from scapy.utils import PcapReader

from dissect import dissect_to_text, raw_bytes
from find_onset import detect_onset

# Classes whose attack signature is a rate phenomenon (flooding): the naive
# "read from the start of the file" approach was found to capture almost
# entirely pre-attack traffic, because the filename's encoded delay (e.g.
# "15m") does not reliably correspond to the packet-index onset -- see
# REPRODUCTION_REPORT.md. detect_onset() finds the real onset via a local
# packet-rate jump and we read from there instead.
RATE_ONSET_CLASSES = {"modbus_query_flood", "modbus_query_flood2", "icmp_flood", "tcp_syn_flood"}

ROOT = Path(__file__).resolve().parent.parent
FRAZAO_ROOT = ROOT / "external_datasets" / "Frazao2019_ModbusTCP_ICS_PCAPS" / "extracted"
OUT_DIR = ROOT / "data_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 32          # matches source paper's N=32
MAX_BYTES = 300           # covers >95% of observed packet lengths in this dataset family
PER_FILE_CAP = 6000       # packets read per capture file (bounds per-file scan time)
CLASS_PACKET_CAP = 45000  # packets per class after aggregation across files (compute budget)
SPLIT_FRACS = (0.70, 0.15, 0.15)  # train/val/test, contiguous per-file blocks

CLASS_DIRS = {
    "benign": ["captures1_v2/clean"],
    "mitm": ["captures1_v2/mitm"],
    "modbus_query_flood": ["captures1_v2/modbusQueryFlooding", "captures2/modbusQueryFlooding", "captures3/modbusQueryFlooding"],
    "modbus_query_flood2": ["captures1_v2/modbusQuery2Flooding"],
    "icmp_flood": ["captures1_v2/pingFloodDDoS", "captures2/pingFloodDDoS", "captures3/pingFloodDDoS"],
    "tcp_syn_flood": ["captures1_v2/tcpSYNFloodDDoS", "captures2/tcpSYNFloodDDoS", "captures3/tcpSYNFloodDDoS"],
}

random.seed(1337)


def list_files(class_name):
    files = []
    for rel in CLASS_DIRS[class_name]:
        d = FRAZAO_ROOT / rel
        if d.exists():
            files.extend(sorted(d.glob("*.pcap")))
    return files


def read_file_packets(path: Path, cap: int, skip: int = 0):
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
    packet_rows = []
    window_rows = []
    manifest_rows = []
    pkt_id_counter = 0
    window_id_counter = 0

    for cls, _ in CLASS_DIRS.items():
        files = list_files(cls)
        if not files:
            print(f"[WARN] no files found for class {cls}")
            continue
        print(f"Class {cls}: {len(files)} candidate files")
        class_total = 0
        # distribute the class budget across files roughly evenly, but let
        # smaller files just contribute what they have
        per_file_target = max(1, CLASS_PACKET_CAP // max(1, len(files)))
        # benign ("clean") only has 3 capture files -- the global PER_FILE_CAP
        # (sized for classes with 11-37 files) would otherwise starve benign
        # to ~18k packets vs ~45-48k for every attack class, which was found
        # to badly skew the anomaly-detector calibration threshold toward
        # flagging almost everything (see REPRODUCTION_REPORT.md). Raise the
        # per-file cap for this class so it reaches a comparable total; the
        # clean captures have 35k-428k packets available, so this is safe.
        file_cap_override = 16000 if cls == "benign" else PER_FILE_CAP
        for fpath in files:
            if class_total >= CLASS_PACKET_CAP:
                break
            cap_this_file = min(file_cap_override, per_file_target * 2)
            skip = 0
            if cls in RATE_ONSET_CLASSES:
                try:
                    onset_idx, scanned_n = detect_onset(fpath)
                    skip = onset_idx
                    print(f"    onset-detect {fpath.name}: onset_idx={onset_idx} (scanned {scanned_n})")
                except Exception as e:
                    print(f"    onset-detect FAILED for {fpath.name}: {e}; reading from start")
            pkts = read_file_packets(fpath, cap_this_file, skip=skip)
            if not pkts:
                continue
            n = len(pkts)
            n_train = int(n * SPLIT_FRACS[0])
            n_val = int(n * SPLIT_FRACS[1])
            blocks = [
                ("train", 0, n_train),
                ("val", n_train, n_train + n_val),
                ("test", n_train + n_val, n),
            ]
            file_pkt_ids = [None] * n
            for (b, t, ts), local_idx in zip(pkts, range(n)):
                pid = pkt_id_counter
                pkt_id_counter += 1
                file_pkt_ids[local_idx] = pid
                packet_rows.append({
                    "pkt_id": pid,
                    "class": cls,
                    "source_file": str(fpath.relative_to(FRAZAO_ROOT)),
                    "local_idx": local_idx,
                    "raw_bytes_hex": b.hex(),
                    "n_bytes": len(b),
                    "dissect_text": t,
                    "timestamp": ts,
                    "onset_skip": skip,
                })

            for split_name, start, end in blocks:
                block_len = end - start
                n_windows = block_len // WINDOW_SIZE
                for w in range(n_windows):
                    ws, we = start + w * WINDOW_SIZE, start + (w + 1) * WINDOW_SIZE
                    ids = file_pkt_ids[ws:we]
                    wid = window_id_counter
                    window_id_counter += 1
                    window_rows.append({
                        "window_id": wid,
                        "class": cls,
                        "split": split_name,
                        "source_file": str(fpath.relative_to(FRAZAO_ROOT)),
                        "pkt_ids": ",".join(map(str, ids)),
                    })
                    manifest_rows.append({
                        "window_id": wid,
                        "class": cls,
                        "split": split_name,
                        "source_file": str(fpath.relative_to(FRAZAO_ROOT)),
                        "start_local_idx": ws,
                        "end_local_idx": we,
                    })
            class_total += n
            print(f"  {fpath.name}: {n} pkts (train/val/test blocks) -> class_total={class_total}")

    packets_df = pd.DataFrame(packet_rows)
    windows_df = pd.DataFrame(window_rows)
    packets_df.to_parquet(OUT_DIR / "modbus_packets.parquet", index=False)
    windows_df.to_parquet(OUT_DIR / "modbus_windows.parquet", index=False)

    with open(ROOT / "DATA_SPLIT_MANIFEST.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["window_id", "class", "split", "source_file", "start_local_idx", "end_local_idx"])
        w.writeheader()
        for r in manifest_rows:
            w.writerow(r)

    print("\n=== Summary ===")
    print(packets_df.groupby("class").size())
    print("\nWindows by class/split:")
    print(windows_df.groupby(["class", "split"]).size())
    print(f"\nWrote {len(packets_df)} packets, {len(windows_df)} windows.")
    print(f"-> {OUT_DIR/'modbus_packets.parquet'}")
    print(f"-> {OUT_DIR/'modbus_windows.parquet'}")
    print(f"-> {ROOT/'DATA_SPLIT_MANIFEST.csv'}")


if __name__ == "__main__":
    main()
