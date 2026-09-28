"""Builds a genuinely independent, out-of-distribution TEST-ONLY dataset from
Edge-IIoTset (Ferrag et al. 2022) -- a different testbed, different devices,
different MAC/IP address space, captured independently of the Frazao et al.
Modbus/TCP dataset used for all training/selection/tuning in this study.
Used exclusively for zero-shot cross-dataset generalization evaluation of the
already-frozen final model; nothing from this dataset is used for training,
calibration, or model selection.

Classes matched to the Frazao taxonomy where the same attack type exists in
Edge-IIoTset: benign (Modbus), mitm (ARP spoofing), icmp_flood, tcp_syn_flood.
Flooding onset is detected the same way as for Frazao (see find_onset.py /
REPRODUCTION_REPORT.md) rather than assumed from capture order.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from scapy.utils import PcapReader

from dissect import dissect_to_text, raw_bytes
from find_onset import detect_onset

ROOT = Path(__file__).resolve().parent.parent
EDGE_ROOT = Path(r"C:\Users\MMASSAOUDI\Desktop\Data\Edge-IIoTset\Edge-IIoTset dataset")
OUT_DIR = ROOT / "data_audit"

WINDOW_SIZE = 32
MAX_BYTES = 300
PER_CLASS_CAP = 6000  # bounded, test-only budget -- this dataset is never trained on

TARGETS = {
    "benign": (EDGE_ROOT / "Normal traffic" / "Modbus" / "Modbus.pcap", False),
    "mitm": (EDGE_ROOT / "Attack traffic" / "MITM (ARP spoofing + DNS) Attack.pcap", False),
    "icmp_flood": (EDGE_ROOT / "Attack traffic" / "DDoS ICMP Flood Attacks.pcap", True),
    "tcp_syn_flood": (EDGE_ROOT / "Attack traffic" / "DDoS TCP SYN Flood Attacks.pcap", True),
}


def read_packets(path, skip, cap):
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
            except Exception:
                continue
            out.append((b, t))
    return out


def main():
    packet_rows = []
    window_rows = []
    pkt_id = 0
    window_id = 0

    for cls, (path, needs_onset) in TARGETS.items():
        if not path.exists():
            print(f"[WARN] missing {path}")
            continue
        skip = 0
        if needs_onset:
            onset_idx, scanned = detect_onset(path, scan_limit=120_000)
            skip = onset_idx
            print(f"{cls}: onset_idx={onset_idx} (scanned {scanned})")
        pkts = read_packets(path, skip, PER_CLASS_CAP)
        n = len(pkts)
        print(f"{cls}: {n} packets read (skip={skip})")
        pkt_ids_this_class = []
        for b, t in pkts:
            packet_rows.append({"pkt_id": pkt_id, "class": cls, "raw_bytes_hex": b.hex(),
                                 "n_bytes": len(b), "dissect_text": t})
            pkt_ids_this_class.append(pkt_id)
            pkt_id += 1
        n_windows = n // WINDOW_SIZE
        for w in range(n_windows):
            ids = pkt_ids_this_class[w * WINDOW_SIZE:(w + 1) * WINDOW_SIZE]
            window_rows.append({"window_id": window_id, "class": cls, "pkt_ids": ",".join(map(str, ids))})
            window_id += 1

    packets_df = pd.DataFrame(packet_rows)
    windows_df = pd.DataFrame(window_rows)
    packets_df.to_parquet(OUT_DIR / "edgeiiotset_packets.parquet", index=False)
    windows_df.to_parquet(OUT_DIR / "edgeiiotset_windows.parquet", index=False)
    print(windows_df.groupby("class").size())
    print(f"\nWrote {len(packets_df)} packets, {len(windows_df)} windows (test-only, never trained on).")


if __name__ == "__main__":
    main()
