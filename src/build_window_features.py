"""Hand-crafted, flow/statistical feature engineering per window -- the feature
source for the classical and boosting benchmark family (Modbus-ML-style: known
header/function-code fields, no learned representation, no raw bytes). Runs on
CPU so it can proceed in parallel with GPU-side BART training.

Output: data_audit/window_features.parquet (one row per window, label + numeric
features), ready for scripts/train_classical_benchmarks.py.
"""
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PACKETS_PARQUET = ROOT / "data_audit" / "modbus_packets.parquet"
WINDOWS_PARQUET = ROOT / "data_audit" / "modbus_windows.parquet"
OUT_PARQUET = ROOT / "data_audit" / "window_features.parquet"

FIELD_RE = re.compile(r"(\w+)=([^,;]+)")
LAYER_RE = re.compile(r"(\w+_(?:Layer|PDU)):([^;]*)")

TCP_FLAG_CHARS = "FSRPAUEC"  # FIN SYN RST PSH ACK URG ECE CWR


def parse_dissect(text: str) -> dict:
    d = {}
    for layer, body in LAYER_RE.findall(text):
        fields = dict(FIELD_RE.findall(body))
        d[layer] = fields
    return d


def featurize_window(rows: pd.DataFrame) -> dict:
    lens = rows["n_bytes"].to_numpy()
    parsed = [parse_dissect(t) for t in rows["dissect_text"]]
    ts = rows["timestamp"].to_numpy() if "timestamp" in rows.columns else None

    layer_presence = Counter()
    src_ips, dst_ips, src_macs, dst_macs = set(), set(), set(), set()
    flows = set()
    tcp_flag_counts = Counter()
    modbus_funcs = Counter()
    icmp_types = Counter()
    arp_ops = Counter()
    n_tcp = n_udp = n_icmp = n_arp = n_modbus = 0

    for d in parsed:
        for layer in d:
            layer_presence[layer] += 1
        eth = d.get("ETH_Layer")
        if eth:
            if "src" in eth: src_macs.add(eth["src"])
            if "dst" in eth: dst_macs.add(eth["dst"])
        ip = d.get("IP_Layer")
        if ip:
            if "src" in ip: src_ips.add(ip["src"])
            if "dst" in ip: dst_ips.add(ip["dst"])
        tcp = d.get("TCP_Layer")
        if tcp:
            n_tcp += 1
            flags = tcp.get("flags", "")
            for ch in flags:
                if ch in TCP_FLAG_CHARS:
                    tcp_flag_counts[ch] += 1
            if ip:
                flows.add((ip.get("src"), ip.get("dst"), tcp.get("sport"), tcp.get("dport")))
        udp = d.get("UDP_Layer")
        if udp:
            n_udp += 1
            if ip:
                flows.add((ip.get("src"), ip.get("dst"), udp.get("sport"), udp.get("dport")))
        icmp = d.get("ICMP_Layer")
        if icmp:
            n_icmp += 1
            icmp_types[icmp.get("type", "?")] += 1
        arp = d.get("ARP_Layer")
        if arp:
            n_arp += 1
            arp_ops[arp.get("op", "?")] += 1
        mb = d.get("Modbus_Layer")
        if mb:
            n_modbus += 1
            modbus_funcs[mb.get("funcCode", "?")] += 1

    n = len(rows)
    feat = {
        "len_mean": float(np.mean(lens)), "len_std": float(np.std(lens)),
        "len_min": float(np.min(lens)), "len_max": float(np.max(lens)),
        "frac_tcp": n_tcp / n, "frac_udp": n_udp / n, "frac_icmp": n_icmp / n,
        "frac_arp": n_arp / n, "frac_modbus": n_modbus / n,
        "n_unique_src_ip": len(src_ips), "n_unique_dst_ip": len(dst_ips),
        "n_unique_src_mac": len(src_macs), "n_unique_dst_mac": len(dst_macs),
        "n_unique_flows": len(flows),
        "tcp_syn_frac": tcp_flag_counts.get("S", 0) / max(1, n_tcp),
        "tcp_ack_frac": tcp_flag_counts.get("A", 0) / max(1, n_tcp),
        "tcp_fin_frac": tcp_flag_counts.get("F", 0) / max(1, n_tcp),
        "tcp_rst_frac": tcp_flag_counts.get("R", 0) / max(1, n_tcp),
        "tcp_psh_frac": tcp_flag_counts.get("P", 0) / max(1, n_tcp),
        "n_unique_modbus_funcs": len(modbus_funcs),
        "n_unique_icmp_types": len(icmp_types),
        "n_unique_arp_ops": len(arp_ops),
        "mac_ip_pairs": len(src_macs.union(dst_macs)),  # proxy for ARP-spoofing style MAC churn
    }
    # Timing / rate features -- the defining signal for flooding attacks,
    # which are a rate phenomenon rather than a structural/content one (see
    # SOURCE_WEAKNESS_ANALYSIS.md and the data-quality note in
    # REPRODUCTION_REPORT.md: window composition features alone cannot
    # separate e.g. Modbus query flooding from normal polling).
    if ts is not None and len(ts) > 1:
        diffs = np.diff(np.sort(ts))
        diffs = diffs[diffs >= 0]
        if len(diffs) > 0:
            feat["mean_iat"] = float(np.mean(diffs))
            feat["std_iat"] = float(np.std(diffs))
            feat["median_iat"] = float(np.median(diffs))
            feat["min_iat"] = float(np.min(diffs))
            span = ts.max() - ts.min()
            feat["pkt_rate_hz"] = float(n / span) if span > 0 else float(n)
        else:
            feat["mean_iat"] = feat["std_iat"] = feat["median_iat"] = feat["min_iat"] = 0.0
            feat["pkt_rate_hz"] = 0.0
    else:
        feat["mean_iat"] = feat["std_iat"] = feat["median_iat"] = feat["min_iat"] = 0.0
        feat["pkt_rate_hz"] = 0.0

    # Shannon entropy of Modbus function-code distribution within window
    if modbus_funcs:
        total = sum(modbus_funcs.values())
        probs = np.array([c / total for c in modbus_funcs.values()])
        feat["modbus_func_entropy"] = float(-(probs * np.log2(probs)).sum())
    else:
        feat["modbus_func_entropy"] = 0.0
    return feat


def main():
    packets = pd.read_parquet(PACKETS_PARQUET).set_index("pkt_id")
    windows = pd.read_parquet(WINDOWS_PARQUET)

    rows = []
    for i, w in windows.iterrows():
        ids = [int(x) for x in w["pkt_ids"].split(",")]
        sub = packets.loc[ids]
        feat = featurize_window(sub)
        feat.update({"window_id": w["window_id"], "class": w["class"], "split": w["split"]})
        rows.append(feat)
        if i % 1000 == 0:
            print(f"  processed {i}/{len(windows)} windows")

    out = pd.DataFrame(rows)
    out.to_parquet(OUT_PARQUET, index=False)
    print(out.groupby(["class", "split"]).size())
    print(f"Wrote {len(out)} window feature rows -> {OUT_PARQUET}")


if __name__ == "__main__":
    main()
