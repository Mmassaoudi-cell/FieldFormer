"""Detect the packet-index onset of a flooding attack within a capture file by
looking for a sustained jump in local arrival rate, rather than trusting the
filename's encoded delay (empirically found unreliable -- see
REPRODUCTION_REPORT.md data-quality note). Used by build_modbus_dataset.py.
"""
from scapy.utils import PcapReader
import numpy as np

SCAN_LIMIT = 120_000
BASELINE_N = 1500       # packets assumed pre-attack, used to estimate baseline rate
WINDOW = 300             # rolling window (packets) used to detect the rate jump
RATE_JUMP_FACTOR = 3.0   # onset = first window where local rate >= factor * baseline rate


def detect_onset(path, scan_limit=SCAN_LIMIT):
    times = []
    with PcapReader(str(path)) as reader:
        for i, pkt in enumerate(reader):
            if i >= scan_limit:
                break
            times.append(float(pkt.time))
    times = np.array(times)
    n = len(times)
    if n < BASELINE_N + WINDOW:
        return 0, n  # too short to detect reliably; read from start

    baseline_dt = np.median(np.diff(times[:BASELINE_N]))
    if baseline_dt <= 0:
        baseline_dt = 1e-3
    baseline_rate = 1.0 / baseline_dt

    onset = None
    for start in range(BASELINE_N, n - WINDOW, WINDOW // 2):
        seg = times[start:start + WINDOW]
        dt = np.median(np.diff(seg))
        if dt <= 0:
            local_rate = float("inf")
        else:
            local_rate = 1.0 / dt
        if local_rate >= RATE_JUMP_FACTOR * baseline_rate:
            onset = start
            break
    if onset is None:
        onset = 0  # no clear jump found within scan window; fall back to start
    return onset, n


if __name__ == "__main__":
    import sys
    p = sys.argv[1]
    onset, n = detect_onset(p)
    print(f"{p}: onset_idx={onset} (scanned {n} packets)")
