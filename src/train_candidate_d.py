"""Candidate D (RelayGraph): builds a small relational graph per window (32
packet-embedding nodes; edges = temporal adjacency + same-flow + same
Modbus-function-code) and aggregates with a GAT, instead of the source
method's purely sequential Stacked LSTM. Flagged in MODEL_CANDIDATES.md as
the highest-implementation-risk, most-expensive candidate, explicitly a
smoke-test-only accuracy-ceiling probe -- prune at screening unless it
clears a meaningful bar over Candidates A-C.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix

from train_lstm_downstream import load_embeddings, ROOT, WINDOWS_PARQUET

PACKETS_PARQUET = ROOT / "data_audit" / "modbus_packets.parquet"
FIELD_RE = re.compile(r"(\w+)=([^,;]+)")
LAYER_RE = re.compile(r"(\w+_(?:Layer|PDU)):([^;]*)")


def parse_meta(text):
    d = {}
    for layer, body in LAYER_RE.findall(text):
        d[layer] = dict(FIELD_RE.findall(body))
    ip = d.get("IP_Layer", {})
    tcp = d.get("TCP_Layer", {})
    udp = d.get("UDP_Layer", {})
    mb = d.get("Modbus_Layer", {})
    flow = (ip.get("src"), ip.get("dst"), tcp.get("sport", udp.get("sport")), tcp.get("dport", udp.get("dport")))
    func = mb.get("funcCode")
    return flow, func


def build_window_graph(pkt_ids, emb, id2row, meta_lookup):
    N = len(pkt_ids)
    x = torch.tensor(np.stack([emb[id2row[i]] for i in pkt_ids]), dtype=torch.float32)
    edges = set()
    flows = [meta_lookup[i][0] for i in pkt_ids]
    funcs = [meta_lookup[i][1] for i in pkt_ids]
    for i in range(N):
        if i + 1 < N:
            edges.add((i, i + 1)); edges.add((i + 1, i))
        for j in range(i + 1, N):
            if flows[i] == flows[j] and flows[i] != (None, None, None, None):
                edges.add((i, j)); edges.add((j, i))
            if funcs[i] is not None and funcs[i] == funcs[j]:
                edges.add((i, j)); edges.add((j, i))
    if not edges:
        edges = {(i, i) for i in range(N)}
    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)


class RelayGraphNet(nn.Module):
    def __init__(self, d_in, d_model=64, n_classes=6, heads=4):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        self.gat1 = GATConv(d_model, d_model // heads, heads=heads)
        self.gat2 = GATConv(d_model, d_model // heads, heads=heads)
        self.cls_head = nn.Linear(d_model, n_classes)
        self.recon_head = nn.Linear(d_model, d_in)

    def forward(self, batch):
        h = torch.relu(self.in_proj(batch.x))
        h = torch.relu(self.gat1(h, batch.edge_index))
        h = torch.relu(self.gat2(h, batch.edge_index))
        from torch_geometric.nn import global_mean_pool
        pooled = global_mean_pool(h, batch.batch)
        return self.cls_head(pooled), self.recon_head(h)


def run(emb_tag, epochs=10, d_model=64, lr=1e-3, batch_size=32, device="cuda"):
    emb, id2row = load_embeddings(emb_tag)
    windows = pd.read_parquet(WINDOWS_PARQUET)
    packets = pd.read_parquet(PACKETS_PARQUET, columns=["pkt_id", "dissect_text"])
    meta_lookup = {row.pkt_id: parse_meta(row.dissect_text) for row in packets.itertuples()}

    le = LabelEncoder().fit(windows["class"])
    classes = list(le.classes_)
    benign_idx = classes.index("benign")

    def to_graphs(df):
        graphs = []
        for _, r in df.iterrows():
            ids = [int(x) for x in r["pkt_ids"].split(",")]
            g = build_window_graph(ids, emb, id2row, meta_lookup)
            g.y = torch.tensor([le.transform([r["class"]])[0]])
            graphs.append(g)
        return graphs

    train_g = to_graphs(windows[windows.split == "train"])
    val_g = to_graphs(windows[windows.split == "val"])
    test_g = to_graphs(windows[windows.split == "test"])

    def batches(graphs, bs, shuffle):
        idx = np.arange(len(graphs))
        if shuffle:
            np.random.shuffle(idx)
        for i in range(0, len(idx), bs):
            yield Batch.from_data_list([graphs[j] for j in idx[i:i + bs]])

    model = RelayGraphNet(emb.shape[1], d_model, len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    best_val_f1, best_state = -1, None
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in batches(train_g, batch_size, True):
            batch = batch.to(device)
            opt.zero_grad()
            logits, _ = model(batch)
            loss = crit(logits, batch.y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in batches(val_g, batch_size, False):
                batch = batch.to(device)
                logits, _ = model(batch)
                preds.extend(logits.argmax(-1).cpu().numpy().tolist())
                trues.extend(batch.y.cpu().numpy().tolist())
        val_f1 = f1_score(trues, preds, average="macro")
        print(f"[CandD] epoch {epoch}: loss={total_loss:.4f} val_macro_f1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1, best_state = val_f1, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in batches(test_g, batch_size, False):
            batch = batch.to(device)
            logits, _ = model(batch)
            preds.extend(logits.argmax(-1).cpu().numpy().tolist())
            trues.extend(batch.y.cpu().numpy().tolist())
    macro_f1 = f1_score(trues, preds, average="macro")
    prec, rec, f1c, support = precision_recall_fscore_support(trues, preds, labels=range(len(classes)), zero_division=0)
    per_class = {classes[i]: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1c[i]), "support": int(support[i])} for i in range(len(classes))}
    cm = confusion_matrix(trues, preds, labels=range(len(classes))).tolist()
    n_params = sum(p.numel() for p in model.parameters())
    return {"macro_f1": float(macro_f1), "balanced_accuracy": float(balanced_accuracy_score(trues, preds)),
            "per_class": per_class, "confusion_matrix": cm, "classes": classes,
            "best_val_macro_f1": float(best_val_f1), "n_params": n_params}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_tag", required=True)
    ap.add_argument("--out_tag", default="candidate_d")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out_dir = ROOT / "candidates" / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run(args.emb_tag, device=args.device)
    with open(out_dir / "classification_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "confusion_matrix"}, indent=2))
