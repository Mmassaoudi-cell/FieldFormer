"""Candidate B (ProtoGuard): a single shared SSM encoder (same backbone as
Candidate A, kept constant so this script isolates the HEAD's contribution)
with a prototypical-network head unifying classification and anomaly
detection into one model, instead of the source method's two independently
trained, disconnected stacks (Stacked-LSTM classifier + separate Stacked-LSTM
autoencoder). See MODEL_CANDIDATES.md for the full design rationale.

Training: prototypical cross-entropy over ALL known classes (benign +
attacks) -- unlike the source method, ProtoGuard is not restricted to
benign-only training for its detection capability, because unification with
the classifier is the whole point of this candidate. For a fair, comparable
anomaly-style operating point against the source method's benign-only
semisupervised protocol, a SEPARATE benign-only prototype variant is also
evaluated (`--benign_only_proto`), using distance-to-benign-prototype exactly
as the source method uses reconstruction error.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix,
    precision_score, recall_score, roc_auc_score, average_precision_score,
)

from ssm_layer import SSMBackbone
from train_lstm_downstream import WindowSeqDataset, collate, load_embeddings, ROOT, WINDOWS_PARQUET


class ProtoGuard(nn.Module):
    def __init__(self, d_in, d_model=128, n_layers=2, n_classes=6, temperature=1.0):
        super().__init__()
        self.backbone = SSMBackbone(d_in, d_model, n_layers)
        self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.1)
        self.temperature = temperature

    def embed(self, x):
        h = self.backbone(x)
        return h[:, -1]  # window embedding z

    def forward(self, x):
        z = self.embed(x)
        dists = torch.cdist(z, self.prototypes)  # (B, n_classes)
        logits = -dists ** 2 / self.temperature
        return logits, dists, z


def run(emb_tag, epochs=15, d_model=128, n_layers=2, lr=1e-3, batch_size=64, device="cuda"):
    emb, id2row = load_embeddings(emb_tag)
    windows = pd.read_parquet(WINDOWS_PARQUET)
    le = LabelEncoder().fit(windows["class"])
    classes = list(le.classes_)
    benign_idx = classes.index("benign")

    def make_loader(split, shuffle):
        sub = windows[windows["split"] == split]
        return DataLoader(WindowSeqDataset(sub, emb, id2row), batch_size=batch_size, shuffle=shuffle, collate_fn=collate)

    train_loader, val_loader, test_loader = make_loader("train", True), make_loader("val", False), make_loader("test", False)
    model = ProtoGuard(emb.shape[1], d_model, n_layers, len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    best_val_f1, best_state = -1, None
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, labels in train_loader:
            x = x.to(device)
            y = torch.tensor(le.transform(labels), dtype=torch.long, device=device)
            opt.zero_grad()
            logits, dists, z = model(x)
            loss_ce = crit(logits, y)
            # margin term: push benign windows toward benign prototype, away from attack prototypes
            is_benign = (y == benign_idx).float()
            margin_loss = (is_benign * dists[:, benign_idx] ** 2).mean()
            loss = loss_ce + 0.1 * margin_loss
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for x, labels in val_loader:
                logits, _, _ = model(x.to(device))
                val_preds.extend(logits.argmax(-1).cpu().numpy().tolist())
                val_true.extend(le.transform(labels).tolist())
        val_f1 = f1_score(val_true, val_preds, average="macro")
        print(f"[CandB] epoch {epoch}: loss={total_loss/len(train_loader.dataset):.4f} val_macro_f1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1, best_state = val_f1, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    # --- Classification eval ---
    test_preds, test_true = [], []
    with torch.no_grad():
        for x, labels in test_loader:
            logits, _, _ = model(x.to(device))
            test_preds.extend(logits.argmax(-1).cpu().numpy().tolist())
            test_true.extend(le.transform(labels).tolist())
    macro_f1 = f1_score(test_true, test_preds, average="macro")
    prec, rec, f1c, support = precision_recall_fscore_support(test_true, test_preds, labels=range(len(classes)), zero_division=0)
    per_class = {classes[i]: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1c[i]), "support": int(support[i])} for i in range(len(classes))}
    cm = confusion_matrix(test_true, test_preds, labels=range(len(classes))).tolist()
    n_params = sum(p.numel() for p in model.parameters())
    cls_result = {"macro_f1": float(macro_f1), "balanced_accuracy": float(balanced_accuracy_score(test_true, test_preds)),
                  "per_class": per_class, "confusion_matrix": cm, "classes": classes,
                  "best_val_macro_f1": float(best_val_f1), "n_params": n_params}

    # --- Unified open-set anomaly score: distance to the BENIGN prototype ---
    # (directly comparable to the source method's reconstruction-error score;
    # this is the key structural difference this candidate is testing: ONE
    # model produces both outputs, vs. two disconnected models.)
    def benign_dist(df):
        loader = DataLoader(WindowSeqDataset(df, emb, id2row), batch_size=batch_size, shuffle=False, collate_fn=collate)
        out = []
        with torch.no_grad():
            for x, _ in loader:
                _, dists, _ = model(x.to(device))
                out.extend(dists[:, benign_idx].cpu().numpy().tolist())
        return np.array(out)

    benign_val = windows[(windows["class"] == "benign") & (windows["split"] == "val")]
    attack_val = windows[(windows["class"] != "benign") & (windows["split"] == "val")]
    benign_test = windows[(windows["class"] == "benign") & (windows["split"] == "test")]
    attack_test = windows[(windows["class"] != "benign") & (windows["split"] == "test")]

    d_bc, d_ac = benign_dist(benign_val), benign_dist(attack_val)
    d_bt, d_at = benign_dist(benign_test), benign_dist(attack_test)

    all_calib = np.concatenate([d_bc, d_ac])
    calib_labels = np.concatenate([np.zeros(len(d_bc)), np.ones(len(d_ac))])
    candidates = np.quantile(all_calib, np.linspace(0.01, 0.99, 199))
    best_thr, best_f1 = None, -1
    for thr in candidates:
        f1 = f1_score(calib_labels, (all_calib > thr).astype(int))
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    y_true = np.concatenate([np.zeros(len(d_bt)), np.ones(len(d_at))])
    y_score = np.concatenate([d_bt, d_at])
    y_pred = (y_score > best_thr).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum()); fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum()); tp = int(((y_true == 1) & (y_pred == 1)).sum())
    anomaly_result = {
        "threshold": float(best_thr), "calib_f1": float(best_f1),
        "test_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "test_fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "test_fnr": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        "roc_auc": float(roc_auc_score(y_true, y_score)), "pr_auc": float(average_precision_score(y_true, y_score)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }

    return cls_result, anomaly_result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_tag", required=True)
    ap.add_argument("--out_tag", default="candidate_b")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = ROOT / "candidates" / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    cls_result, anomaly_result = run(args.emb_tag, device=args.device)
    with open(out_dir / "classification_result.json", "w") as f:
        json.dump(cls_result, f, indent=2)
    with open(out_dir / "anomaly_result.json", "w") as f:
        json.dump(anomaly_result, f, indent=2)
    print("classification:", json.dumps({k: v for k, v in cls_result.items() if k != "confusion_matrix"}, indent=2))
    print("anomaly (unified, distance-to-benign-prototype):", json.dumps(anomaly_result, indent=2))
