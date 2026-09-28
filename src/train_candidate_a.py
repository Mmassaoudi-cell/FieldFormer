"""Candidate A (FieldFormer-SSM) end-to-end downstream training: consumes
FieldFormer packet embeddings (train_fieldformer_encoder.py) with the
SSMBackbone (ssm_layer.py) in place of the source method's Stacked LSTM, for
both the classification and semisupervised anomaly-detection tasks. Mirrors
train_lstm_downstream.py's protocol exactly (same splits, same calibration
procedure, same metrics) so results are directly comparable.
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


class SSMClassifier(nn.Module):
    def __init__(self, d_in, d_model=128, n_layers=2, n_classes=6):
        super().__init__()
        self.backbone = SSMBackbone(d_in, d_model, n_layers)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        h = self.backbone(x)
        pooled = h[:, -1]  # final timestep, analogous to LSTM's final hidden state
        return self.fc(pooled)


class SSMAutoencoder(nn.Module):
    def __init__(self, d_in, d_model=128, n_layers=2):
        super().__init__()
        self.backbone = SSMBackbone(d_in, d_model, n_layers)
        self.decoder_backbone = SSMBackbone(d_model, d_model, n_layers)
        self.out_proj = nn.Linear(d_model, d_in)

    def forward(self, x):
        h = self.backbone(x)
        latent = h[:, -1:].expand(-1, x.size(1), -1)
        dec = self.decoder_backbone(latent)
        return self.out_proj(dec)


def train_classifier(emb_tag, epochs=15, d_model=128, n_layers=2, lr=1e-3, batch_size=64, device="cuda"):
    emb, id2row = load_embeddings(emb_tag)
    windows = pd.read_parquet(WINDOWS_PARQUET)
    le = LabelEncoder().fit(windows["class"])
    classes = list(le.classes_)

    def make_loader(split, shuffle):
        sub = windows[windows["split"] == split]
        return DataLoader(WindowSeqDataset(sub, emb, id2row), batch_size=batch_size, shuffle=shuffle, collate_fn=collate)

    train_loader, val_loader, test_loader = make_loader("train", True), make_loader("val", False), make_loader("test", False)
    model = SSMClassifier(emb.shape[1], d_model, n_layers, len(classes)).to(device)
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
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for x, labels in val_loader:
                logits = model(x.to(device))
                val_preds.extend(logits.argmax(-1).cpu().numpy().tolist())
                val_true.extend(le.transform(labels).tolist())
        val_f1 = f1_score(val_true, val_preds, average="macro")
        print(f"[CandA-cls] epoch {epoch}: loss={total_loss/len(train_loader.dataset):.4f} val_macro_f1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1, best_state = val_f1, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for x, labels in test_loader:
            logits = model(x.to(device))
            test_preds.extend(logits.argmax(-1).cpu().numpy().tolist())
            test_true.extend(le.transform(labels).tolist())

    macro_f1 = f1_score(test_true, test_preds, average="macro")
    weighted_f1 = f1_score(test_true, test_preds, average="weighted")
    bal_acc = balanced_accuracy_score(test_true, test_preds)
    prec, rec, f1c, support = precision_recall_fscore_support(test_true, test_preds, labels=range(len(classes)), zero_division=0)
    per_class = {classes[i]: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1c[i]), "support": int(support[i])} for i in range(len(classes))}
    cm = confusion_matrix(test_true, test_preds, labels=range(len(classes))).tolist()
    n_params = sum(p.numel() for p in model.parameters())
    return {"macro_f1": float(macro_f1), "weighted_f1": float(weighted_f1), "balanced_accuracy": float(bal_acc),
            "per_class": per_class, "confusion_matrix": cm, "classes": classes,
            "best_val_macro_f1": float(best_val_f1), "n_params": n_params}


def train_autoencoder(emb_tag, epochs=20, d_model=128, n_layers=2, lr=1e-3, batch_size=64, device="cuda"):
    emb, id2row = load_embeddings(emb_tag)
    windows = pd.read_parquet(WINDOWS_PARQUET)

    benign_train = windows[(windows["class"] == "benign") & (windows["split"] == "train")]
    benign_calib = windows[(windows["class"] == "benign") & (windows["split"] == "val")]
    benign_test = windows[(windows["class"] == "benign") & (windows["split"] == "test")]
    attack_calib = windows[(windows["class"] != "benign") & (windows["split"] == "val")]
    attack_test = windows[(windows["class"] != "benign") & (windows["split"] == "test")]

    train_loader = DataLoader(WindowSeqDataset(benign_train, emb, id2row), batch_size=batch_size, shuffle=True, collate_fn=collate)
    model = SSMAutoencoder(emb.shape[1], d_model, n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            opt.zero_grad()
            recon = model(x)
            loss = crit(recon, x)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
        print(f"[CandA-AE] epoch {epoch}: recon_mse={total_loss/len(train_loader.dataset):.6f}")

    def recon_errors(df):
        loader = DataLoader(WindowSeqDataset(df, emb, id2row), batch_size=batch_size, shuffle=False, collate_fn=collate)
        errs = []
        model.eval()
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                recon = model(x)
                errs.extend(((x - recon) ** 2).mean(dim=(1, 2)).cpu().numpy().tolist())
        return np.array(errs)

    err_bc, err_ac = recon_errors(benign_calib), recon_errors(attack_calib)
    err_bt, err_at = recon_errors(benign_test), recon_errors(attack_test)

    all_calib = np.concatenate([err_bc, err_ac])
    calib_labels = np.concatenate([np.zeros(len(err_bc)), np.ones(len(err_ac))])
    candidates = np.quantile(all_calib, np.linspace(0.01, 0.99, 199))
    best_thr, best_f1 = None, -1
    for thr in candidates:
        f1 = f1_score(calib_labels, (all_calib > thr).astype(int))
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    y_true = np.concatenate([np.zeros(len(err_bt)), np.ones(len(err_at))])
    y_score = np.concatenate([err_bt, err_at])
    y_pred = (y_score > best_thr).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum()); fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum()); tp = int(((y_true == 1) & (y_pred == 1)).sum())
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "threshold": float(best_thr), "calib_f1": float(best_f1),
        "test_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "test_fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "test_fnr": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        "roc_auc": float(roc_auc_score(y_true, y_score)), "pr_auc": float(average_precision_score(y_true, y_score)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp, "n_params": n_params,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_tag", required=True)
    ap.add_argument("--out_tag", default="candidate_a")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = ROOT / "candidates" / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Candidate A: classifier ===")
    cls_result = train_classifier(args.emb_tag, device=args.device)
    with open(out_dir / "classification_result.json", "w") as f:
        json.dump(cls_result, f, indent=2)
    print(json.dumps({k: v for k, v in cls_result.items() if k != "confusion_matrix"}, indent=2))

    print("=== Candidate A: autoencoder ===")
    ae_result = train_autoencoder(args.emb_tag, device=args.device)
    with open(out_dir / "anomaly_result.json", "w") as f:
        json.dump(ae_result, f, indent=2)
    print(json.dumps(ae_result, indent=2))
