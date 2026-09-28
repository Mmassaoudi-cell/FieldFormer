"""Candidate C (BalancedFusion): single shared SSM backbone (same as
Candidates A/B, held constant to isolate this candidate's actual
contribution) with (1) attention pooling instead of last-timestep pooling,
and (2) class-balanced focal loss for classification, trained JOINTLY with a
benign-only reconstruction head -- one backbone instead of the source
method's two independently trained stacks. Targets the one place the source
method's own numbers (and our reproduction) show a clear, numeric weakness:
low-footprint/minority-class recall (paper's Reconnaissance class 94.58% vs
>99% elsewhere; our reproduction's modbus_query_flood 0.736 F1 vs >0.96 for
several other classes).
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


class AttentionPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w = nn.Linear(d_model, 1)

    def forward(self, h):
        # h: (B, N, D)
        scores = self.w(h).squeeze(-1)  # (B, N)
        alpha = torch.softmax(scores, dim=-1)
        return (h * alpha.unsqueeze(-1)).sum(1)


class BalancedFusion(nn.Module):
    """`use_attention_pool=False` ablates attention pooling to last-timestep
    pooling (matching the source method's own pooling choice); `use_recon=
    False` ablates the joint reconstruction head (classification-only,
    matching the source method's fully separate-model design instead of a
    shared backbone)."""
    def __init__(self, d_in, d_model=128, n_layers=2, n_classes=6, use_attention_pool=True, use_recon=True):
        super().__init__()
        self.backbone = SSMBackbone(d_in, d_model, n_layers)
        self.use_attention_pool = use_attention_pool
        self.use_recon = use_recon
        if use_attention_pool:
            self.pool = AttentionPool(d_model)
        self.cls_head = nn.Linear(d_model, n_classes)
        if use_recon:
            self.recon_head = nn.Linear(d_model, d_in)

    def forward(self, x):
        h = self.backbone(x)  # (B, N, D)
        pooled = self.pool(h) if self.use_attention_pool else h[:, -1]
        logits = self.cls_head(pooled)
        recon = self.recon_head(h) if self.use_recon else None
        return logits, recon, pooled


def focal_loss(logits, targets, alpha, gamma=2.0):
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    ce = torch.nn.functional.nll_loss(logp, targets, reduction="none")
    pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)
    a = alpha[targets]
    loss = a * (1 - pt) ** gamma * ce
    return loss.mean()


def run(emb_tag, epochs=15, d_model=128, n_layers=2, lr=1e-3, batch_size=64, lam_recon=0.3, device="cuda",
        use_attention_pool=True, use_recon=True, use_focal=True, save_path=None):
    emb, id2row = load_embeddings(emb_tag)
    windows = pd.read_parquet(WINDOWS_PARQUET)
    le = LabelEncoder().fit(windows["class"])
    classes = list(le.classes_)
    benign_idx = classes.index("benign")

    train_counts = windows[windows["split"] == "train"]["class"].value_counts()
    freq = np.array([train_counts.get(c, 1) for c in classes], dtype=np.float32)
    alpha = torch.tensor((1.0 / freq) / (1.0 / freq).sum() * len(classes), dtype=torch.float32, device=device)

    def make_loader(split, shuffle):
        sub = windows[windows["split"] == split]
        return DataLoader(WindowSeqDataset(sub, emb, id2row), batch_size=batch_size, shuffle=shuffle, collate_fn=collate)

    train_loader, val_loader, test_loader = make_loader("train", True), make_loader("val", False), make_loader("test", False)
    model = BalancedFusion(emb.shape[1], d_model, n_layers, len(classes),
                            use_attention_pool=use_attention_pool, use_recon=use_recon).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    plain_ce = nn.CrossEntropyLoss()

    best_val_f1, best_state = -1, None
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, labels in train_loader:
            x = x.to(device)
            y = torch.tensor(le.transform(labels), dtype=torch.long, device=device)
            opt.zero_grad()
            logits, recon, _ = model(x)
            loss_cls = focal_loss(logits, y, alpha) if use_focal else plain_ce(logits, y)
            if use_recon:
                is_benign = (y == benign_idx)
                loss_recon = mse(recon[is_benign], x[is_benign]) if is_benign.any() else torch.tensor(0.0, device=device)
            else:
                loss_recon = torch.tensor(0.0, device=device)
            loss = loss_cls + (lam_recon * loss_recon if use_recon else 0.0)
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
        print(f"[CandC] epoch {epoch}: loss={total_loss/len(train_loader.dataset):.4f} val_macro_f1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1, best_state = val_f1, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    if save_path is not None:
        torch.save({"state_dict": model.state_dict(), "d_model": d_model, "n_layers": n_layers,
                    "classes": classes, "use_attention_pool": use_attention_pool, "use_recon": use_recon}, save_path)
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

    if not use_recon:
        return cls_result, None

    # benign-only reconstruction-error anomaly score, same calibration protocol
    def recon_errors(df):
        loader = DataLoader(WindowSeqDataset(df, emb, id2row), batch_size=batch_size, shuffle=False, collate_fn=collate)
        errs = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                _, recon, _ = model(x)
                errs.extend(((x - recon) ** 2).mean(dim=(1, 2)).cpu().numpy().tolist())
        return np.array(errs)

    benign_val = windows[(windows["class"] == "benign") & (windows["split"] == "val")]
    attack_val = windows[(windows["class"] != "benign") & (windows["split"] == "val")]
    benign_test = windows[(windows["class"] == "benign") & (windows["split"] == "test")]
    attack_test = windows[(windows["class"] != "benign") & (windows["split"] == "test")]
    err_bc, err_ac = recon_errors(benign_val), recon_errors(attack_val)
    err_bt, err_at = recon_errors(benign_test), recon_errors(attack_test)

    all_calib = np.concatenate([err_bc, err_ac])
    calib_labels = np.concatenate([np.zeros(len(err_bc)), np.ones(len(err_ac))])
    cands = np.quantile(all_calib, np.linspace(0.01, 0.99, 199))
    best_thr, best_f1 = None, -1
    for thr in cands:
        f1 = f1_score(calib_labels, (all_calib > thr).astype(int))
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    y_true = np.concatenate([np.zeros(len(err_bt)), np.ones(len(err_at))])
    y_score = np.concatenate([err_bt, err_at])
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
    ap.add_argument("--out_tag", default="candidate_c")
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
    print("anomaly:", json.dumps(anomaly_result, indent=2))
