"""Downstream detection models matching the source paper's Fig. 3 (Stacked
LSTM classifier) and Fig. 4 (Stacked LSTM autoencoder, semisupervised
anomaly detection), consuming the BART-embedding sequences produced by
extract_embeddings.py.

Classification: cross-entropy over window embeddings -> final hidden state ->
FC -> softmax (Eq. 1).
Anomaly detection: benign-only training; reconstruction MSE (Eq. 2); the
paper's 60/20/20 benign train/calibration/test protocol (Section IV-J) is
approximated here using our existing train/val/test split's benign windows
(train=train, calibration=val, test=test), since the paper does not specify
how that split interacts with the classification split -- documented as an
implementation assumption in REPRODUCTION_REPORT.md.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix,
    precision_score, recall_score, roc_auc_score, average_precision_score,
)

ROOT = Path(__file__).resolve().parent.parent
WINDOWS_PARQUET = ROOT / "data_audit" / "modbus_windows.parquet"


def load_embeddings(tag):
    emb = np.load(ROOT / "data_audit" / f"embeddings_{tag}.npy")
    pktids = np.load(ROOT / "data_audit" / f"embeddings_{tag}_pktids.npy")
    id2row = {int(pid): i for i, pid in enumerate(pktids)}
    return emb, id2row


class WindowSeqDataset(Dataset):
    def __init__(self, windows_df, emb, id2row):
        self.rows = windows_df.reset_index(drop=True)
        self.emb = emb
        self.id2row = id2row

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows.iloc[idx]
        ids = [int(x) for x in r["pkt_ids"].split(",")]
        seq = np.stack([self.emb[self.id2row[i]] for i in ids])  # (N, D)
        return torch.from_numpy(seq).float(), r["class"]


class StackedLSTMClassifier(nn.Module):
    def __init__(self, d_in, hidden=128, n_layers=2, n_classes=6, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(d_in, hidden, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        last = h[-1]  # final hidden state of top layer
        return self.fc(last)


class StackedLSTMAutoencoder(nn.Module):
    def __init__(self, d_in, hidden=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.encoder = nn.LSTM(d_in, hidden, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(hidden, hidden, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.out_proj = nn.Linear(hidden, d_in)

    def forward(self, x):
        N = x.size(1)
        _, (h, c) = self.encoder(x)
        z = h[-1].unsqueeze(1).repeat(1, N, 1)  # broadcast latent across window
        dec_out, _ = self.decoder(z)
        recon = self.out_proj(dec_out)
        return recon


def collate(batch):
    seqs, labels = zip(*batch)
    return torch.stack(seqs), list(labels)


def train_classifier(emb_tag, epochs=15, hidden=128, n_layers=2, lr=1e-3, batch_size=64, device="cuda"):
    emb, id2row = load_embeddings(emb_tag)
    windows = pd.read_parquet(WINDOWS_PARQUET)
    le = LabelEncoder().fit(windows["class"])
    classes = list(le.classes_)

    def make_loader(split, shuffle):
        sub = windows[windows["split"] == split]
        ds = WindowSeqDataset(sub, emb, id2row)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)

    train_loader = make_loader("train", True)
    val_loader = make_loader("val", False)
    test_loader = make_loader("test", False)

    model = StackedLSTMClassifier(emb.shape[1], hidden, n_layers, len(classes)).to(device)
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
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for x, labels in val_loader:
                x = x.to(device)
                logits = model(x)
                val_preds.extend(logits.argmax(-1).cpu().numpy().tolist())
                val_true.extend(le.transform(labels).tolist())
        val_f1 = f1_score(val_true, val_preds, average="macro")
        print(f"epoch {epoch}: train_loss={total_loss/len(train_loader.dataset):.4f} val_macro_f1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for x, labels in test_loader:
            x = x.to(device)
            logits = model(x)
            test_preds.extend(logits.argmax(-1).cpu().numpy().tolist())
            test_true.extend(le.transform(labels).tolist())

    macro_f1 = f1_score(test_true, test_preds, average="macro")
    weighted_f1 = f1_score(test_true, test_preds, average="weighted")
    bal_acc = balanced_accuracy_score(test_true, test_preds)
    prec, rec, f1c, support = precision_recall_fscore_support(test_true, test_preds, labels=range(len(classes)), zero_division=0)
    per_class = {classes[i]: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1c[i]), "support": int(support[i])} for i in range(len(classes))}
    cm = confusion_matrix(test_true, test_preds, labels=range(len(classes))).tolist()

    return {
        "macro_f1": float(macro_f1), "weighted_f1": float(weighted_f1),
        "balanced_accuracy": float(bal_acc), "per_class": per_class,
        "confusion_matrix": cm, "classes": classes, "best_val_macro_f1": float(best_val_f1),
    }


def train_autoencoder(emb_tag, epochs=20, hidden=128, n_layers=2, lr=1e-3, batch_size=64, device="cuda"):
    emb, id2row = load_embeddings(emb_tag)
    windows = pd.read_parquet(WINDOWS_PARQUET)

    benign_train = windows[(windows["class"] == "benign") & (windows["split"] == "train")]
    benign_calib = windows[(windows["class"] == "benign") & (windows["split"] == "val")]
    benign_test = windows[(windows["class"] == "benign") & (windows["split"] == "test")]
    attack_calib = windows[(windows["class"] != "benign") & (windows["split"] == "val")]
    attack_test = windows[(windows["class"] != "benign") & (windows["split"] == "test")]

    train_loader = DataLoader(WindowSeqDataset(benign_train, emb, id2row), batch_size=batch_size, shuffle=True, collate_fn=collate)

    model = StackedLSTMAutoencoder(emb.shape[1], hidden, n_layers).to(device)
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
        print(f"AE epoch {epoch}: train_recon_mse={total_loss/len(train_loader.dataset):.6f}")

    def recon_errors(df):
        loader = DataLoader(WindowSeqDataset(df, emb, id2row), batch_size=batch_size, shuffle=False, collate_fn=collate)
        errs = []
        model.eval()
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                recon = model(x)
                e = ((x - recon) ** 2).mean(dim=(1, 2))
                errs.extend(e.cpu().numpy().tolist())
        return np.array(errs)

    err_benign_calib = recon_errors(benign_calib)
    err_attack_calib = recon_errors(attack_calib)
    err_benign_test = recon_errors(benign_test)
    err_attack_test = recon_errors(attack_test)

    # threshold sweep on calibration partition, maximize F1 (benign=negative, attack=positive)
    # -- exactly as the source paper describes (Section IV-J). Note (see
    # REPRODUCTION_REPORT.md): under an attack-dominated calibration set this
    # systematically biases toward high-FPR operating points regardless of
    # representation quality, so a class-BALANCED variant is also computed
    # below for comparison (subsample the majority class in calibration to
    # match the minority before the same F1-max sweep).
    all_calib = np.concatenate([err_benign_calib, err_attack_calib])
    calib_labels = np.concatenate([np.zeros(len(err_benign_calib)), np.ones(len(err_attack_calib))])
    candidates = np.quantile(all_calib, np.linspace(0.01, 0.99, 199))
    best_thr, best_f1 = None, -1
    for thr in candidates:
        pred = (all_calib > thr).astype(int)
        f1 = f1_score(calib_labels, pred)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    rng = np.random.RandomState(0)
    n_bal = min(len(err_benign_calib), len(err_attack_calib))
    bal_benign = rng.choice(err_benign_calib, n_bal, replace=False) if len(err_benign_calib) > n_bal else err_benign_calib
    bal_attack = rng.choice(err_attack_calib, n_bal, replace=False) if len(err_attack_calib) > n_bal else err_attack_calib
    bal_calib = np.concatenate([bal_benign, bal_attack])
    bal_labels = np.concatenate([np.zeros(n_bal), np.ones(n_bal)])
    bal_candidates = np.quantile(bal_calib, np.linspace(0.01, 0.99, 199))
    best_thr_bal, best_f1_bal = None, -1
    for thr in bal_candidates:
        f1 = f1_score(bal_labels, (bal_calib > thr).astype(int))
        if f1 > best_f1_bal:
            best_f1_bal, best_thr_bal = f1, thr

    y_true = np.concatenate([np.zeros(len(err_benign_test)), np.ones(len(err_attack_test))])
    y_score = np.concatenate([err_benign_test, err_attack_test])
    y_pred = (y_score > best_thr).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    roc_auc = float(roc_auc_score(y_true, y_score))
    pr_auc = float(average_precision_score(y_true, y_score))

    # balanced-calibration operating point (see note above threshold sweep)
    y_pred_bal = (y_score > best_thr_bal).astype(int)
    tn_b = int(((y_true == 0) & (y_pred_bal == 0)).sum())
    fp_b = int(((y_true == 0) & (y_pred_bal == 1)).sum())
    fn_b = int(((y_true == 1) & (y_pred_bal == 0)).sum())
    tp_b = int(((y_true == 1) & (y_pred_bal == 1)).sum())
    balanced_operating_point = {
        "threshold": float(best_thr_bal), "calib_f1": float(best_f1_bal),
        "test_precision": float(precision_score(y_true, y_pred_bal, zero_division=0)),
        "test_recall": float(recall_score(y_true, y_pred_bal, zero_division=0)),
        "test_f1": float(f1_score(y_true, y_pred_bal, zero_division=0)),
        "test_fpr": fp_b / (fp_b + tn_b) if (fp_b + tn_b) > 0 else 0.0,
        "test_fnr": fn_b / (fn_b + tp_b) if (fn_b + tp_b) > 0 else 0.0,
        "tn": tn_b, "fp": fp_b, "fn": fn_b, "tp": tp_b,
    }

    # per-class detection rate on test (attack classes), matching Table VII
    per_class_detect = {}
    for cls in sorted(windows["class"].unique()):
        if cls == "benign":
            continue
        sub = windows[(windows["class"] == cls) & (windows["split"] == "test")]
        if len(sub) == 0:
            continue
        e = recon_errors(sub)
        det_rate = float((e > best_thr).mean())
        per_class_detect[cls] = {"windows": len(sub), "detected_pct": det_rate * 100}

    return {
        "threshold": float(best_thr), "calib_f1": float(best_f1),
        "test_precision": float(precision), "test_recall": float(recall), "test_f1": float(f1),
        "test_specificity": float(specificity), "test_fpr": float(fpr), "test_fnr": float(fnr),
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "n_benign_test": len(err_benign_test), "n_attack_test": len(err_attack_test),
        "per_class_detection_rate": per_class_detect,
        "balanced_calibration_operating_point": balanced_operating_point,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_tag", required=True)
    ap.add_argument("--out_tag", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = ROOT / "SOURCE_METHOD_REPRODUCTION"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Training classifier ===")
    cls_result = train_classifier(args.emb_tag, device=args.device)
    with open(out_dir / f"classification_result_{args.out_tag}.json", "w") as f:
        json.dump(cls_result, f, indent=2)
    print(json.dumps({k: v for k, v in cls_result.items() if k != "confusion_matrix"}, indent=2))

    print("=== Training autoencoder ===")
    ae_result = train_autoencoder(args.emb_tag, device=args.device)
    with open(out_dir / f"anomaly_result_{args.out_tag}.json", "w") as f:
        json.dump(ae_result, f, indent=2)
    print(json.dumps(ae_result, indent=2))
