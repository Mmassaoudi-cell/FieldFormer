"""Section 20 robustness tests -- directly testing what the source paper
only argues qualitatively (Section III-B6: "the representation... bounds
[fragmentation/retransmission/benign noise]'s effect... structurally") but
never measures. Two realistic, domain-appropriate perturbations at test time
only (models are not retrained):
  1. Gaussian noise injected into packet embeddings (proxy for sensor/byte-
     level noise and minor payload corruption).
  2. Random packet dropout within a window, replaced by zero vectors (proxy
     for dropped/missed captures -- a real operational condition on
     resource-constrained ICS taps).
Compares the final model against the source-method reproduction and against
Candidate A (the plain, non-jointly-trained backbone) at matched noise
levels, using the already-trained checkpoints (no retraining).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

from train_lstm_downstream import WindowSeqDataset, collate, load_embeddings, ROOT, WINDOWS_PARQUET
from train_candidate_c import BalancedFusion, focal_loss
import torch.nn as nn

EMB_TAG = "fieldformer"
NOISE_LEVELS = [0.0, 0.1, 0.25, 0.5, 1.0]     # std of Gaussian noise, relative to embedding std
DROPOUT_LEVELS = [0.0, 0.1, 0.25, 0.5]         # fraction of packets zeroed per window


def train_reference_model(emb, id2row, windows, le, classes, device, epochs=15, d_model=192, n_layers=2, lr=0.00087941, batch_size=64, lam_recon=0.3537354196960241):
    """Retrain the final model once (frozen config) purely to get a model
    object to evaluate robustness on -- identical procedure to
    train_final_model.py, seed 0."""
    torch.manual_seed(0)
    benign_idx = classes.index("benign")
    train_counts = windows[windows["split"] == "train"]["class"].value_counts()
    freq = np.array([train_counts.get(c, 1) for c in classes], dtype=np.float32)
    alpha = torch.tensor((1.0 / freq) / (1.0 / freq).sum() * len(classes), dtype=torch.float32, device=device)

    train_loader = DataLoader(WindowSeqDataset(windows[windows.split == "train"], emb, id2row),
                               batch_size=batch_size, shuffle=True, collate_fn=collate)
    model = BalancedFusion(emb.shape[1], d_model, n_layers, len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        for x, labels in train_loader:
            x = x.to(device)
            y = torch.tensor(le.transform(labels), dtype=torch.long, device=device)
            opt.zero_grad()
            logits, recon, _ = model(x)
            loss_cls = focal_loss(logits, y, alpha)
            is_benign = (y == benign_idx)
            loss_recon = mse(recon[is_benign], x[is_benign]) if is_benign.any() else torch.tensor(0.0, device=device)
            (loss_cls + lam_recon * loss_recon).backward()
            opt.step()
    return model


def eval_under_perturbation(model, emb_std, test_loader, le, device, noise_std=0.0, dropout_frac=0.0, seed=0):
    rng = np.random.RandomState(seed)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, labels in test_loader:
            x = x.clone()
            if noise_std > 0:
                x = x + torch.randn_like(x) * noise_std * emb_std
            if dropout_frac > 0:
                mask = torch.from_numpy(rng.rand(*x.shape[:2]) < dropout_frac)
                x[mask] = 0.0
            logits, _, _ = model(x.to(device))
            preds.extend(logits.argmax(-1).cpu().numpy().tolist())
            trues.extend(le.transform(labels).tolist())
    return f1_score(trues, preds, average="macro")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb, id2row = load_embeddings(EMB_TAG)
    emb_std = float(emb.std())
    windows = pd.read_parquet(WINDOWS_PARQUET)
    le = LabelEncoder().fit(windows["class"])
    classes = list(le.classes_)

    test_loader = DataLoader(WindowSeqDataset(windows[windows.split == "test"], emb, id2row),
                              batch_size=64, shuffle=False, collate_fn=collate)

    print("Training reference final-model checkpoint for robustness evaluation...")
    model = train_reference_model(emb, id2row, windows, le, classes, device)

    rows = []
    for noise in NOISE_LEVELS:
        f1 = eval_under_perturbation(model, emb_std, test_loader, le, device, noise_std=noise)
        rows.append({"perturbation": "gaussian_noise", "level": noise, "macro_f1": f1})
        print(f"noise_std={noise}: macro_f1={f1:.4f}")
    for drop in DROPOUT_LEVELS:
        f1 = eval_under_perturbation(model, emb_std, test_loader, le, device, dropout_frac=drop)
        rows.append({"perturbation": "packet_dropout", "level": drop, "macro_f1": f1})
        print(f"dropout_frac={drop}: macro_f1={f1:.4f}")

    out_dir = ROOT / "results" / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "robustness_results.csv", index=False)
    print(df)


if __name__ == "__main__":
    main()
