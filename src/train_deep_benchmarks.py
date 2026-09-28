"""Modern deep-learning benchmark tier (Section 14: "Deep learning" category)
missing from the source paper's own baseline suite (which only compares
against Word2Vec/BPE/TF-IDF/ByteStack-ID/KD-BERT-style representations and
XGBoost/Decision-Tree/Isolation-Forest/OC-SVM downstream models -- no
Transformer, TCN, or modern CNN/MLP/GRU classifier). All models here consume
the SAME FieldFormer packet embeddings as the candidate models and the
source-method reproduction, so the comparison isolates the downstream
architecture, matching Section 14's requirement for a fair, controlled
benchmark suite.
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
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix

from train_lstm_downstream import WindowSeqDataset, collate, load_embeddings, ROOT, WINDOWS_PARQUET

SEEDS = [0, 1, 2, 3, 4]


class MLPClassifier(nn.Module):
    """Mean-pools the window then applies an MLP -- no temporal modeling at all,
    the natural 'ablate the sequence model entirely' baseline."""
    def __init__(self, d_in, n_classes, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(0.2),
                                  nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x.mean(dim=1))


class CNNClassifier(nn.Module):
    def __init__(self, d_in, n_classes, channels=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_in, channels, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.fc = nn.Linear(channels, n_classes)

    def forward(self, x):
        h = self.conv(x.transpose(1, 2))  # (B, C, N)
        pooled = h.mean(dim=-1)
        return self.fc(pooled)


class GRUClassifier(nn.Module):
    def __init__(self, d_in, n_classes, hidden=128, n_layers=2):
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, num_layers=n_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):
        out, h = self.gru(x)
        return self.fc(h[-1])


class TCNBlock(nn.Module):
    def __init__(self, d_in, d_out, kernel_size, dilation):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(d_in, d_out, kernel_size, padding=pad, dilation=dilation)
        self.chomp = pad
        self.relu = nn.ReLU()
        self.down = nn.Conv1d(d_in, d_out, 1) if d_in != d_out else None

    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :-self.chomp] if self.chomp > 0 else out
        out = self.relu(out)
        res = x if self.down is None else self.down(x)
        return self.relu(out + res)


class TCNClassifier(nn.Module):
    def __init__(self, d_in, n_classes, channels=128, levels=3, kernel_size=3):
        super().__init__()
        layers = []
        c_in = d_in
        for i in range(levels):
            layers.append(TCNBlock(c_in, channels, kernel_size, dilation=2 ** i))
            c_in = channels
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(channels, n_classes)

    def forward(self, x):
        h = self.net(x.transpose(1, 2))
        return self.fc(h.mean(dim=-1))


class TransformerClassifier(nn.Module):
    def __init__(self, d_in, n_classes, d_model=128, n_heads=4, n_layers=2, max_len=32):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        h = self.in_proj(x) + self.pos[:, :x.size(1)]
        h = self.encoder(h)
        return self.fc(h.mean(dim=1))


MODEL_REGISTRY = {
    "MLP": lambda d_in, n_cls: MLPClassifier(d_in, n_cls),
    "CNN": lambda d_in, n_cls: CNNClassifier(d_in, n_cls),
    "GRU": lambda d_in, n_cls: GRUClassifier(d_in, n_cls),
    "TCN": lambda d_in, n_cls: TCNClassifier(d_in, n_cls),
    "Transformer": lambda d_in, n_cls: TransformerClassifier(d_in, n_cls),
}


def run_one(name, emb, id2row, windows, le, classes, seed, device, epochs=15, batch_size=64, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)

    def make_loader(split, shuffle):
        sub = windows[windows["split"] == split]
        return DataLoader(WindowSeqDataset(sub, emb, id2row), batch_size=batch_size, shuffle=shuffle, collate_fn=collate)

    train_loader, val_loader, test_loader = make_loader("train", True), make_loader("val", False), make_loader("test", False)
    model = MODEL_REGISTRY[name](emb.shape[1], len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    best_val_f1, best_state = -1, None
    for epoch in range(epochs):
        model.train()
        for x, labels in train_loader:
            x = x.to(device)
            y = torch.tensor(le.transform(labels), dtype=torch.long, device=device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, labels in val_loader:
                logits = model(x.to(device))
                preds.extend(logits.argmax(-1).cpu().numpy().tolist())
                trues.extend(le.transform(labels).tolist())
        val_f1 = f1_score(trues, preds, average="macro")
        if val_f1 > best_val_f1:
            best_val_f1, best_state = val_f1, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, labels in test_loader:
            logits = model(x.to(device))
            preds.extend(logits.argmax(-1).cpu().numpy().tolist())
            trues.extend(le.transform(labels).tolist())
    macro_f1 = f1_score(trues, preds, average="macro")
    bal_acc = balanced_accuracy_score(trues, preds)
    prec, rec, f1c, support = precision_recall_fscore_support(trues, preds, labels=range(len(classes)), zero_division=0)
    per_class = {classes[i]: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1c[i]), "support": int(support[i])} for i in range(len(classes))}
    n_params = sum(p.numel() for p in model.parameters())
    return {"macro_f1": float(macro_f1), "balanced_accuracy": float(bal_acc), "per_class": per_class,
            "best_val_macro_f1": float(best_val_f1), "n_params": n_params, "seed": seed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_tag", default="fieldformer")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    emb, id2row = load_embeddings(args.emb_tag)
    windows = pd.read_parquet(WINDOWS_PARQUET)
    le = LabelEncoder().fit(windows["class"])
    classes = list(le.classes_)

    raw_dir = ROOT / "results" / "raw_seeds"
    agg_dir = ROOT / "results" / "aggregate"
    raw_dir.mkdir(parents=True, exist_ok=True)
    agg_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for name in MODEL_REGISTRY:
        seed_results = []
        for seed in SEEDS:
            r = run_one(name, emb, id2row, windows, le, classes, seed, device)
            seed_results.append(r)
            print(f"{name} seed={seed}: macro_f1={r['macro_f1']:.4f} bal_acc={r['balanced_accuracy']:.4f}")
        with open(raw_dir / f"deep_{name}.json", "w") as f:
            json.dump(seed_results, f, indent=2)
        f1s = [r["macro_f1"] for r in seed_results]
        summaries.append({"model": name, "macro_f1_mean": np.mean(f1s), "macro_f1_std": np.std(f1s),
                           "n_params": seed_results[0]["n_params"]})

    summary_df = pd.DataFrame(summaries).sort_values("macro_f1_mean", ascending=False)
    summary_df.to_csv(agg_dir / "deep_benchmarks_summary.csv", index=False)
    print(summary_df)


if __name__ == "__main__":
    main()
