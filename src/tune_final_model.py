"""Stage 3 (Section 9): validation-only Optuna (TPE) tuning of the selected
final model (Candidate C / BalancedFusion on the FieldFormer backbone).
Optimizes VALIDATION macro-F1 only -- the test set is never touched here,
consistent with freezing the architecture/hyperparameters before any test
evaluation (Section 11).
"""
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

from train_candidate_c import BalancedFusion, focal_loss
from train_lstm_downstream import WindowSeqDataset, collate, load_embeddings, ROOT, WINDOWS_PARQUET

EMB_TAG = "fieldformer"


def objective(trial, emb, id2row, windows, le, classes, benign_idx, device):
    d_model = trial.suggest_categorical("d_model", [64, 128, 192])
    n_layers = trial.suggest_int("n_layers", 1, 3)
    lr = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
    lam_recon = trial.suggest_float("lam_recon", 0.05, 1.0, log=True)
    gamma = trial.suggest_float("gamma", 0.5, 3.0)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    epochs = 12

    train_df = windows[windows["split"] == "train"]
    val_df = windows[windows["split"] == "val"]
    train_counts = train_df["class"].value_counts()
    freq = np.array([train_counts.get(c, 1) for c in classes], dtype=np.float32)
    alpha = torch.tensor((1.0 / freq) / (1.0 / freq).sum() * len(classes), dtype=torch.float32, device=device)

    train_loader = DataLoader(WindowSeqDataset(train_df, emb, id2row), batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(WindowSeqDataset(val_df, emb, id2row), batch_size=batch_size, shuffle=False, collate_fn=collate)

    model = BalancedFusion(emb.shape[1], d_model, n_layers, len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    best_val_f1 = -1
    for epoch in range(epochs):
        model.train()
        for x, labels in train_loader:
            x = x.to(device)
            y = torch.tensor(le.transform(labels), dtype=torch.long, device=device)
            opt.zero_grad()
            logits, recon, _ = model(x)
            loss_cls = focal_loss(logits, y, alpha, gamma=gamma)
            is_benign = (y == benign_idx)
            loss_recon = mse(recon[is_benign], x[is_benign]) if is_benign.any() else torch.tensor(0.0, device=device)
            (loss_cls + lam_recon * loss_recon).backward()
            opt.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, labels in val_loader:
                logits, _, _ = model(x.to(device))
                preds.extend(logits.argmax(-1).cpu().numpy().tolist())
                trues.extend(le.transform(labels).tolist())
        val_f1 = f1_score(trues, preds, average="macro")
        best_val_f1 = max(best_val_f1, val_f1)
        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_f1


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb, id2row = load_embeddings(EMB_TAG)
    windows = pd.read_parquet(WINDOWS_PARQUET)
    le = LabelEncoder().fit(windows["class"])
    classes = list(le.classes_)
    benign_idx = classes.index("benign")

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=4))
    study.optimize(lambda t: objective(t, emb, id2row, windows, le, classes, benign_idx, device),
                   n_trials=25, show_progress_bar=False)

    print("Best value (val macro-F1):", study.best_value)
    print("Best params:", study.best_params)

    out_dir = ROOT / "candidates" / "final_model_tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "optuna_best.json", "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=2)
    trials_df = study.trials_dataframe()
    trials_df.to_csv(out_dir / "optuna_trials.csv", index=False)
    print(f"Saved -> {out_dir}")


if __name__ == "__main__":
    main()
