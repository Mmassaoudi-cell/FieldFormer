"""Fair generalization-to-unseen-attack test: completely excludes ONE attack
class from training AND calibration, then measures whether that class's TEST
windows are flagged as anomalous. This is the genuine test the source
method's semisupervised design claims to pass (attack classes "withheld from
training," Section III-A/§IV-J), and is the necessary check on Candidate B's
suspiciously strong closed-set anomaly numbers (0.999 F1) -- those numbers
were computed with the prototype classifier having been TRAINED on every
attack class, which is not the same task as detecting a genuinely novel one.

Compares, on the same held-out class:
  - source method reproduction (LSTM autoencoder, already benign-only in
    training; here calibration is also restricted to exclude the held-out
    class's val windows)
  - Candidate B (ProtoGuard): retrained with the held-out class's train
    windows removed entirely, then evaluated via distance-to-nearest-KNOWN
    prototype (not just distance-to-benign) on the held-out class's test
    windows -- the fair open-set test this candidate's design claims to pass
  - Candidate C (BalancedFusion): same treatment as source method (benign-
    only reconstruction training already excludes all attacks; retrained
    classification head excludes the held-out class so its focal-loss
    representation isn't shaped by it either)
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
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_score, recall_score

from train_lstm_downstream import WindowSeqDataset, collate, load_embeddings, StackedLSTMAutoencoder, ROOT, WINDOWS_PARQUET
from train_candidate_a import SSMAutoencoder
from train_candidate_b import ProtoGuard
from train_candidate_c import BalancedFusion, focal_loss


def calibrate_and_eval(err_benign_calib, err_attack_calib, err_benign_test, err_holdout_test):
    all_calib = np.concatenate([err_benign_calib, err_attack_calib])
    labels = np.concatenate([np.zeros(len(err_benign_calib)), np.ones(len(err_attack_calib))])
    cands = np.quantile(all_calib, np.linspace(0.01, 0.99, 199))
    best_thr, best_f1 = None, -1
    for thr in cands:
        f1 = f1_score(labels, (all_calib > thr).astype(int))
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    y_true = np.concatenate([np.zeros(len(err_benign_test)), np.ones(len(err_holdout_test))])
    y_score = np.concatenate([err_benign_test, err_holdout_test])
    y_pred = (y_score > best_thr).astype(int)
    holdout_detect_rate = float((err_holdout_test > best_thr).mean())
    return {
        "threshold": float(best_thr),
        "holdout_detection_rate_pct": holdout_detect_rate * 100,
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true.tolist())) > 1 else None,
        "test_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def eval_source_method(emb_tag, holdout_class, device, epochs=20, hidden=128, n_layers=2, lr=1e-3, batch_size=64):
    emb, id2row = load_embeddings(emb_tag)
    windows = pd.read_parquet(WINDOWS_PARQUET)

    benign_train = windows[(windows["class"] == "benign") & (windows["split"] == "train")]
    benign_calib = windows[(windows["class"] == "benign") & (windows["split"] == "val")]
    benign_test = windows[(windows["class"] == "benign") & (windows["split"] == "test")]
    attack_calib = windows[(windows["class"] != "benign") & (windows["class"] != holdout_class) & (windows["split"] == "val")]
    holdout_test = windows[(windows["class"] == holdout_class) & (windows["split"] == "test")]

    loader = DataLoader(WindowSeqDataset(benign_train, emb, id2row), batch_size=batch_size, shuffle=True, collate_fn=collate)
    model = StackedLSTMAutoencoder(emb.shape[1], hidden, n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        for x, _ in loader:
            x = x.to(device)
            opt.zero_grad()
            loss = crit(model(x), x)
            loss.backward()
            opt.step()

    def errors(df):
        loader = DataLoader(WindowSeqDataset(df, emb, id2row), batch_size=batch_size, shuffle=False, collate_fn=collate)
        out = []
        model.eval()
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                recon = model(x)
                out.extend(((x - recon) ** 2).mean(dim=(1, 2)).cpu().numpy().tolist())
        return np.array(out)

    return calibrate_and_eval(errors(benign_calib), errors(attack_calib), errors(benign_test), errors(holdout_test))


def eval_candidate_a(emb_tag, holdout_class, device, epochs=20, d_model=128, n_layers=2, lr=1e-3, batch_size=64):
    emb, id2row = load_embeddings(emb_tag)
    windows = pd.read_parquet(WINDOWS_PARQUET)

    benign_train = windows[(windows["class"] == "benign") & (windows["split"] == "train")]
    benign_calib = windows[(windows["class"] == "benign") & (windows["split"] == "val")]
    benign_test = windows[(windows["class"] == "benign") & (windows["split"] == "test")]
    attack_calib = windows[(windows["class"] != "benign") & (windows["class"] != holdout_class) & (windows["split"] == "val")]
    holdout_test = windows[(windows["class"] == holdout_class) & (windows["split"] == "test")]

    loader = DataLoader(WindowSeqDataset(benign_train, emb, id2row), batch_size=batch_size, shuffle=True, collate_fn=collate)
    model = SSMAutoencoder(emb.shape[1], d_model, n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        for x, _ in loader:
            x = x.to(device)
            opt.zero_grad()
            loss = crit(model(x), x)
            loss.backward()
            opt.step()

    def errors(df):
        loader = DataLoader(WindowSeqDataset(df, emb, id2row), batch_size=batch_size, shuffle=False, collate_fn=collate)
        out = []
        model.eval()
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                recon = model(x)
                out.extend(((x - recon) ** 2).mean(dim=(1, 2)).cpu().numpy().tolist())
        return np.array(out)

    return calibrate_and_eval(errors(benign_calib), errors(attack_calib), errors(benign_test), errors(holdout_test))


def eval_candidate_b(emb_tag, holdout_class, device, epochs=15, d_model=128, n_layers=2, lr=1e-3, batch_size=64):
    emb, id2row = load_embeddings(emb_tag)
    windows = pd.read_parquet(WINDOWS_PARQUET)
    known_classes = [c for c in sorted(windows["class"].unique()) if c != holdout_class]
    le = LabelEncoder().fit(known_classes)
    benign_idx = known_classes.index("benign")

    train_df = windows[(windows["class"] != holdout_class) & (windows["split"] == "train")]
    calib_known_df = windows[(windows["class"] != holdout_class) & (windows["split"] == "val")]
    holdout_test_df = windows[(windows["class"] == holdout_class) & (windows["split"] == "test")]
    benign_test_df = windows[(windows["class"] == "benign") & (windows["split"] == "test")]
    attack_calib_df = windows[(windows["class"] != "benign") & (windows["class"] != holdout_class) & (windows["split"] == "val")]
    benign_calib_df = windows[(windows["class"] == "benign") & (windows["split"] == "val")]

    train_loader = DataLoader(WindowSeqDataset(train_df, emb, id2row), batch_size=batch_size, shuffle=True, collate_fn=collate)
    model = ProtoGuard(emb.shape[1], d_model, n_layers, len(known_classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for x, labels in train_loader:
            x = x.to(device)
            y = torch.tensor(le.transform(labels), dtype=torch.long, device=device)
            opt.zero_grad()
            logits, dists, _ = model(x)
            is_benign = (y == benign_idx).float()
            loss = crit(logits, y) + 0.1 * (is_benign * dists[:, benign_idx] ** 2).mean()
            loss.backward()
            opt.step()

    def min_known_dist(df):
        loader = DataLoader(WindowSeqDataset(df, emb, id2row), batch_size=batch_size, shuffle=False, collate_fn=collate)
        out = []
        model.eval()
        with torch.no_grad():
            for x, _ in loader:
                _, dists, _ = model(x.to(device))
                out.extend(dists.min(dim=1).values.cpu().numpy().tolist())
        return np.array(out)

    # anomaly score = distance to NEAREST known-class prototype (open-set score),
    # not just distance-to-benign -- this is the fair generalization test.
    return calibrate_and_eval(min_known_dist(benign_calib_df), min_known_dist(attack_calib_df),
                               min_known_dist(benign_test_df), min_known_dist(holdout_test_df))


def eval_candidate_c(emb_tag, holdout_class, device, epochs=15, d_model=128, n_layers=2, lr=1e-3, batch_size=64,
                      lam_recon=0.3, use_attention_pool=True, use_recon=True, use_focal=True):
    emb, id2row = load_embeddings(emb_tag)
    windows = pd.read_parquet(WINDOWS_PARQUET)
    known_classes = [c for c in sorted(windows["class"].unique()) if c != holdout_class]
    le = LabelEncoder().fit(known_classes)
    benign_idx = known_classes.index("benign")

    train_df = windows[(windows["class"] != holdout_class) & (windows["split"] == "train")]
    train_counts = train_df["class"].value_counts()
    freq = np.array([train_counts.get(c, 1) for c in known_classes], dtype=np.float32)
    alpha = torch.tensor((1.0 / freq) / (1.0 / freq).sum() * len(known_classes), dtype=torch.float32, device=device)

    if not use_recon:
        return None  # no reconstruction head -> no anomaly/held-out score to compute

    train_loader = DataLoader(WindowSeqDataset(train_df, emb, id2row), batch_size=batch_size, shuffle=True, collate_fn=collate)
    model = BalancedFusion(emb.shape[1], d_model, n_layers, len(known_classes),
                            use_attention_pool=use_attention_pool, use_recon=use_recon).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    plain_ce = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for x, labels in train_loader:
            x = x.to(device)
            y = torch.tensor(le.transform(labels), dtype=torch.long, device=device)
            opt.zero_grad()
            logits, recon, _ = model(x)
            loss_cls = focal_loss(logits, y, alpha) if use_focal else plain_ce(logits, y)
            is_benign = (y == benign_idx)
            loss_recon = mse(recon[is_benign], x[is_benign]) if is_benign.any() else torch.tensor(0.0, device=device)
            loss = loss_cls + lam_recon * loss_recon
            loss.backward()
            opt.step()

    def errors(df):
        loader = DataLoader(WindowSeqDataset(df, emb, id2row), batch_size=batch_size, shuffle=False, collate_fn=collate)
        out = []
        model.eval()
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                _, recon, _ = model(x)
                out.extend(((x - recon) ** 2).mean(dim=(1, 2)).cpu().numpy().tolist())
        return np.array(out)

    benign_calib = windows[(windows["class"] == "benign") & (windows["split"] == "val")]
    attack_calib = windows[(windows["class"] != "benign") & (windows["class"] != holdout_class) & (windows["split"] == "val")]
    benign_test = windows[(windows["class"] == "benign") & (windows["split"] == "test")]
    holdout_test = windows[(windows["class"] == holdout_class) & (windows["split"] == "test")]
    return calibrate_and_eval(errors(benign_calib), errors(attack_calib), errors(benign_test), errors(holdout_test))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_tag", required=True)
    ap.add_argument("--holdout_class", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--confirmatory", action="store_true",
                     help="Mark this class as never used during candidate selection/tuning (post-freeze confirmatory run).")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results = {"seed": args.seed, "confirmatory": args.confirmatory}
    print(f"=== Held-out class: {args.holdout_class} ===")
    print("Source method (LSTM-AE)...")
    results["source_method_lstm_ae"] = eval_source_method(args.emb_tag, args.holdout_class, args.device)
    print(json.dumps(results["source_method_lstm_ae"], indent=2))

    print("Candidate A (FieldFormer-SSM autoencoder)...")
    results["candidate_a"] = eval_candidate_a(args.emb_tag, args.holdout_class, args.device)
    print(json.dumps(results["candidate_a"], indent=2))

    print("Candidate B (ProtoGuard, open-set nearest-known-prototype score)...")
    results["candidate_b"] = eval_candidate_b(args.emb_tag, args.holdout_class, args.device)
    print(json.dumps(results["candidate_b"], indent=2))

    print("Candidate C (BalancedFusion)...")
    results["candidate_c"] = eval_candidate_c(args.emb_tag, args.holdout_class, args.device)
    print(json.dumps(results["candidate_c"], indent=2))

    out_dir = Path(__file__).resolve().parent.parent / "results" / "raw_seeds"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "confirmatory" if args.confirmatory else "selection"
    out_path = out_dir / f"heldout_{args.holdout_class}_{args.emb_tag}_{tag}_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")
