"""Cross-dataset generalization: trains the final model once on Frazao et al.
(with the frozen configuration, exactly as in train_final_model.py) and
evaluates it, with NO retraining and NO recalibration, on Edge-IIoTset
(Ferrag et al. 2022) -- a completely independent testbed, different physical
devices, different address space, captured by a different research group.
This directly answers the single-testbed/overgeneralization concern raised
in peer review: the calibration threshold, class prototypes, and all learned
weights are frozen from Frazao-only training before Edge-IIoTset is touched
at all.

Shared classes between the two datasets: benign (Modbus), mitm (ARP
spoofing), icmp_flood, tcp_syn_flood. Edge-IIoTset has no Modbus-query-flood
equivalent, so classification is evaluated over these 4 shared classes only
(logits for the 2 Frazao-only classes are masked out at inference).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix

from train_fieldformer_encoder import FieldFormerEncoder, bytes_to_hex_tokens
from train_candidate_c import BalancedFusion, focal_loss
from train_lstm_downstream import WindowSeqDataset, collate, WINDOWS_PARQUET

ROOT = Path(__file__).resolve().parent.parent
FF_CKPT = ROOT / "candidates" / "fieldformer" / "encoder.pt"
TOKENIZER_PATH = ROOT / "data_audit" / "shared_tokenizer.json"
FROZEN = dict(epochs=15, d_model=192, n_layers=2, lr=0.00087941, batch_size=64, lam_recon=0.3537354196960241)
SEED = 0


def extract_embeddings_for(packets_parquet, out_tag, ff, tokenizer, device, batch_size=256, max_len=320):
    df = pd.read_parquet(packets_parquet, columns=["pkt_id", "raw_bytes_hex"])
    n = len(df)
    d_model = ff.embed.embedding_dim
    out = np.zeros((n, d_model), dtype=np.float32)
    pkt_ids = df["pkt_id"].to_numpy()
    ff.eval()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = df.iloc[i:i + batch_size]
            texts = [bytes_to_hex_tokens(h) for h in batch["raw_bytes_hex"]]
            enc = tokenizer(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            h, _ = ff(enc["input_ids"], enc["attention_mask"])
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            out[i:i + len(batch)] = pooled.cpu().numpy()
    emb_path = ROOT / "data_audit" / f"embeddings_{out_tag}.npy"
    ids_path = ROOT / "data_audit" / f"embeddings_{out_tag}_pktids.npy"
    np.save(emb_path, out)
    np.save(ids_path, pkt_ids)
    return out, pkt_ids


def train_frozen_final_model(device):
    """Train the final model exactly per FINAL_MODEL_CONFIG.yaml (seed 0),
    and additionally compute + persist its Frazao-only calibration threshold,
    so Edge-IIoTset evaluation below reuses it unchanged."""
    from train_lstm_downstream import load_embeddings
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    emb, id2row = load_embeddings("fieldformer")
    windows = pd.read_parquet(WINDOWS_PARQUET)
    classes = sorted(windows["class"].unique())
    le = LabelEncoder().fit(classes)
    benign_idx = classes.index("benign")

    train_df = windows[windows["split"] == "train"]
    val_df = windows[windows["split"] == "val"]
    train_counts = train_df["class"].value_counts()
    freq = np.array([train_counts.get(c, 1) for c in classes], dtype=np.float32)
    alpha = torch.tensor((1.0 / freq) / (1.0 / freq).sum() * len(classes), dtype=torch.float32, device=device)

    train_loader = DataLoader(WindowSeqDataset(train_df, emb, id2row), batch_size=FROZEN["batch_size"], shuffle=True, collate_fn=collate)
    val_loader = DataLoader(WindowSeqDataset(val_df, emb, id2row), batch_size=FROZEN["batch_size"], shuffle=False, collate_fn=collate)

    model = BalancedFusion(emb.shape[1], FROZEN["d_model"], FROZEN["n_layers"], len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=FROZEN["lr"])
    mse = nn.MSELoss()

    best_val_f1, best_state = -1, None
    for epoch in range(FROZEN["epochs"]):
        model.train()
        for x, labels in train_loader:
            x = x.to(device)
            y = torch.tensor(le.transform(labels), dtype=torch.long, device=device)
            opt.zero_grad()
            logits, recon, _ = model(x)
            loss_cls = focal_loss(logits, y, alpha)
            is_benign = (y == benign_idx)
            loss_recon = mse(recon[is_benign], x[is_benign]) if is_benign.any() else torch.tensor(0.0, device=device)
            (loss_cls + FROZEN["lam_recon"] * loss_recon).backward()
            opt.step()
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, labels in val_loader:
                logits, _, _ = model(x.to(device))
                preds.extend(logits.argmax(-1).cpu().numpy().tolist())
                trues.extend(le.transform(labels).tolist())
        val_f1 = f1_score(trues, preds, average="macro")
        if val_f1 > best_val_f1:
            best_val_f1, best_state = val_f1, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    print(f"Trained frozen final model, best_val_macro_f1={best_val_f1:.4f}")

    # Frazao-only calibration threshold (benign val vs attack val reconstruction error)
    def errors(df):
        loader = DataLoader(WindowSeqDataset(df, emb, id2row), batch_size=64, shuffle=False, collate_fn=collate)
        out = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                _, recon, _ = model(x)
                out.extend(((x - recon) ** 2).mean(dim=(1, 2)).cpu().numpy().tolist())
        return np.array(out)

    benign_calib = windows[(windows["class"] == "benign") & (windows["split"] == "val")]
    attack_calib = windows[(windows["class"] != "benign") & (windows["split"] == "val")]
    err_bc, err_ac = errors(benign_calib), errors(attack_calib)
    all_calib = np.concatenate([err_bc, err_ac])
    labels_calib = np.concatenate([np.zeros(len(err_bc)), np.ones(len(err_ac))])
    cands = np.quantile(all_calib, np.linspace(0.01, 0.99, 199))
    best_thr, best_f1 = None, -1
    for thr in cands:
        f1 = f1_score(labels_calib, (all_calib > thr).astype(int))
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    return model, classes, float(best_thr)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=== Step 1: train frozen final model on Frazao (source domain) ===")
    model, source_classes, threshold = train_frozen_final_model(device)
    print(f"Frazao-calibrated threshold: {threshold:.6f}")

    print("=== Step 2: extract FieldFormer embeddings for Edge-IIoTset (target domain, never trained on) ===")
    ff_ckpt = torch.load(FF_CKPT, map_location=device)
    ff = FieldFormerEncoder(ff_ckpt["vocab_size"], ff_ckpt["d_model"], ff_ckpt["n_layers"], pad_id=ff_ckpt["pad_id"]).to(device)
    ff.load_state_dict(ff_ckpt["state_dict"])
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", bos_token="<s>", eos_token="</s>", unk_token="<unk>")

    edge_packets = ROOT / "data_audit" / "edgeiiotset_packets.parquet"
    emb, pkt_ids = extract_embeddings_for(edge_packets, "edgeiiotset", ff, tokenizer, device)
    id2row = {int(pid): i for i, pid in enumerate(pkt_ids)}

    print("=== Step 3: zero-shot evaluation on Edge-IIoTset ===")
    edge_windows = pd.read_parquet(ROOT / "data_audit" / "edgeiiotset_windows.parquet")
    edge_windows["split"] = "test"  # WindowSeqDataset expects a split col; unused here beyond loading all rows

    shared_classes = ["benign", "mitm", "icmp_flood", "tcp_syn_flood"]
    class_to_source_idx = {c: source_classes.index(c) for c in shared_classes}

    loader = DataLoader(WindowSeqDataset(edge_windows, emb, id2row), batch_size=32, shuffle=False, collate_fn=collate)
    all_logits, all_labels, all_recon_err = [], [], []
    model.eval()
    with torch.no_grad():
        for x, labels in loader:
            x = x.to(device)
            logits, recon, _ = model(x)
            all_logits.append(logits.cpu().numpy())
            all_labels.extend(labels)
            err = ((x - recon) ** 2).mean(dim=(1, 2)).cpu().numpy()
            all_recon_err.extend(err.tolist())
    all_logits = np.concatenate(all_logits, axis=0)

    # classification restricted to the 4 shared classes (mask out Frazao-only logits)
    shared_idx = [class_to_source_idx[c] for c in shared_classes]
    restricted_logits = all_logits[:, shared_idx]
    preds_shared = restricted_logits.argmax(axis=1)
    le_shared = LabelEncoder().fit(shared_classes)
    trues_shared = le_shared.transform(all_labels)
    macro_f1 = f1_score(trues_shared, preds_shared, average="macro")
    bal_acc = balanced_accuracy_score(trues_shared, preds_shared)
    prec, rec, f1c, support = precision_recall_fscore_support(trues_shared, preds_shared, labels=range(4), zero_division=0)
    per_class = {shared_classes[i]: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1c[i]), "support": int(support[i])} for i in range(4)}
    cm = confusion_matrix(trues_shared, preds_shared, labels=range(4)).tolist()

    # anomaly detection using the Frazao-calibrated threshold, unchanged
    is_benign = np.array([lbl == "benign" for lbl in all_labels])
    y_true = (~is_benign).astype(int)
    y_score = np.array(all_recon_err)
    y_pred = (y_score > threshold).astype(int)
    roc_auc = float(roc_auc_score(y_true, y_score)) if len(set(y_true.tolist())) > 1 else None
    pr_auc = float(average_precision_score(y_true, y_score)) if len(set(y_true.tolist())) > 1 else None
    anomaly_f1 = float(f1_score(y_true, y_pred, zero_division=0))
    tn = int(((y_true == 0) & (y_pred == 0)).sum()); fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum()); tp = int(((y_true == 1) & (y_pred == 1)).sum())

    # Cheap target-domain recalibration check: NO model retraining, only the
    # scalar threshold is recomputed, using a disjoint half of Edge-IIoTset's
    # own windows (stratified benign/attack), evaluated on the other half.
    rng = np.random.RandomState(0)
    idx = np.arange(len(y_true))
    benign_idx = idx[y_true == 0]
    attack_idx = idx[y_true == 1]
    rng.shuffle(benign_idx); rng.shuffle(attack_idx)
    b_calib, b_test = benign_idx[:len(benign_idx)//2], benign_idx[len(benign_idx)//2:]
    a_calib, a_test = attack_idx[:len(attack_idx)//2], attack_idx[len(attack_idx)//2:]
    calib_scores = np.concatenate([y_score[b_calib], y_score[a_calib]])
    calib_labels = np.concatenate([np.zeros(len(b_calib)), np.ones(len(a_calib))])
    cands = np.quantile(calib_scores, np.linspace(0.01, 0.99, 199))
    best_thr_recal, best_f1_recal = None, -1
    for thr in cands:
        f1 = f1_score(calib_labels, (calib_scores > thr).astype(int))
        if f1 > best_f1_recal:
            best_f1_recal, best_thr_recal = f1, thr
    test_idx = np.concatenate([b_test, a_test])
    y_true_recal = y_true[test_idx]
    y_score_recal = y_score[test_idx]
    y_pred_recal = (y_score_recal > best_thr_recal).astype(int)
    tn_r = int(((y_true_recal == 0) & (y_pred_recal == 0)).sum()); fp_r = int(((y_true_recal == 0) & (y_pred_recal == 1)).sum())
    fn_r = int(((y_true_recal == 1) & (y_pred_recal == 0)).sum()); tp_r = int(((y_true_recal == 1) & (y_pred_recal == 1)).sum())

    result = {
        "classification_macro_f1": float(macro_f1), "classification_balanced_accuracy": float(bal_acc),
        "classification_per_class": per_class, "classification_confusion_matrix": cm,
        "anomaly_roc_auc_zero_shot": roc_auc, "anomaly_pr_auc_zero_shot": pr_auc,
        "anomaly_f1_zero_shot": anomaly_f1, "anomaly_fpr_zero_shot": fp / (fp + tn) if (fp + tn) > 0 else None,
        "anomaly_fnr_zero_shot": fn / (fn + tp) if (fn + tp) > 0 else None,
        "threshold_reused_from_frazao": threshold,
        "recalibrated_threshold": float(best_thr_recal),
        "recalibrated_f1": float(f1_score(y_true_recal, y_pred_recal, zero_division=0)),
        "recalibrated_fpr": fp_r / (fp_r + tn_r) if (fp_r + tn_r) > 0 else None,
        "recalibrated_fnr": fn_r / (fn_r + tp_r) if (fn_r + tp_r) > 0 else None,
        "recalibrated_n_benign_calib": int(len(b_calib)), "recalibrated_n_attack_calib": int(len(a_calib)),
        "n_windows": len(edge_windows), "shared_classes": shared_classes,
    }
    out_dir = ROOT / "results" / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "cross_dataset_edgeiiotset.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "classification_confusion_matrix"}, indent=2))


if __name__ == "__main__":
    main()
