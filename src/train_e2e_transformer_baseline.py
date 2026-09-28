"""Independent end-to-end Transformer baseline that does NOT consume
FieldFormer embeddings, addressing the confound flagged in peer review
(Required Revision R3): every other deep-learning baseline in this study is
evaluated on embeddings produced by this paper's own contributed encoder,
which does not establish how such an architecture performs in its own
natural, standard configuration. This baseline learns its own byte-level
token embeddings from scratch, jointly with a two-level (packet + window)
Transformer encoder and the classification head, with no pretraining stage
and no dependency on any component proposed elsewhere in this paper.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
PACKETS_PARQUET = ROOT / "data_audit" / "modbus_packets.parquet"
WINDOWS_PARQUET = ROOT / "data_audit" / "modbus_windows.parquet"
TOKENIZER_PATH = ROOT / "data_audit" / "shared_tokenizer.json"
MAX_LEN = 320
SEEDS = [0, 1, 2, 3, 4]


def bytes_to_hex_tokens(hexstr):
    return " ".join(f"<0x{hexstr[i:i+2].upper()}>" for i in range(0, len(hexstr), 2))


class WindowRawDataset(Dataset):
    """Same window/class targets as every other benchmark, but returns raw
    tokenized byte sequences per packet (not precomputed embeddings)."""
    def __init__(self, windows_df, packets_df, tokenizer, max_len=MAX_LEN):
        self.rows = windows_df.reset_index(drop=True)
        self.packets = packets_df.set_index("pkt_id")
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows.iloc[idx]
        ids = [int(x) for x in r["pkt_ids"].split(",")]
        seqs = []
        for pid in ids:
            hexstr = self.packets.loc[pid, "raw_bytes_hex"]
            token_ids = self.tok(bytes_to_hex_tokens(hexstr), truncation=True, max_length=self.max_len)["input_ids"]
            seqs.append(token_ids)
        return seqs, r["class"]


def collate(batch, pad_id):
    seqs_batch, labels = zip(*batch)
    N = len(seqs_batch[0])
    max_len = max(len(s) for seqs in seqs_batch for s in seqs)
    B = len(seqs_batch)
    input_ids = torch.full((B, N, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((B, N, max_len), dtype=torch.long)
    for b, seqs in enumerate(seqs_batch):
        for n, s in enumerate(seqs):
            input_ids[b, n, :len(s)] = torch.tensor(s)
            attn[b, n, :len(s)] = 1
    return input_ids, attn, list(labels)


class E2ETransformer(nn.Module):
    def __init__(self, vocab_size, n_classes, d_model=128, n_heads=4, pkt_layers=2, win_layers=2, pad_id=0, max_len=MAX_LEN):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_embed = nn.Embedding(max_len, d_model)
        pkt_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model, dropout=0.1, batch_first=True)
        self.pkt_encoder = nn.TransformerEncoder(pkt_layer, pkt_layers)
        win_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model, dropout=0.1, batch_first=True)
        self.win_encoder = nn.TransformerEncoder(win_layer, win_layers)
        self.win_pos = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
        self.cls_head = nn.Linear(d_model, n_classes)

    def forward(self, input_ids, attn):
        B, N, T = input_ids.shape
        x = input_ids.view(B * N, T)
        m = attn.view(B * N, T)
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B * N, T)
        h = self.embed(x) + self.pos_embed(pos)
        key_padding = ~m.bool()
        # guard against fully-padded rows (shouldn't occur, but avoids NaN)
        h = self.pkt_encoder(h, src_key_padding_mask=key_padding)
        mask_f = m.unsqueeze(-1).float()
        pkt_repr = (h * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)  # (B*N, d_model)
        pkt_repr = pkt_repr.view(B, N, -1) + self.win_pos[:, :N]
        w = self.win_encoder(pkt_repr)
        pooled = w.mean(dim=1)
        return self.cls_head(pooled)


def run_one(seed, tokenizer, packets_df, windows_df, le, classes, device, epochs=15, batch_size=16, lr=5e-4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    pad_id = tokenizer.pad_token_id

    def make_loader(split, shuffle):
        sub = windows_df[windows_df["split"] == split]
        ds = WindowRawDataset(sub, packets_df, tokenizer)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda b: collate(b, pad_id))

    train_loader, val_loader, test_loader = make_loader("train", True), make_loader("val", False), make_loader("test", False)
    model = E2ETransformer(tokenizer.vocab_size, len(classes), pad_id=pad_id).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    best_val_f1, best_state = -1, None
    for epoch in range(epochs):
        model.train()
        for input_ids, attn, labels in train_loader:
            input_ids, attn = input_ids.to(device), attn.to(device)
            y = torch.tensor(le.transform(labels), dtype=torch.long, device=device)
            opt.zero_grad()
            logits = model(input_ids, attn)
            loss = crit(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for input_ids, attn, labels in val_loader:
                logits = model(input_ids.to(device), attn.to(device))
                preds.extend(logits.argmax(-1).cpu().numpy().tolist())
                trues.extend(le.transform(labels).tolist())
        val_f1 = f1_score(trues, preds, average="macro")
        print(f"  seed={seed} epoch={epoch}: val_macro_f1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1, best_state = val_f1, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for input_ids, attn, labels in test_loader:
            logits = model(input_ids.to(device), attn.to(device))
            preds.extend(logits.argmax(-1).cpu().numpy().tolist())
            trues.extend(le.transform(labels).tolist())
    macro_f1 = f1_score(trues, preds, average="macro")
    bal_acc = balanced_accuracy_score(trues, preds)
    n_params = sum(p.numel() for p in model.parameters())
    return {"macro_f1": float(macro_f1), "balanced_accuracy": float(bal_acc),
            "best_val_macro_f1": float(best_val_f1), "n_params": n_params, "seed": seed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", bos_token="<s>",
                                         eos_token="</s>", unk_token="<unk>")
    packets_df = pd.read_parquet(PACKETS_PARQUET, columns=["pkt_id", "raw_bytes_hex"])
    windows_df = pd.read_parquet(WINDOWS_PARQUET)
    le = LabelEncoder().fit(windows_df["class"])
    classes = list(le.classes_)

    raw_dir = ROOT / "results" / "raw_seeds"
    raw_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in args.seeds:
        r = run_one(seed, tokenizer, packets_df, windows_df, le, classes, device)
        results.append(r)
        print(f"seed={seed}: test_macro_f1={r['macro_f1']:.4f}")

    with open(raw_dir / "deep_E2ETransformer_no_pretrain.json", "w") as f:
        json.dump(results, f, indent=2)
    f1s = [r["macro_f1"] for r in results]
    summary = {"model": "E2ETransformer_no_pretrain", "macro_f1_mean": float(np.mean(f1s)),
               "macro_f1_std": float(np.std(f1s)), "n_params": results[0]["n_params"]}
    print(json.dumps(summary, indent=2))
    with open(ROOT / "results" / "aggregate" / "e2e_transformer_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
