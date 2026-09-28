"""Candidate A (FieldFormer-SSM) packet encoder: a compact Transformer
encoder trained with a masked-token prediction (MLM) objective directly on
the raw byte-hex token stream -- no decoder, no protocol dissector needed to
build training targets (self-supervised purely on bytes the model already
has). This is deliberately immune to the representation-collapse failure
mode found in the source method's reproduction (see REPRODUCTION_REPORT.md):
there is no autoregressive decoder for the encoder to be bypassed by, since
prediction happens directly from each masked position's own encoder output.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parent.parent
PACKETS_PARQUET = ROOT / "data_audit" / "modbus_packets.parquet"
WINDOWS_PARQUET = ROOT / "data_audit" / "modbus_windows.parquet"
TOKENIZER_PATH = ROOT / "data_audit" / "shared_tokenizer.json"
MAX_LEN = 320
MASK_PROB = 0.15


def bytes_to_hex_tokens(hexstr):
    return " ".join(f"<0x{hexstr[i:i+2].upper()}>" for i in range(0, len(hexstr), 2))


def load_tokenizer():
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", bos_token="<s>", eos_token="</s>", unk_token="<unk>")


class MLMDataset(Dataset):
    def __init__(self, df, tok, ids):
        self.df = df[df["pkt_id"].isin(ids)].reset_index(drop=True)
        self.tok = tok

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        ids = self.tok(bytes_to_hex_tokens(r["raw_bytes_hex"]), truncation=True, max_length=MAX_LEN)["input_ids"]
        return ids


def collate(batch, pad_id, mask_id, vocab_size, mlm_prob=MASK_PROB):
    max_len = max(len(x) for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, ids in enumerate(batch):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        attn[i, :len(ids)] = 1
    mask_arr = (torch.rand(input_ids.shape) < mlm_prob) & (attn.bool())
    labels[mask_arr] = input_ids[mask_arr]
    rand = torch.rand(input_ids.shape)
    replace_mask = mask_arr & (rand < 0.8)
    random_mask = mask_arr & (rand >= 0.8) & (rand < 0.9)
    input_ids[replace_mask] = mask_id
    input_ids[random_mask] = torch.randint(0, vocab_size, (random_mask.sum(),))
    return input_ids, attn, labels


class FieldFormerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=2, n_heads=4, max_len=MAX_LEN, pad_id=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_embed = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                            dropout=0.1, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.mlm_head = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, input_ids, attention_mask):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.embed(input_ids) + self.pos_embed(pos)
        key_padding_mask = ~attention_mask.bool()
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        logits = self.mlm_head(h)
        return h, logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out_tag", default="fieldformer")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer()
    vocab_size = tok.vocab_size
    mask_id = tok.convert_tokens_to_ids("<unk>")  # reuse <unk> slot as [MASK] substitute token id space is shared

    df = pd.read_parquet(PACKETS_PARQUET)
    windows = pd.read_parquet(WINDOWS_PARQUET)

    def pkt_ids_for_split(split):
        ids = set()
        for s in windows.loc[windows["split"] == split, "pkt_ids"]:
            ids.update(int(x) for x in s.split(","))
        return ids

    train_ids = pkt_ids_for_split("train")
    train_ds = MLMDataset(df, tok, train_ids)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                         collate_fn=lambda b: collate(b, tok.pad_token_id, mask_id, vocab_size))

    model = FieldFormerEncoder(vocab_size, args.d_model, args.n_layers, pad_id=tok.pad_token_id).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"FieldFormer encoder params: {n_params/1e6:.3f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss(ignore_index=-100)

    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for input_ids, attn, labels in loader:
            input_ids, attn, labels = input_ids.to(device), attn.to(device), labels.to(device)
            opt.zero_grad()
            _, logits = model(input_ids, attn)
            loss = crit(logits.view(-1, vocab_size), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"epoch {epoch}: mlm_loss={total_loss/n_batches:.4f}")
    train_time = time.time() - t0

    out_dir = ROOT / "candidates" / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "d_model": args.d_model, "n_layers": args.n_layers,
                "vocab_size": vocab_size, "pad_id": tok.pad_token_id}, out_dir / "encoder.pt")
    with open(out_dir / "train_info.json", "w") as f:
        json.dump({"n_params": n_params, "train_seconds": train_time, "final_mlm_loss": total_loss / n_batches}, f, indent=2)
    print(f"Saved -> {out_dir}")

    # quick collapse sanity check on 4 random distinct packets
    model.eval()
    sample = df.sample(4, random_state=1)
    texts = [bytes_to_hex_tokens(x) for x in sample["raw_bytes_hex"]]
    enc = tok(texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        h, _ = model(enc["input_ids"], enc["attention_mask"])
    print("pos0 norms:", h[:, 0].norm(dim=-1).cpu().numpy())
    print("pairwise diff sample0 vs sample1 pos0:", (h[0, 0] - h[1, 0]).norm().item())


if __name__ == "__main__":
    main()
