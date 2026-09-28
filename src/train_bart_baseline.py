"""Reproduces the source paper's weakest Table III row: pretrained
facebook/bart-base fine-tuned on the SAME byte->text task but using BART's
own default English-oriented BPE tokenizer instead of our custom hex-byte +
protocol-field vocabulary. This is both (a) the "baseline BART [6]" row and
(b) doubles as the "Ours without custom vocab" ablation condition, isolating
what the custom vocabulary actually buys (paper Table III claims this is
substantial: BLEU 0.15->0.59, ROUGE-1 0.42->0.95).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    BartForConditionalGeneration, BartTokenizerFast,
    Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq,
)
import sacrebleu
from rouge_score import rouge_scorer

ROOT = Path(__file__).resolve().parent.parent
PACKETS_PARQUET = ROOT / "data_audit" / "modbus_packets.parquet"
WINDOWS_PARQUET = ROOT / "data_audit" / "modbus_windows.parquet"

MAX_SRC_LEN = 512  # bart-base's default tokenizer will need more tokens per byte (poor fit)
MAX_TGT_LEN = 256


def bytes_to_hex_tokens(hexstr: str) -> str:
    return " ".join(f"0x{hexstr[i:i+2].upper()}" for i in range(0, len(hexstr), 2))


class PacketSeq2SeqDataset(Dataset):
    def __init__(self, df, tokenizer, split_pkt_ids):
        self.df = df[df["pkt_id"].isin(split_pkt_ids)].reset_index(drop=True)
        self.tok = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src = self.tok(bytes_to_hex_tokens(row["raw_bytes_hex"]), truncation=True, max_length=MAX_SRC_LEN)
        tgt = self.tok(text_target=row["dissect_text"], truncation=True, max_length=MAX_TGT_LEN)
        return {"input_ids": src["input_ids"], "attention_mask": src["attention_mask"], "labels": tgt["input_ids"]}


def pkt_ids_for_split(windows_df, split):
    ids = set()
    for s in windows_df.loc[windows_df["split"] == split, "pkt_ids"]:
        ids.update(int(x) for x in s.split(","))
    return ids


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"baseline BART params: {n_params/1e6:.2f}M")

    df = pd.read_parquet(PACKETS_PARQUET)
    windows_df = pd.read_parquet(WINDOWS_PARQUET)
    train_ids = pkt_ids_for_split(windows_df, "train")
    val_ids = pkt_ids_for_split(windows_df, "val")
    test_ids = pkt_ids_for_split(windows_df, "test")

    train_ds = PacketSeq2SeqDataset(df, tokenizer, train_ids)
    val_ds = PacketSeq2SeqDataset(df, tokenizer, val_ids)
    test_ds = PacketSeq2SeqDataset(df, tokenizer, test_ids)
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    out_dir = ROOT / "SOURCE_METHOD_REPRODUCTION" / "bart_baseline_no_custom_vocab"
    out_dir.mkdir(parents=True, exist_ok=True)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100)

    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir / "ckpt"), num_train_epochs=2, per_device_train_batch_size=16,
        per_device_eval_batch_size=16, learning_rate=5e-5, warmup_ratio=0.06, weight_decay=0.01,
        logging_steps=100, eval_strategy="epoch", save_strategy="no", fp16=(device == "cuda"),
        report_to=[], dataloader_num_workers=0,
    )
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, data_collator=collator)

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    model.eval()
    n_eval = min(300, len(test_ds))
    preds, refs = [], []
    with torch.no_grad():
        for i in range(0, n_eval, 16):
            batch = [test_ds[j] for j in range(i, min(i + 16, n_eval))]
            max_len = max(len(b["input_ids"]) for b in batch)
            input_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
            attn = torch.zeros((len(batch), max_len), dtype=torch.long)
            for k, b in enumerate(batch):
                L = len(b["input_ids"])
                input_ids[k, :L] = torch.tensor(b["input_ids"])
                attn[k, :L] = torch.tensor(b["attention_mask"])
            input_ids, attn = input_ids.to(device), attn.to(device)
            gen = model.generate(input_ids=input_ids, attention_mask=attn, max_new_tokens=MAX_TGT_LEN, num_beams=1)
            for k, b in enumerate(batch):
                preds.append(tokenizer.decode(gen[k], skip_special_tokens=True))
                refs.append(tokenizer.decode(b["labels"], skip_special_tokens=True))

    bleu = sacrebleu.corpus_bleu(preds, [refs]).score / 100.0
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    r1, r2, rl = [], [], []
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        r1.append(s["rouge1"].fmeasure); r2.append(s["rouge2"].fmeasure); rl.append(s["rougeL"].fmeasure)

    metrics = {
        "bleu": bleu, "rouge1": float(np.mean(r1)), "rouge2": float(np.mean(r2)), "rougeL": float(np.mean(rl)),
        "n_eval": len(preds), "train_seconds": train_time, "n_params": n_params, "condition": "baseline_bart_no_custom_vocab",
    }
    with open(out_dir / "translation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
