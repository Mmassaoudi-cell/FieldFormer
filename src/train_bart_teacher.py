"""Train the ICS-Packet-Bart reproduction: byte-code -> structured protocol
text, using our shared custom-vocabulary tokenizer (hex-byte tokens + BPE
over the dissection text), matching the source paper's stated teacher
architecture (8 encoder + 8 decoder layers, d_model=1024).

Real training, real data (Frazao et al. 2019 Modbus/TCP pcaps). Sequence
lengths here are much shorter than the paper's reported 1000-2000 tokens
because our dissection format (Layer:field=value;...) is far more compact
than a full Wireshark dissection tree -- this is itself evidence for
SOURCE_WEAKNESS_ANALYSIS.md item #1 (disproportionate compute cost).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from transformers import (
    PreTrainedTokenizerFast, BartConfig, BartForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq,
)
import sacrebleu
from rouge_score import rouge_scorer

ROOT = Path(__file__).resolve().parent.parent
PACKETS_PARQUET = ROOT / "data_audit" / "modbus_packets.parquet"
WINDOWS_PARQUET = ROOT / "data_audit" / "modbus_windows.parquet"
TOKENIZER_PATH = ROOT / "data_audit" / "shared_tokenizer.json"

MAX_SRC_LEN = 320
MAX_TGT_LEN = 256


def bytes_to_hex_tokens(hexstr: str) -> str:
    return " ".join(f"<0x{hexstr[i:i+2].upper()}>" for i in range(0, len(hexstr), 2))


def load_tokenizer():
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>", bos_token="<s>", eos_token="</s>", unk_token="<unk>",
    )
    return fast


class PacketSeq2SeqDataset(Dataset):
    def __init__(self, df, tokenizer, split_pkt_ids):
        self.df = df[df["pkt_id"].isin(split_pkt_ids)].reset_index(drop=True)
        self.tok = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src_text = bytes_to_hex_tokens(row["raw_bytes_hex"])
        tgt_text = row["dissect_text"]
        src = self.tok(src_text, truncation=True, max_length=MAX_SRC_LEN)
        with self.tok.as_target_tokenizer() if hasattr(self.tok, "as_target_tokenizer") else _nullctx():
            tgt = self.tok(tgt_text, truncation=True, max_length=MAX_TGT_LEN)
        return {
            "input_ids": src["input_ids"],
            "attention_mask": src["attention_mask"],
            "labels": tgt["input_ids"],
        }


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def pkt_ids_for_split(windows_df, split):
    ids = set()
    for pkt_ids_str in windows_df.loc[windows_df["split"] == split, "pkt_ids"]:
        ids.update(int(x) for x in pkt_ids_str.split(","))
    return ids


def build_model(tokenizer, size="teacher"):
    vocab_size = tokenizer.vocab_size
    if size == "teacher":
        cfg = BartConfig(
            vocab_size=vocab_size, d_model=1024, encoder_layers=8, decoder_layers=8,
            encoder_attention_heads=16, decoder_attention_heads=16,
            encoder_ffn_dim=4096, decoder_ffn_dim=4096,
            max_position_embeddings=512,
            pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id, decoder_start_token_id=tokenizer.bos_token_id,
        )
    elif size == "student256":
        cfg = BartConfig(
            vocab_size=vocab_size, d_model=256, encoder_layers=2, decoder_layers=2,
            encoder_attention_heads=4, decoder_attention_heads=4,
            encoder_ffn_dim=1024, decoder_ffn_dim=1024,
            max_position_embeddings=512,
            pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id, decoder_start_token_id=tokenizer.bos_token_id,
        )
    else:
        raise ValueError(size)
    model = BartForConditionalGeneration(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Built {size} model: {n_params/1e6:.2f}M params")
    return model, n_params


def evaluate_generation(model, tokenizer, dataset, device, n_samples=300, batch_size=16):
    model.eval()
    idxs = list(range(min(n_samples, len(dataset))))
    preds, refs = [], []
    with torch.no_grad():
        for i in range(0, len(idxs), batch_size):
            batch_idx = idxs[i:i+batch_size]
            batch = [dataset[j] for j in batch_idx]
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
                pred_text = tokenizer.decode(gen[k], skip_special_tokens=True)
                ref_text = tokenizer.decode(b["labels"], skip_special_tokens=True)
                preds.append(pred_text)
                refs.append(ref_text)
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score / 100.0
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    r1, r2, rl = [], [], []
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    return {
        "bleu": bleu, "rouge1": float(np.mean(r1)), "rouge2": float(np.mean(r2)), "rougeL": float(np.mean(rl)),
        "n_eval": len(preds),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="teacher", choices=["teacher", "student256"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out_tag", default="teacher")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tokenizer = load_tokenizer()
    df = pd.read_parquet(PACKETS_PARQUET)
    windows_df = pd.read_parquet(WINDOWS_PARQUET)

    train_ids = pkt_ids_for_split(windows_df, "train")
    val_ids = pkt_ids_for_split(windows_df, "val")
    test_ids = pkt_ids_for_split(windows_df, "test")

    train_ds = PacketSeq2SeqDataset(df, tokenizer, train_ids)
    val_ds = PacketSeq2SeqDataset(df, tokenizer, val_ids)
    test_ds = PacketSeq2SeqDataset(df, tokenizer, test_ids)
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    model, n_params = build_model(tokenizer, args.size)
    model.to(device)

    out_dir = ROOT / "SOURCE_METHOD_REPRODUCTION" / f"bart_{args.out_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir / "ckpt"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.06,
        weight_decay=0.01,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        fp16=(device == "cuda"),
        report_to=[],
        predict_with_generate=False,
        dataloader_num_workers=0,
    )

    trainer = Seq2SeqTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator,
    )

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    print("Running BLEU/ROUGE evaluation on test set (generation) ...")
    metrics = evaluate_generation(model, tokenizer, test_ds, device)
    metrics["train_seconds"] = train_time
    metrics["n_params"] = n_params
    metrics["size"] = args.size

    model.save_pretrained(out_dir / "final_model")
    tokenizer.save_pretrained(out_dir / "final_model")

    with open(out_dir / "translation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved model + metrics -> {out_dir}")


if __name__ == "__main__":
    main()
