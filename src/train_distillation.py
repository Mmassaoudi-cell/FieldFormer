"""Encoder-focused knowledge distillation, matching the source paper's Eqs.
3-4: teacher encoder frozen (half precision); student encoder + a learnable
head-lifting projection W_H are trained against a hidden-state MSE + a
head-averaged attention-map MSE at each mapped layer pair
g(m) = (L_t/L_s)*m; the decoder is NOT distilled -- it is trained from
scratch against the original generation cross-entropy loss on the same
parsed-protocol-text targets. Overall loss L_KD = alpha*L_gen + L_enc, alpha=1.
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast, BartConfig, BartForConditionalGeneration
import sacrebleu
from rouge_score import rouge_scorer
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PACKETS_PARQUET = ROOT / "data_audit" / "modbus_packets.parquet"
WINDOWS_PARQUET = ROOT / "data_audit" / "modbus_windows.parquet"
TOKENIZER_PATH = ROOT / "data_audit" / "shared_tokenizer.json"
MAX_SRC_LEN, MAX_TGT_LEN = 320, 256


def bytes_to_hex_tokens(hexstr):
    return " ".join(f"<0x{hexstr[i:i+2].upper()}>" for i in range(0, len(hexstr), 2))


def load_tokenizer():
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", bos_token="<s>", eos_token="</s>", unk_token="<unk>")


class PacketDS(Dataset):
    def __init__(self, df, tok, ids):
        self.df = df[df["pkt_id"].isin(ids)].reset_index(drop=True)
        self.tok = tok

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        src = self.tok(bytes_to_hex_tokens(r["raw_bytes_hex"]), truncation=True, max_length=MAX_SRC_LEN)
        tgt = self.tok(r["dissect_text"], truncation=True, max_length=MAX_TGT_LEN)
        return src["input_ids"], tgt["input_ids"]


def collate(batch, pad_id):
    srcs, tgts = zip(*batch)
    max_s = max(len(s) for s in srcs)
    max_t = max(len(t) for t in tgts)
    src_ids = torch.full((len(batch), max_s), pad_id, dtype=torch.long)
    src_mask = torch.zeros((len(batch), max_s), dtype=torch.long)
    tgt_ids = torch.full((len(batch), max_t), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_t), -100, dtype=torch.long)
    for i, (s, t) in enumerate(batch):
        src_ids[i, :len(s)] = torch.tensor(s)
        src_mask[i, :len(s)] = 1
        tgt_ids[i, :len(t)] = torch.tensor(t)
        labels[i, :len(t)] = torch.tensor(t)
    return src_ids, src_mask, tgt_ids, labels


def pkt_ids_for_split(w, split):
    ids = set()
    for s in w.loc[w["split"] == split, "pkt_ids"]:
        ids.update(int(x) for x in s.split(","))
    return ids


def build_student_config(vocab_size, tok, d=256, layers=2):
    return BartConfig(
        vocab_size=vocab_size, d_model=d, encoder_layers=layers, decoder_layers=layers,
        encoder_attention_heads=max(1, d // 64), decoder_attention_heads=max(1, d // 64),
        encoder_ffn_dim=4 * d, decoder_ffn_dim=4 * d, max_position_embeddings=512,
        pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        decoder_start_token_id=tok.bos_token_id,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_dir", required=True)
    ap.add_argument("--student_dim", type=int, default=256)
    ap.add_argument("--student_layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out_tag", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer()
    teacher = BartForConditionalGeneration.from_pretrained(args.teacher_dir).to(device).half().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student_cfg = build_student_config(tok.vocab_size, tok, args.student_dim, args.student_layers)
    student = BartForConditionalGeneration(student_cfg).to(device)
    n_student = sum(p.numel() for p in student.parameters())
    n_teacher = sum(p.numel() for p in teacher.parameters())
    print(f"teacher={n_teacher/1e6:.2f}M student={n_student/1e6:.2f}M ratio={n_teacher/n_student:.1f}x")

    d_teacher, d_student = teacher.config.d_model, student_cfg.d_model
    W_H = nn.Linear(d_student, d_teacher, bias=False).to(device)

    L_t, L_s = teacher.config.encoder_layers, student_cfg.encoder_layers
    layer_map = {m: min(L_t - 1, int(round((L_t / L_s) * (m + 1))) - 1) for m in range(L_s)}
    print("layer map (student idx -> teacher idx):", layer_map)

    df = pd.read_parquet(PACKETS_PARQUET)
    windows = pd.read_parquet(WINDOWS_PARQUET)
    train_ids, val_ids, test_ids = (pkt_ids_for_split(windows, s) for s in ("train", "val", "test"))
    train_ds, val_ds, test_ds = PacketDS(df, tok, train_ids), PacketDS(df, tok, val_ids), PacketDS(df, tok, test_ids)
    pad_id = tok.pad_token_id
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate(b, pad_id))

    opt = torch.optim.AdamW(list(student.parameters()) + list(W_H.parameters()), lr=args.lr)
    mse = nn.MSELoss()

    out_dir = ROOT / "SOURCE_METHOD_REPRODUCTION" / f"bart_{args.out_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for epoch in range(args.epochs):
        student.train()
        total_gen, total_enc = 0.0, 0.0
        for step, (src_ids, src_mask, dec_in, labels) in enumerate(train_loader):
            src_ids, src_mask, labels = src_ids.to(device), src_mask.to(device), labels.to(device)
            opt.zero_grad()

            student_out = student(input_ids=src_ids, attention_mask=src_mask, labels=labels,
                                   output_hidden_states=True, output_attentions=True)
            L_gen = student_out.loss

            with torch.no_grad():
                teacher_enc = teacher.get_encoder()(input_ids=src_ids, attention_mask=src_mask,
                                                     output_hidden_states=True, output_attentions=True)

            student_enc_hidden = student_out.encoder_hidden_states  # tuple len L_s+1
            student_enc_attn = student_out.encoder_attentions
            teacher_enc_hidden = teacher_enc.hidden_states
            teacher_enc_attn = teacher_enc.attentions

            L_enc = 0.0
            for m in range(L_s):
                t_idx = layer_map[m] + 1  # +1: hidden_states[0] is embeddings
                h_s = student_enc_hidden[m + 1].float()
                h_t = teacher_enc_hidden[t_idx].float()
                L_enc = L_enc + mse(W_H(h_s), h_t)

                a_s = student_enc_attn[m].float().mean(dim=1)  # head-averaged, (B,T,T)
                a_t = teacher_enc_attn[layer_map[m]].float().mean(dim=1)
                if a_s.shape == a_t.shape:
                    L_enc = L_enc + mse(a_s, a_t)

            loss = args.alpha * L_gen + L_enc
            loss.backward()
            opt.step()
            total_gen += L_gen.item()
            total_enc += float(L_enc)
            if step % 200 == 0:
                print(f"epoch {epoch} step {step}: L_gen={L_gen.item():.4f} L_enc={float(L_enc):.4f}")
        print(f"epoch {epoch} done: mean L_gen={total_gen/len(train_loader):.4f} mean L_enc={total_enc/len(train_loader):.4f}")

    train_time = time.time() - t0

    # translation-quality eval on test (student full model, generation)
    student.eval()
    n_eval = min(300, len(test_ds))
    preds, refs = [], []
    with torch.no_grad():
        for i in range(0, n_eval, 16):
            batch = [test_ds[j] for j in range(i, min(i + 16, n_eval))]
            srcs = [b[0] for b in batch]
            tgts = [b[1] for b in batch]
            max_len = max(len(s) for s in srcs)
            input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
            attn = torch.zeros((len(batch), max_len), dtype=torch.long)
            for k, s in enumerate(srcs):
                input_ids[k, :len(s)] = torch.tensor(s)
                attn[k, :len(s)] = 1
            input_ids, attn = input_ids.to(device), attn.to(device)
            gen = student.generate(input_ids=input_ids, attention_mask=attn, max_new_tokens=MAX_TGT_LEN, num_beams=1)
            for k in range(len(batch)):
                preds.append(tok.decode(gen[k], skip_special_tokens=True))
                refs.append(tok.decode(tgts[k], skip_special_tokens=True))

    bleu = sacrebleu.corpus_bleu(preds, [refs]).score / 100.0
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    r1, r2, rl = [], [], []
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        r1.append(s["rouge1"].fmeasure); r2.append(s["rouge2"].fmeasure); rl.append(s["rougeL"].fmeasure)

    metrics = {
        "bleu": bleu, "rouge1": float(np.mean(r1)), "rouge2": float(np.mean(r2)), "rougeL": float(np.mean(rl)),
        "n_eval": len(preds), "train_seconds": train_time, "n_params_student": n_student,
        "n_params_teacher": n_teacher, "compression_ratio": n_teacher / n_student,
        "student_dim": args.student_dim, "student_layers": args.student_layers,
    }
    student.save_pretrained(out_dir / "final_model")
    tok.save_pretrained(out_dir / "final_model")
    with open(out_dir / "translation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
