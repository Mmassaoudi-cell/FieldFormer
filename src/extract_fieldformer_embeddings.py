"""Run a trained FieldFormer encoder (train_fieldformer_encoder.py) over
every packet to produce embeddings for the SSM downstream stage (Candidate
A), mirroring extract_embeddings.py's interface/output format so
train_lstm_downstream.py-style consumers work unchanged."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from train_fieldformer_encoder import FieldFormerEncoder, bytes_to_hex_tokens

ROOT = Path(__file__).resolve().parent.parent
PACKETS_PARQUET = ROOT / "data_audit" / "modbus_packets.parquet"
TOKENIZER_PATH = ROOT / "data_audit" / "shared_tokenizer.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_tag", required=True)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_len", type=int, default=320)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = FieldFormerEncoder(ckpt["vocab_size"], ckpt["d_model"], ckpt["n_layers"], pad_id=ckpt["pad_id"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", bos_token="<s>",
                                         eos_token="</s>", unk_token="<unk>")

    df = pd.read_parquet(PACKETS_PARQUET, columns=["pkt_id", "raw_bytes_hex"])
    n = len(df)
    out = np.zeros((n, ckpt["d_model"]), dtype=np.float32)
    pkt_ids = df["pkt_id"].to_numpy()

    with torch.no_grad():
        for i in range(0, n, args.batch_size):
            batch = df.iloc[i:i + args.batch_size]
            texts = [bytes_to_hex_tokens(h) for h in batch["raw_bytes_hex"]]
            enc = tokenizer(texts, truncation=True, max_length=args.max_len, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            h, _ = model(enc["input_ids"], enc["attention_mask"])
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            out[i:i + len(batch)] = pooled.cpu().numpy()
            if i % (args.batch_size * 50) == 0:
                print(f"  {i}/{n}")

    out_dir = ROOT / "data_audit"
    np.save(out_dir / f"embeddings_{args.out_tag}.npy", out)
    np.save(out_dir / f"embeddings_{args.out_tag}_pktids.npy", pkt_ids)
    print(f"Saved {out.shape} embeddings -> embeddings_{args.out_tag}.npy")


if __name__ == "__main__":
    main()
