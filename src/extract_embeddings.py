"""Run a trained BART (teacher or student) encoder over every packet to
produce the per-packet semantic embedding the source paper's downstream
Stacked-LSTM models consume (mean-pooled encoder last-hidden-state, since
BART has no dedicated pooler)."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
PACKETS_PARQUET = ROOT / "data_audit" / "modbus_packets.parquet"


def bytes_to_hex_tokens(hexstr: str) -> str:
    return " ".join(f"<0x{hexstr[i:i+2].upper()}>" for i in range(0, len(hexstr), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out_tag", required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--max_len", type=int, default=320)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BartForConditionalGeneration.from_pretrained(args.model_dir).to(device).eval()
    tok = Tokenizer.from_file(str(Path(args.model_dir) / "tokenizer.json"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", bos_token="<s>",
                                         eos_token="</s>", unk_token="<unk>")

    df = pd.read_parquet(PACKETS_PARQUET, columns=["pkt_id", "raw_bytes_hex"])
    n = len(df)
    d_model = model.config.d_model
    out = np.zeros((n, d_model), dtype=np.float32)
    pkt_ids = df["pkt_id"].to_numpy()

    with torch.no_grad():
        for i in range(0, n, args.batch_size):
            batch = df.iloc[i:i + args.batch_size]
            texts = [bytes_to_hex_tokens(h) for h in batch["raw_bytes_hex"]]
            enc = tokenizer(texts, truncation=True, max_length=args.max_len, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            encoder_out = model.get_encoder()(**enc)
            hidden = encoder_out.last_hidden_state  # (B, T, D)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            out[i:i + len(batch)] = pooled.cpu().numpy()
            if i % (args.batch_size * 50) == 0:
                print(f"  {i}/{n}")

    out_dir = ROOT / "data_audit"
    np.save(out_dir / f"embeddings_{args.out_tag}.npy", out)
    np.save(out_dir / f"embeddings_{args.out_tag}_pktids.npy", pkt_ids)
    print(f"Saved {out.shape} embeddings -> embeddings_{args.out_tag}.npy")


if __name__ == "__main__":
    main()
