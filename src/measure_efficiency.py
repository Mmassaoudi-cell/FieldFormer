"""Section 23 efficiency analysis, matching the source paper's Table VIII
format: per-32-packet-window latency, parameter count, and memory, comparing
the source-method reproduction (BART teacher + Stacked LSTM) against the
final selected model (FieldFormer + BalancedFusion).
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from tokenizers import Tokenizer

from train_fieldformer_encoder import FieldFormerEncoder, bytes_to_hex_tokens
from train_candidate_c import BalancedFusion
from train_lstm_downstream import StackedLSTMClassifier, StackedLSTMAutoencoder

ROOT = Path(__file__).resolve().parent.parent


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_latency_fn(fn, n_warmup=5, n_trials=30):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}

    # --- Source method: BART teacher encoder + Stacked LSTM classifier ---
    bart_dir = ROOT / "SOURCE_METHOD_REPRODUCTION" / "bart_teacher_v2" / "final_model"
    bart = BartForConditionalGeneration.from_pretrained(bart_dir).to(device).eval()
    tok = Tokenizer.from_file(str(bart_dir / "tokenizer.json"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", bos_token="<s>",
                                         eos_token="</s>", unk_token="<unk>")
    lstm_cls = StackedLSTMClassifier(bart.config.d_model, 128, 2, 6).to(device).eval()
    lstm_ae = StackedLSTMAutoencoder(bart.config.d_model, 128, 2).to(device).eval()

    dummy_bytes = bytes(np.random.randint(0, 256, 100, dtype=np.uint8).tobytes())
    text = bytes_to_hex_tokens(dummy_bytes.hex())
    enc = tokenizer([text] * 32, truncation=True, max_length=320, padding=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    def source_forward():
        with torch.no_grad():
            out = bart.get_encoder()(**enc)
            hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)  # (32, D) one embedding per packet
            seq = pooled.unsqueeze(0)  # (1, 32, D) one window
            _ = lstm_cls(seq)
            _ = lstm_ae(seq)

    mean_ms, std_ms = measure_latency_fn(source_forward)
    n_params = count_params(bart) + count_params(lstm_cls) + count_params(lstm_ae)
    results["source_method_reproduction"] = {
        "mean_ms_per_window": mean_ms, "std_ms_per_window": std_ms, "n_params": n_params,
        "n_params_millions": n_params / 1e6,
    }

    # --- Final model: FieldFormer + BalancedFusion ---
    ff_ckpt = torch.load(ROOT / "candidates" / "fieldformer" / "encoder.pt", map_location=device)
    ff = FieldFormerEncoder(ff_ckpt["vocab_size"], ff_ckpt["d_model"], ff_ckpt["n_layers"], pad_id=ff_ckpt["pad_id"]).to(device).eval()
    ff.load_state_dict(ff_ckpt["state_dict"])
    final_model = BalancedFusion(ff_ckpt["d_model"], d_model=192, n_layers=2, n_classes=6).to(device).eval()

    ff_tok = Tokenizer.from_file(str(ROOT / "data_audit" / "shared_tokenizer.json"))
    ff_tokenizer = PreTrainedTokenizerFast(tokenizer_object=ff_tok, pad_token="<pad>", bos_token="<s>",
                                            eos_token="</s>", unk_token="<unk>")
    enc2 = ff_tokenizer([text] * 32, truncation=True, max_length=320, padding=True, return_tensors="pt")
    enc2 = {k: v.to(device) for k, v in enc2.items()}

    def final_forward():
        with torch.no_grad():
            h, _ = ff(enc2["input_ids"], enc2["attention_mask"])
            mask = enc2["attention_mask"].unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            seq = pooled.unsqueeze(0)
            _ = final_model(seq)

    mean_ms2, std_ms2 = measure_latency_fn(final_forward)
    n_params2 = count_params(ff) + count_params(final_model)
    results["final_model"] = {
        "mean_ms_per_window": mean_ms2, "std_ms_per_window": std_ms2, "n_params": n_params2,
        "n_params_millions": n_params2 / 1e6,
    }

    results["speedup_x"] = mean_ms / mean_ms2
    results["compression_x"] = n_params / n_params2
    results["device"] = device

    out_dir = ROOT / "results" / "efficiency"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "efficiency_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
