"""Train a single shared BPE tokenizer over BOTH the raw-byte source stream
(represented as hex-byte tokens, one per byte) and the structured dissection
target text — mirroring the source paper's "custom vocabulary...hexadecimal
byte values...protocol-field tokens...composite field values" design
(Table II), implemented here with a standard BPE trainer rather than a
hand-curated table.
"""
import json
from pathlib import Path

import pandas as pd
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

ROOT = Path(__file__).resolve().parent.parent
PACKETS_PARQUET = ROOT / "data_audit" / "modbus_packets.parquet"
OUT_TOKENIZER = ROOT / "data_audit" / "shared_tokenizer.json"

VOCAB_SIZE = 4096
SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>"]
HEX_TOKENS = [f"<0x{i:02X}>" for i in range(256)]


def bytes_to_hex_tokens(hexstr: str) -> str:
    # hexstr is bytes.hex() output, 2 chars per byte
    return " ".join(f"<0x{hexstr[i:i+2].upper()}>" for i in range(0, len(hexstr), 2))


def main():
    df = pd.read_parquet(PACKETS_PARQUET, columns=["raw_bytes_hex", "dissect_text", "class"])
    train_texts = df["dissect_text"].tolist()
    # Include a sample of byte-hex streams so BPE can merge common consecutive
    # byte patterns (e.g. repeated Modbus header prefixes) into fewer tokens.
    sample_byte_streams = [bytes_to_hex_tokens(h) for h in df["raw_bytes_hex"].sample(
        n=min(20000, len(df)), random_state=1337).tolist()]

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(),
        pre_tokenizers.Punctuation(),
    ])
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS + HEX_TOKENS,
        min_frequency=2,
    )
    tokenizer.train_from_iterator(train_texts + sample_byte_streams, trainer=trainer)
    tokenizer.decoder = decoders.BPEDecoder()

    tokenizer.save(str(OUT_TOKENIZER))
    vocab = tokenizer.get_vocab()
    print(f"Vocab size: {len(vocab)}")
    print(f"Saved tokenizer -> {OUT_TOKENIZER}")

    # sanity check: hex tokens must be present as single tokens
    ids = tokenizer.encode("<0x01> <0x02> <0xFF>").ids
    print("hex sanity encode ids:", ids, "-> len should be 3:", len(ids) == 3)

    sample = df.iloc[0]
    enc = tokenizer.encode(sample["dissect_text"])
    print("Sample dissect_text:", sample["dissect_text"][:120])
    print("Token count:", len(enc.ids))


if __name__ == "__main__":
    main()
