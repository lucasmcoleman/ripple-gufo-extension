#!/usr/bin/env python3
"""Hugging Face `tokenizers` reference for a hex file (one string per line).
usage: hf_tok_batch.py tokenizer.json IN.hex OUT.txt
Output per line: ids with add_special_tokens=False (HF always matches added tokens)."""
import sys
from tokenizers import Tokenizer
t = Tokenizer.from_file(sys.argv[1])
def flush(batch, o):
    for e in t.encode_batch(batch, add_special_tokens=False):
        o.write(" ".join(map(str, e.ids)) + "\n")
with open(sys.argv[2]) as f, open(sys.argv[3], "w") as o:
    batch = []
    for line in f:
        batch.append(bytes.fromhex(line.strip()).decode("utf-8"))
        if len(batch) >= 20000:
            flush(batch, o); batch = []
    if batch: flush(batch, o)
