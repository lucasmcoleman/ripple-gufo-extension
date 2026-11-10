#!/usr/bin/env python3
"""Pre-tokenizer pieces of the real tokenizer.json (regex split only, no special tokens,
no BPE) as piece lengths in code points, one line per hex input line.
usage: hf_pieces_batch.py tokenizer.json IN.hex OUT.txt"""
import sys
from tokenizers import Tokenizer
pt = Tokenizer.from_file(sys.argv[1]).pre_tokenizer
with open(sys.argv[2]) as f, open(sys.argv[3], "w") as o:
    for line in f:
        s = bytes.fromhex(line.strip()).decode("utf-8")
        o.write(" ".join(str(b - a) for _, (a, b) in pt.pre_tokenize_str(s)) + "\n")
