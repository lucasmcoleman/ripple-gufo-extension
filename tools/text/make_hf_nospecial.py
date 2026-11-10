#!/usr/bin/env python3
"""Hugging Face oracle for parse_special=false: the real tokenizer.json with the
CONTROL-type added tokens removed (only the three USER_DEFINED |||...||| tokens stay
matched, as in llama.cpp). Validated against the goldens' ids_no_parse_special.
usage: make_hf_nospecial.py tokenizer.json OUT.json"""
import json, sys
j = json.load(open(sys.argv[1]))
j["added_tokens"] = [a for a in j["added_tokens"] if a["id"] in (100261, 100262, 100263)]
json.dump(j, open(sys.argv[2], "w"), ensure_ascii=False)
