#!/usr/bin/env python3
"""Every Unicode scalar value (surrogates excluded) in probe contexts, as hex lines.
usage: gen_cp_probes.py OUT.hex [CONTEXTS=4|8]   -> CONTEXTS * 1,112,064 lines"""
import sys
ctx = [lambda c: "a" + c + "a", lambda c: "1" + c + "1", lambda c: " " + c, lambda c: c + "\n",
       lambda c: c + "'d", lambda c: c + "123", lambda c: "!" + c + "a", lambda c: "\t" + c + "\t"]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
with open(sys.argv[1], "w") as f:
    for cp in list(range(0xD800)) + list(range(0xE000, 0x110000)):
        c = chr(cp)
        for k in ctx[:n]:
            f.write(k(c).encode("utf-8").hex() + "\n")
