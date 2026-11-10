#!/usr/bin/env python3
"""Score two float32 logits files [T][vocab] (ripple_logits / ricursive_gpu_logits format).

  /server/staging/ripple-gufo/venv/bin/python tools/score_logits.py REF.f32 TEST.f32 --vocab N
         [--max-abs X] [--min-top1 F]

REF is the reference (llama.cpp), TEST the engine under test. KL is KL(REF || TEST) per
position, float64 log-softmax. Prints mean KL, 99th percentile KL, max KL with its
position, top-1 agreement (fraction of positions where the argmax is equal) and the max
abs logit difference with its position. Any NaN/Inf in either file, a size that does not
fit [T][vocab], or different sizes: exit 2 with a loud message and no scores.
Exit 1 when --max-abs / --min-top1 is given and violated, else 0.
"""
import argparse
import sys

import numpy as np


def die(msg):
    print("REFUSED: " + msg, file=sys.stderr)
    sys.exit(2)


def load(path, vocab):
    a = np.memmap(path, dtype="<f4", mode="r")
    if a.size == 0 or a.size % vocab != 0:
        die(f"{path}: {a.size} floats is not a positive multiple of vocab {vocab}")
    return a.reshape(-1, vocab)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--vocab", type=int, required=True, help="logits per position")
    ap.add_argument("--max-abs", type=float, default=None)
    ap.add_argument("--min-top1", type=float, default=None)
    ap.add_argument("--chunk", type=int, default=64, help="positions per block")
    args = ap.parse_args()
    if args.vocab <= 0:
        die("--vocab must be positive")
    r = load(args.ref, args.vocab)
    t = load(args.test, args.vocab)
    if r.shape != t.shape:
        die(f"shape mismatch: {args.ref} has {r.shape[0]} positions, {args.test} has {t.shape[0]}")
    T = r.shape[0]

    kls = np.empty(T)
    top1 = np.empty(T, dtype=bool)
    max_abs, max_abs_at = -1.0, (0, 0)
    for s in range(0, T, args.chunk):
        e = min(T, s + args.chunk)
        a = np.asarray(r[s:e], dtype=np.float64)
        b = np.asarray(t[s:e], dtype=np.float64)
        for name, x in ((args.ref, a), (args.test, b)):
            bad = ~np.isfinite(x)
            if bad.any():
                pos, col = np.argwhere(bad)[0]
                die(f"{name}: {int(bad.sum())} non-finite values (first at position {s + int(pos)}, "
                    f"column {int(col)}); scores would be meaningless")
        d = np.abs(a - b)
        k = np.unravel_index(int(np.argmax(d)), d.shape)
        if d[k] > max_abs:
            max_abs, max_abs_at = float(d[k]), (s + int(k[0]), int(k[1]))
        la = a - a.max(axis=1, keepdims=True)
        la -= np.log(np.exp(la).sum(axis=1, keepdims=True))
        lb = b - b.max(axis=1, keepdims=True)
        lb -= np.log(np.exp(lb).sum(axis=1, keepdims=True))
        kls[s:e] = (np.exp(la) * (la - lb)).sum(axis=1)
        top1[s:e] = a.argmax(axis=1) == b.argmax(axis=1)

    if not np.isfinite(kls).all():
        die("KL is not finite")
    worst = int(np.argmax(kls))
    print(f"positions {T}, vocab {args.vocab}")
    print(f"mean KL        {kls.mean():.6e}")
    print(f"p99 KL         {np.percentile(kls, 99):.6e}")
    print(f"max KL         {kls[worst]:.6e} at position {worst}")
    print(f"top-1 agree    {top1.mean() * 100:.4f}% ({int(top1.sum())}/{T})")
    print(f"max abs diff   {max_abs:.6e} at position {max_abs_at[0]}, column {max_abs_at[1]}")
    failed = False
    if args.max_abs is not None and not (max_abs <= args.max_abs):
        print(f"FAIL: max abs {max_abs:.3e} exceeds {args.max_abs:.3e}")
        failed = True
    if args.min_top1 is not None and not (top1.mean() >= args.min_top1):
        print(f"FAIL: top-1 {top1.mean():.4f} below {args.min_top1:.4f}")
        failed = True
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
