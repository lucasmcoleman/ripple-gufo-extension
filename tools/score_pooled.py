#!/usr/bin/env python3
"""Pooled scorer for the S3 real-weights test: two directories of per-text float32 logits files.

  /server/staging/ripple-gufo/venv/bin/python tools/score_pooled.py --vocab N \
      --ref-dir llama-dir --test-dir gufo-dir \
      [--floor-ref-dir llama-run-A --floor-test-dir llama-run-B] [--tokens-dir DIR] \
      [--worst 10] [--decades 1.0] [--jobs 8] [--tsv OUT.tsv] [--names-file F]

Files <name>.f32 ([T][vocab] float32, the ripple_logits / ricursive_gpu_logits format) are matched by name
between the two directories; a name present on one side only is a refusal. KL is KL(REF || TEST) per
position (float64 log-softmax), like tools/score_logits.py (whose CLI is untouched). Prints, per text and
pooled over all positions: mean KL, 99th percentile KL, max KL (text, position), top-1 agreement.

With --floor-ref-dir/--floor-test-dir (the SAME texts run twice through llama.cpp at two batch sizes) the
floor's pooled numbers are computed and an explicit verdict is printed. The rule (spec S3: "within the
measured floor's order of magnitude and no outlier position unexplained"), with D = --decades (default 1):
  mean:      gufo mean KL <= floor mean KL x 10^D
  p99:       gufo p99 KL  <= floor p99 KL  x 10^D
  top-1:     gufo top-1 disagreement <= max(floor disagreement, 1/positions) x 10^D
  outliers:  count of gufo positions with KL > floor max KL x 10^D   (these are listed; "unexplained" is a
             human call, so any outlier makes the verdict REVIEW, never PASS)
Floors below 1e-9 are raised to 1e-9 (KL that small is float32 noise; printed when it applies).
Verdict: FAIL if mean, p99 or top-1 is outside; else REVIEW if outliers > 0; else PASS.
The 10 worst gufo positions (--worst) are listed with text name, position, KL, input token, both argmax ids.
Exit: 0 PASS, 3 REVIEW, 1 FAIL, 2 refusal (missing/mismatched files, non-finite values, bad sizes).
Without a floor: numbers only, verdict NO-FLOOR, exit 0.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

EPS_FLOOR = 1e-9


def die(msg):
    print("REFUSED: " + msg, file=sys.stderr)
    sys.exit(2)


def load(path, vocab):
    a = np.memmap(path, dtype="<f4", mode="r")
    if a.size == 0 or a.size % vocab != 0:
        die(f"{path}: {a.size} floats is not a positive multiple of vocab {vocab}")
    return a.reshape(-1, vocab)


def compare_one(args):
    """(ref_path, test_path, vocab) -> (kl[T], top1_equal[T], ref_argmax[T], test_argmax[T], max_abs)"""
    ref_path, test_path, vocab = args
    r = load(ref_path, vocab)
    t = load(test_path, vocab)
    if r.shape != t.shape:
        die(f"shape mismatch: {ref_path} has {r.shape[0]} positions, {test_path} has {t.shape[0]}")
    T = r.shape[0]
    kl = np.empty(T)
    ra = np.empty(T, dtype=np.int64)
    ta = np.empty(T, dtype=np.int64)
    max_abs = 0.0
    for s in range(0, T, 32):
        e = min(T, s + 32)
        a = np.asarray(r[s:e], dtype=np.float64)
        b = np.asarray(t[s:e], dtype=np.float64)
        for name, x in ((ref_path, a), (test_path, b)):
            if not np.isfinite(x).all():
                die(f"{name}: non-finite values near position {s}; scores would be meaningless")
        max_abs = max(max_abs, float(np.abs(a - b).max()))
        la = a - a.max(axis=1, keepdims=True)
        la -= np.log(np.exp(la).sum(axis=1, keepdims=True))
        lb = b - b.max(axis=1, keepdims=True)
        lb -= np.log(np.exp(lb).sum(axis=1, keepdims=True))
        kl[s:e] = (np.exp(la) * (la - lb)).sum(axis=1)
        ra[s:e] = a.argmax(axis=1)
        ta[s:e] = b.argmax(axis=1)
    return kl, ra == ta, ra, ta, max_abs


def names_in(d):
    if not os.path.isdir(d):
        die(f"not a directory: {d}")
    return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".f32"))


def compare_dirs(ref_dir, test_dir, vocab, jobs, only=None):
    a, b = names_in(ref_dir), names_in(test_dir)
    if only is not None:
        a = [n for n in a if n in only]
        b = [n for n in b if n in only]
        missing = sorted(only - set(a))[:5]
        if missing:
            die(f"{ref_dir}: missing files for listed names, e.g. {missing}")
    if not a:
        die(f"no .f32 files in {ref_dir}")
    if a != b:
        die(f"file sets differ: only in {ref_dir}: {sorted(set(a) - set(b))[:5]}; "
            f"only in {test_dir}: {sorted(set(b) - set(a))[:5]}")
    work = [(os.path.join(ref_dir, n + ".f32"), os.path.join(test_dir, n + ".f32"), vocab) for n in a]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        res = list(ex.map(compare_one, work))
    return a, res


def pooled(names, res):
    kl = np.concatenate([r[0] for r in res])
    top1 = np.concatenate([r[1] for r in res])
    owner = np.concatenate([np.full(len(r[0]), i) for i, r in enumerate(res)])
    pos = np.concatenate([np.arange(len(r[0])) for r in res])
    w = int(np.argmax(kl))
    return {
        "n": kl.size, "mean": float(kl.mean()), "p99": float(np.percentile(kl, 99)), "max": float(kl[w]),
        "max_name": names[owner[w]], "max_pos": int(pos[w]), "top1": float(top1.mean()),
        "disagree": float(1.0 - top1.mean()), "kl": kl, "owner": owner, "pos": pos,
        "max_abs": max(r[4] for r in res),
    }


def per_text_table(names, res, tsv):
    rows = []
    for n, r in zip(names, res):
        kl = r[0]
        w = int(np.argmax(kl))
        rows.append((n, kl.size, float(kl.mean()), float(np.percentile(kl, 99)), float(kl[w]), w,
                     float(r[1].mean()), r[4]))
    print(f"{'text':36s} {'T':>5s} {'meanKL':>11s} {'p99KL':>11s} {'maxKL':>11s} {'@pos':>5s} {'top1%':>8s} {'maxabs':>9s}")
    for x in rows:
        print(f"{x[0]:36s} {x[1]:5d} {x[2]:11.3e} {x[3]:11.3e} {x[4]:11.3e} {x[5]:5d} {x[6] * 100:8.3f} {x[7]:9.2e}")
    if tsv:
        with open(tsv, "w") as f:
            f.write("text\tT\tmean_kl\tp99_kl\tmax_kl\tmax_pos\ttop1\tmax_abs\n")
            for x in rows:
                f.write("\t".join(str(v) for v in x) + "\n")


def show(label, p):
    print(f"{label}: positions {p['n']}  mean KL {p['mean']:.4e}  p99 KL {p['p99']:.4e}  "
          f"max KL {p['max']:.4e} ({p['max_name']} pos {p['max_pos']})  top-1 {p['top1'] * 100:.4f}%  "
          f"max abs logit diff {p['max_abs']:.3e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocab", type=int, required=True)
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--floor-ref-dir")
    ap.add_argument("--floor-test-dir")
    ap.add_argument("--tokens-dir", help="directory of <name>.i32 token files, to print the input token at worst positions")
    ap.add_argument("--worst", type=int, default=10)
    ap.add_argument("--decades", type=float, default=1.0)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--tsv", help="write the per-text table here")
    ap.add_argument("--names-file", help="only these text names (one per line, no extension); other files in the directories are ignored")
    ap.add_argument("--label", default="test vs ref")
    args = ap.parse_args()
    if args.vocab <= 0:
        die("--vocab must be positive")
    if bool(args.floor_ref_dir) != bool(args.floor_test_dir):
        die("give both --floor-ref-dir and --floor-test-dir, or neither")

    only = None
    if args.names_file:
        only = set(l.strip() for l in open(args.names_file) if l.strip() and not l.startswith("#"))
    names, res = compare_dirs(args.ref_dir, args.test_dir, args.vocab, args.jobs, only)
    print(f"== {args.label}: {len(names)} texts, vocab {args.vocab}")
    per_text_table(names, res, args.tsv)
    g = pooled(names, res)
    show("POOLED", g)

    floor = None
    if args.floor_ref_dir:
        fn, fres = compare_dirs(args.floor_ref_dir, args.floor_test_dir, args.vocab, args.jobs, only)
        if fn != names:
            die("floor texts differ from the compared texts")
        floor = pooled(fn, fres)
        show("FLOOR ", floor)

    worst = np.argsort(-g["kl"])[: args.worst]
    print(f"-- {len(worst)} worst positions ({args.label})")
    print(f"{'rank':>4s} {'text':36s} {'pos':>5s} {'KL':>11s} {'in-tok':>7s} {'ref-top1':>8s} {'test-top1':>9s}"
          + ("  floor KL at this position" if floor else ""))
    for k, w in enumerate(worst):
        i, p = int(g["owner"][w]), int(g["pos"][w])
        tok = "-"
        if args.tokens_dir:
            tp = os.path.join(args.tokens_dir, names[i] + ".i32")
            if os.path.exists(tp):
                tok = str(int(np.fromfile(tp, dtype="<i4")[p]))
        extra = f"  {floor['kl'][w]:.3e}" if floor else ""
        print(f"{k + 1:4d} {names[i]:36s} {p:5d} {g['kl'][w]:11.3e} {tok:>7s} {int(res[i][2][p]):8d} {int(res[i][3][p]):9d}{extra}")

    if floor is None:
        print("VERDICT: NO-FLOOR (numbers only)")
        return 0

    scale = 10.0 ** args.decades
    note = []

    def floorval(x):
        if x < EPS_FLOOR:
            note.append(f"floor value {x:.2e} raised to {EPS_FLOOR:.0e}")
            return EPS_FLOOR
        return x

    if floor["mean"] == 0.0:
        print("   WARNING: the floor is exactly 0 (the two floor runs are bit-identical): the 10x rule then compares against the 1e-9 "
              "raise, not against real noise; the verdict is not meaningful (llama.cpp batches >= 32 are identical, use a 16-row run B)")
    f_mean, f_p99, f_max = floorval(floor["mean"]), floorval(floor["p99"]), floorval(floor["max"])
    f_dis = max(floor["disagree"], 1.0 / g["n"])
    lim_mean, lim_p99, lim_max, lim_dis = f_mean * scale, f_p99 * scale, f_max * scale, f_dis * scale
    checks = [
        ("mean KL", g["mean"], lim_mean, g["mean"] <= lim_mean),
        ("p99 KL", g["p99"], lim_p99, g["p99"] <= lim_p99),
        ("top-1 disagreement", g["disagree"], lim_dis, g["disagree"] <= lim_dis),
    ]
    outliers = int((g["kl"] > lim_max).sum())
    print(f"-- verdict rule: within floor x 10^{args.decades:g}, outliers = gufo KL > floor max x 10^{args.decades:g}")
    for n_, val, lim, ok in checks:
        print(f"   {n_:20s} gufo {val:.4e}  limit {lim:.4e}  ratio-to-floor {val / (lim / scale):8.2f}  {'ok' if ok else 'OUTSIDE'}")
    print(f"   {'outlier positions':20s} {outliers} above {lim_max:.4e}")
    if outliers:
        keep = g["kl"] <= lim_max
        print(f"   info: mean KL without the {outliers} outlier position(s): {g['kl'][keep].mean():.4e} "
              f"(limit {lim_mean:.4e}) -- shows whether the aggregate is driven by those positions alone")
    for n_ in note:
        print("   note: " + n_)
    if not all(c[3] for c in checks):
        print("VERDICT: FAIL (aggregate outside the floor's order of magnitude)")
        return 1
    if outliers:
        print(f"VERDICT: REVIEW ({outliers} outlier position(s): a human must explain each from the worst-positions list)")
        return 3
    print("VERDICT: PASS (aggregates within the floor's order of magnitude, no outlier position)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
