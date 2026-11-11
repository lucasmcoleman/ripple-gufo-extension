#!/usr/bin/env python3
"""Compares the drafter's drafts on real weights with the stage 0 oracle's (docs/DRAFTER-S2.md, STAGE2-GPU.md).

usage: drafter_compare.py DRAFTS.i32=ORACLE.npz [DRAFTS.i32=ORACLE.npz ...] [--tolerance 0.02] [--variant R]

DRAFTS.i32: int32 [anchors][15], written by `ricursive_drafter_bench --drafts-out F --from A --to B` with A and B
the oracle's first anchor and last anchor + 1 (one block of 15 tokens per anchor of the dump, the context being the
dump's rows below the anchor). ORACLE.npz: the oracle's output for the same dump (variant R: `R_drafts`
[anchors][15], `anchors`, `ids`). For each pair, and pooled over all pairs, prints
  - the fraction of drafted token rows equal to the oracle's, and the number of anchors with all 15 equal;
  - the simulated acceptance (the walk of tools/drafter_report.py: the drafter proposes d tokens at the anchor, the
    target accepts the longest prefix equal to the dumped continuation, the next anchor is a + 1 positions later)
    for d in 2, 3, 4, 6, 15 with these drafts and with the oracle's, and the relative difference of
    tau(d) = 1 + accepted / steps.
Exit 0 when at least 95% of the pooled token rows are equal and every pooled |tau difference| / tau(oracle) <=
tolerance (default 2%). Only numpy; reads files this repository's tools wrote, runs no downloaded code.
"""
import argparse
import sys

import numpy as np

DS = (2, 3, 4, 6, 15)


def replay(anchors, drafts, ids, d):
    """(steps, accepted) of the walk from the first anchor; stops where an anchor is missing or the dump ends."""
    index = {int(s): i for i, s in enumerate(anchors)}
    s, steps, accepted = int(anchors[0]), 0, 0
    while s in index and s + d < len(ids):
        row, n = drafts[index[s]], 0
        while n < d and s + 1 + n < len(ids) and row[n] == ids[s + 1 + n]:
            n += 1
        steps += 1
        accepted += n
        s += n + 1
    return steps, accepted


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="+", help="DRAFTS.i32=ORACLE.npz")
    ap.add_argument("--tolerance", type=float, default=0.02)
    ap.add_argument("--variant", default="R")
    args = ap.parse_args()
    pooled = {d: [0, 0, 0, 0] for d in DS}  # steps, accepted (these drafts), steps, accepted (oracle)
    equal_total = rows_total = 0
    for pair in args.pairs:
        drafts_path, oracle_path = pair.split("=", 1)
        o = np.load(oracle_path)
        anchors, ids = o["anchors"].astype(np.int64), o["ids"].astype(np.int64)
        want = o[f"{args.variant}_drafts"].astype(np.int64)
        got = np.fromfile(drafts_path, dtype=np.int32).astype(np.int64).reshape(-1, 15)
        if len(got) != len(anchors):
            print(f"{drafts_path}: {len(got)} anchors but the oracle has {len(anchors)} ({anchors[0]} .. {anchors[-1]})")
            return 2
        equal = got == want
        equal_total += int(equal.sum())
        rows_total += equal.size
        line = [f"{drafts_path}: {len(got)} anchors, {int(equal.sum())} of {equal.size} token rows equal "
                f"({100 * equal.mean():.2f}%), {int(equal.all(axis=1).sum())} anchors with all 15 equal;"]
        for d in DS:
            sg, ag = replay(anchors, got, ids, d)
            so, ao = replay(anchors, want, ids, d)
            p = pooled[d]
            p[0] += sg
            p[1] += ag
            p[2] += so
            p[3] += ao
            if sg and so:
                line.append(f"tau({d}) {1 + ag / sg:.3f} vs {1 + ao / so:.3f}")
        print(" ".join(line))
    ok = equal_total >= 0.95 * rows_total
    print(f"pooled: {equal_total} of {rows_total} token rows equal ({100 * equal_total / rows_total:.2f}%), need >= 95%")
    worst = 0.0
    for d in DS:
        sg, ag, so, ao = pooled[d]
        if not sg or not so:
            print(f"  d = {d}: no walk (too few anchors)")
            continue
        tg, to = 1 + ag / sg, 1 + ao / so
        diff = abs(tg - to) / to
        worst = max(worst, diff)
        print(f"  d = {d:2d}: tau {tg:.3f} with these drafts ({sg} steps), {to:.3f} with the oracle's ({so} steps), "
              f"relative difference {100 * diff:.2f}%")
    ok = ok and worst <= args.tolerance
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
