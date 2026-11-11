#!/usr/bin/env python3
"""The three acceptance tests of stage 0 of the DFlash2 drafter design (docs/DRAFTER-STAGE0.md).

usage: drafter_report.py DIR [--sets FILE] [--drafter-ms 41] [--overhead-ms 3] [--tolerance 0.08]

DIR holds  dumps/<name>/   (ricursive_dump_taps output: meta.json, ids.i32, taps.f32)
           oracle/<name>/  (drafter_oracle.py run output: oracle.npz)
FILE (default DIR/sets.tsv) has one prompt per line: NAME<TAB>SET<TAB>REF, SET is `long` or `short`, REF is
`ref` for the prompt llama.cpp was measured on (long: 141 of 228 drafted tokens accepted, 0.618; short:
51 of 116, 0.44, both at 2 draft tokens) or `-`.

Replay. At the anchor (the last committed token) the drafter proposes d tokens, the target accepts the
longest prefix equal to its own greedy continuation (the dumped ids), and the next anchor is a+1 positions
later: the walk a real speculative run would take. Per-anchor oracle output (stride 1) is enough to replay
it for every d. tau(d) = 1 + mean accepted per step.

  1. Oracle wiring: variant L at d = 2 reproduces llama.cpp's acceptance on its two prompts within the
     tolerance. If not, taps, oracle or tap layer are wrong: stop and fix.
  2. Exactness precondition: every logits row of the verify-row passes hashes equal to the 1-row decode's
     at the same position, on every dump.
  3. Go/no-go: with the measured recipe variant R, tok/s = tau(d) / (T(d+1) + D + O) per set; go when the
     best d reaches 1.10 x llama.cpp on both sets (11.3 long, 9.6 short).

T(k): the gufo pass time for k rows, ms, from DRAFTER-GUFO-DESIGN.md section 5 (measured 1/2/4/8/16-row
points, linear between): T(2)=197.5, T(4)=213.3, T(8)=294.3, T(16)=584.6. D is the drafter step time
(41 ms pessimistic, 20 ms estimate), O the logits copy and argmax (3 ms, estimate).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

T_POINTS = {2: 197.5, 4: 213.3, 8: 294.3, 16: 584.6}
LLAMA = {"long": {"rate": 141 / 228, "tps": 10.25}, "short": {"rate": 51 / 116, "tps": 8.71}}
GO_FACTOR = 1.10


def t_ms(k: int) -> float:
    xs = sorted(T_POINTS)
    if k <= xs[0]:
        return T_POINTS[xs[0]]
    for lo, hi in zip(xs, xs[1:]):
        if lo <= k <= hi:
            return T_POINTS[lo] + (T_POINTS[hi] - T_POINTS[lo]) * (k - lo) / (hi - lo)
    raise ValueError(k)


def replay(anchors: np.ndarray, acc: np.ndarray, d: int) -> tuple[int, int, bool]:
    """(steps, accepted, exact) for the walk from the first anchor. exact = every anchor of the walk existed."""
    index = {int(s): i for i, s in enumerate(anchors)}
    s, steps, accepted = int(anchors[0]), 0, 0
    complete = len(anchors) == anchors[-1] - anchors[0] + 1
    while s in index:
        a = int(acc[index[s], d - 1])
        if a < 0:  # fewer than d continuation tokens left, or d not computed
            break
        steps += 1
        accepted += a
        s += a + 1
    return steps, accepted, complete


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--sets")
    ap.add_argument("--drafter-ms", type=float, default=41.0)
    ap.add_argument("--overhead-ms", type=float, default=3.0)
    ap.add_argument("--tolerance", type=float, default=0.08)
    ap.add_argument("--wire-only", action="store_true",
                    help="plumbing check on a random model: print the numbers, apply no thresholds")
    args = ap.parse_args()
    root = Path(args.dir)
    sets_path = Path(args.sets) if args.sets else root / "sets.tsv"
    prompts: list[tuple[str, str, bool]] = []
    for line in sets_path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, st, *ref = line.split("\t")
        prompts.append((name, st, bool(ref) and ref[0] == "ref"))

    meta, oracle = {}, {}
    for name, _, _ in prompts:
        meta[name] = json.loads((root / "dumps" / name / "meta.json").read_text())
        oracle[name] = np.load(root / "oracle" / name / "oracle.npz")

    print(f"prompts: {len(prompts)}; {sum(m['generated'] for m in meta.values())} generated tokens")
    verdicts = []

    # ---- test 2 ------------------------------------------------------------------------
    eq = sum(m["exact_equal"] for m in meta.values())
    cmp_ = sum(m["exact_compared"] for m in meta.values())
    prefill_bad = [n for n, m in meta.items() if m["prefill_hash_a"] != m["prefill_hash_b"]]
    verify_rows = {m["verify_rows"] for m in meta.values()}
    print(f"\nTEST 2  exactness precondition: {eq} of {cmp_} logits rows equal between {sorted(verify_rows)}-row "
          f"passes and 1-row passes; prompt-pass rows differing: {prefill_bad or 'none'}")
    if eq != cmp_:
        bad = [(n, m["exact_compared"] - m["exact_equal"]) for n, m in meta.items() if m["exact_equal"] != m["exact_compared"]]
        print(f"        differing rows per prompt: {bad}")
    ok2 = eq == cmp_ and not prefill_bad and cmp_ > 0
    verdicts.append(("2 exactness precondition", ok2))

    # ---- test 1 ------------------------------------------------------------------------
    print("\nTEST 1  oracle wiring: variant L at d = 2, acceptance = accepted / drafted over the replayed walk")
    ok1 = True
    pooled: dict[str, list[int]] = {}
    any_ref = False
    for name, st, ref in prompts:
        o = oracle[name]
        if "L_acc" not in o.files:
            print(f"        {name}: no L variant in the oracle output")
            continue
        steps, accepted, complete = replay(o["anchors"], o["L_acc"], 2)
        rate = accepted / (2 * steps) if steps else float("nan")
        p = pooled.setdefault(st, [0, 0])
        p[0] += steps
        p[1] += accepted
        line = f"        {name:28s} set {st:5s} steps {steps:4d} accepted {accepted:4d} rate {rate:.3f}"
        if ref:
            any_ref = True
            want = LLAMA[st]["rate"]
            good = abs(rate - want) <= args.tolerance
            ok1 &= good
            line += f"   llama.cpp {want:.3f}  diff {rate - want:+.3f}  {'within' if good else 'OUTSIDE'} +-{args.tolerance}"
        if not complete:
            line += "   (stride > 1: anchors averaged, walk not replayed)"
        print(line)
    for st, (steps, accepted) in pooled.items():
        print(f"        pooled {st:5s}: {accepted}/{2 * steps} = {accepted / (2 * steps) if steps else float('nan'):.3f}")
    ok1 = ok1 and any_ref
    verdicts.append(("1 oracle wiring", ok1 if not args.wire_only else None))

    # ---- test 3 ------------------------------------------------------------------------
    D, O = args.drafter_ms, args.overhead_ms
    print(f"\nTEST 3  go/no-go: tok/s = tau(d) / (T(d+1) + D + O), D = {D:g} ms, O = {O:g} ms; llama.cpp long "
          f"{LLAMA['long']['tps']}, short {LLAMA['short']['tps']} tok/s; go at {GO_FACTOR}x "
          f"({GO_FACTOR * LLAMA['long']['tps']:.1f} / {GO_FACTOR * LLAMA['short']['tps']:.1f})")
    ok3 = True
    for variant in ("R", "Rs", "L"):
        for st in ("long", "short"):
            members = [n for n, s_, _ in prompts if s_ == st and f"{variant}_acc" in oracle[n].files]
            if not members:
                continue
            row, best = [], (0.0, 0)
            for d in range(1, 16):
                steps = accepted = 0
                for n in members:
                    o = oracle[n]
                    a, b, _ = replay(o["anchors"], o[f"{variant}_acc"], d)
                    steps += a
                    accepted += b
                if steps == 0:
                    continue
                tau = 1 + accepted / steps
                tps = tau / ((t_ms(d + 1) + D + O) / 1000)
                row.append((d, tau, tps, accepted / (d * steps)))
                if tps > best[0]:
                    best = (tps, d)
            if not row:
                continue
            print(f"  variant {variant:2s} set {st:5s} ({len(members)} prompts): "
                  + "  ".join(f"d{d}:tau {tau:.2f}/{tps:.2f}" for d, tau, tps, _ in row))
            need = GO_FACTOR * LLAMA[st]["tps"]
            if variant == "R":
                ok = best[0] >= need
                ok3 &= ok
                print(f"      best d = {best[1]}: {best[0]:.2f} tok/s, needs {need:.1f}: {'GO' if ok else 'NO-GO'}"
                      f"   (acceptance per drafted token at d = {best[1]}: "
                      f"{[r[3] for r in row if r[0] == best[1]][0]:.3f})")
    verdicts.append(("3 go/no-go (R, D=%g ms)" % D, ok3 if not args.wire_only else None))

    print("\nSUMMARY")
    for label, ok in verdicts:
        print(f"  test {label}: {'not judged (--wire-only)' if ok is None else 'PASS' if ok else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
