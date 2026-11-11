"""Throwaway check for S3: KL(ref || test) per position on the 8 x 2048-token set, split at the
1024-token attention window. Pairs: gufo rows 256, gufo rows 16 and llama.cpp batch 16 (the floor),
each against llama.cpp batch 2048. CPU only."""
import os
import sys

import numpy as np

V = 100352
T = 2048
BASE = sys.argv[1]
REF = "2048-llama-hip-kvf16-faoff-b2048"
PAIRS = [
    ("floor: llama b16", "2048-llama-hip-kvf16-faoff-b16"),
    ("gufo rows 256", "2048-gufo-gpu-r256-kvf16"),
    ("gufo rows 16", "2048-gufo-gpu-r16-kvf16"),
]
BUCKETS = [(0, 512), (512, 1024), (1024, 1536), (1536, 2048), (0, 1024), (1024, 2048), (1000, 1024), (1024, 1048)]


def log_softmax(x):
    x = x.astype(np.float64)
    x -= x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


names = sorted(f for f in os.listdir(os.path.join(BASE, REF)) if f.endswith(".f32"))
assert len(names) == 8, names
res = {}
for label, d in PAIRS:
    kl_all = np.zeros((len(names), T))
    agree = np.zeros((len(names), T), dtype=bool)
    for i, n in enumerate(names):
        a = np.memmap(os.path.join(BASE, REF, n), dtype=np.float32, mode="r", shape=(T, V))
        b = np.memmap(os.path.join(BASE, d, n), dtype=np.float32, mode="r", shape=(T, V))
        for s in range(0, T, 128):
            la = log_softmax(np.asarray(a[s:s + 128]))
            lb = log_softmax(np.asarray(b[s:s + 128]))
            assert np.isfinite(la).all() and np.isfinite(lb).all()
            kl_all[i, s:s + 128] = (np.exp(la) * (la - lb)).sum(axis=1)
            agree[i, s:s + 128] = la.argmax(axis=1) == lb.argmax(axis=1)
    res[label] = (kl_all, agree)
    print(f"done {label}: pooled mean KL {kl_all.mean():.4e} top-1 {100 * agree.mean():.4f}%", flush=True)

print()
print(f"{'positions':>12} | " + " | ".join(f"{l:>34}" for l, _ in PAIRS))
print(f"{'':>12} | " + " | ".join(f"{'mean KL':>11} {'p99 KL':>11} {'top-1 %':>9}" for _ in PAIRS))
for lo, hi in BUCKETS:
    cells = []
    for label, _ in PAIRS:
        kl, ag = res[label]
        k = kl[:, lo:hi].ravel()
        cells.append(f"{k.mean():11.4e} {np.percentile(k, 99):11.4e} {100 * ag[:, lo:hi].mean():9.3f}")
    print(f"{lo:5d}-{hi - 1:<6d} | " + " | ".join(cells))
print()
fl = res["floor: llama b16"][0]
for label in ("gufo rows 256", "gufo rows 16"):
    kl = res[label][0]
    r = [kl[:, lo:hi].mean() / fl[:, lo:hi].mean() for lo, hi in ((0, 1024), (1024, 2048))]
    print(f"{label}: mean KL / floor mean KL = {r[0]:.3f} before the window, {r[1]:.3f} after")
print("per text, positions 1024-2047, mean KL: " + "; ".join(
    f"{n[:-4]} floor {fl[i, 1024:].mean():.2e} g256 {res['gufo rows 256'][0][i, 1024:].mean():.2e}" for i, n in enumerate(names)))
