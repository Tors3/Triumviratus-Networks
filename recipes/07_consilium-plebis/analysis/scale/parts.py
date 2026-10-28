"""Scale of the two parts (psqt, positional) of Consilium-2 against Consilium, per layer-stack bucket (21/10/2026).
Usage: python parts.py fens c2.eng consilium.eng"""
import sys

import numpy as np

fens = [l.strip() for l in open(sys.argv[1]) if l.strip()]
c2 = np.loadtxt(sys.argv[2], dtype=np.int64).reshape(-1, 3)
c1 = np.loadtxt(sys.argv[3], dtype=np.int64).reshape(-1, 3)
pieces = np.array([sum(ch.isalpha() for ch in f.split()[0]) for f in fens])
bucket = (pieces - 1) // 4
keep = (np.abs(c1[:, 2]) <= 1500) & (np.abs(c2[:, 2]) <= 1500)
print(f"{'bucket':>6} {'n':>6} | {'psqt sd C1':>10} {'psqt sd C2':>10} {'ratio':>6} | {'pos sd C1':>9} {'pos sd C2':>9} {'ratio':>6} |"
      f" {'final/raw C1':>12} {'final/raw C2':>12}")
for k in range(8):
    m = keep & (bucket == k)
    if m.sum() < 30:
        continue
    p1, p2 = c1[m, 0], c2[m, 0]
    q1, q2 = c1[m, 1], c2[m, 1]
    r1, r2 = c1[m, 0] + c1[m, 1], c2[m, 0] + c2[m, 1]
    f1, f2 = c1[m, 2], c2[m, 2]
    g1 = np.polyfit(r1, f1, 1)[0]   # how the engine's scaling maps raw to final, per network
    g2 = np.polyfit(r2, f2, 1)[0]
    print(f"{k:6d} {m.sum():6d} | {p1.std():10.1f} {p2.std():10.1f} {p2.std() / p1.std():6.3f} | {q1.std():9.1f} {q2.std():9.1f}"
          f" {q2.std() / q1.std():6.3f} | {g1:12.3f} {g2:12.3f}")
