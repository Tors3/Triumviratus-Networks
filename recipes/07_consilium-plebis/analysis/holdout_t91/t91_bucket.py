"""t91 loss per layer-stack bucket, Consilium Plebis against Consilium on the same positions (26/10/2026).
Reads the "b<k>" lines of c2_val_loss_fixed2.py --dump --by-bucket (network, batch, bucket, positions, loss per
lambda). Per bucket: loss of each network (positions-weighted over batches), the difference C2 - C1 in % of C1 and its
paired s.e. (per-batch differences, weighted by positions).
Usage: python t91_bucket.py <c1_dump.txt> <c2_dump.txt>
"""
import math
import sys
from collections import defaultdict

LAMBDAS = ("1.0", "0.75", "0")


def load(path):
    d = defaultdict(dict)   # bucket -> batch -> (n, [loss per lambda])
    for line in open(path):
        p = line.split()
        if len(p) == 7 and p[2].startswith("b"):
            d[int(p[2][1:])][int(p[1])] = (int(p[3]), [float(x) for x in p[4:]])
    return d


c1, c2 = load(sys.argv[1]), load(sys.argv[2])
total = sum(n for b in c1 for n, _ in c1[b].values())
print(f"{'bucket':>6} {'pieces':>7} {'share':>6}" + "".join(f" | lambda {l:<4} C1     C2     diff%   +-" for l in LAMBDAS))
for b in sorted(c1):
    common = sorted(set(c1[b]) & set(c2[b]))
    w = [c1[b][i][0] for i in common]
    W = sum(w)
    cells = ""
    for k in range(3):
        m1 = sum(c1[b][i][0] * c1[b][i][1][k] for i in common) / W
        m2 = sum(c1[b][i][0] * c2[b][i][1][k] for i in common) / W
        d = [c2[b][i][1][k] - c1[b][i][1][k] for i in common]
        dm = sum(wi * di for wi, di in zip(w, d)) / W
        se = math.sqrt(sum((wi * (di - dm)) ** 2 for wi, di in zip(w, d))) / W
        cells += f" | {m1 * 1e3:12.4f} {m2 * 1e3:6.4f} {dm / m1 * 100:+6.2f} {se / m1 * 100:4.2f}"
    print(f"{b:>6} {4 * b + 1:>3}-{4 * b + 4:<3} {W / total * 100:5.1f}%" + cells)
