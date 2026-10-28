"""Paired comparison of t91 holdout losses (25/10/2026): per-batch losses from c2_val_loss_fixed2.py --dump, every
network on the same positions. Prints mean per network and the difference from the reference network (first in the
file unless --ref), with the s.e. of the paired per-batch differences, at each lambda.
Usage: python t91_paired.py <dump.txt> [--ref nn-consilium.nnue]
"""
import argparse
import math
from collections import OrderedDict

LAMBDAS = (1.0, 0.75, 0.0)

ap = argparse.ArgumentParser()
ap.add_argument("dump")
ap.add_argument("--ref", default=None)
a = ap.parse_args()

nets = OrderedDict()
for line in open(a.dump):
    name, i, *v = line.split()
    nets.setdefault(name, {})[int(i)] = [float(x) for x in v]
ref = a.ref or next(iter(nets))
R = nets[ref]
print(f"reference {ref}; x1e-3; difference = network minus reference over common batches (paired s.e.)")
print(f"{'network':<26}{'batches':>8}" + "".join(f"  {'lam ' + str(l):>9} {'diff':>8} {'+-':>6} {'%':>6}" for l in LAMBDAS))
for name, rows in nets.items():
    common = sorted(set(rows) & set(R))
    n = len(common)
    cells = ""
    for k in range(len(LAMBDAS)):
        mean = sum(rows[i][k] for i in common) / n
        d = [rows[i][k] - R[i][k] for i in common]
        dm = sum(d) / n
        se = math.sqrt(sum((x - dm) ** 2 for x in d) / max(1, n - 1) / n) if n > 1 else float("nan")
        rm = sum(R[i][k] for i in common) / n
        cells += f"  {mean * 1e3:9.4f} {dm * 1e3:+8.4f} {se * 1e3:6.4f} {dm / rm * 100:+6.2f}"
    print(f"{name:<26}{n:>8}" + cells)
