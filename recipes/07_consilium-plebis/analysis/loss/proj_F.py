"""Projection of F's final training loss (27/10/2026): least-squares fit of L = a + f(LR) + g*E on P (LR <= 30%, after
the warmup) and F (after its rise), E = global epoch (F step-epoch k = 497 + k), LR at mid-epoch as % of P's peak.
f linear or quadratic in LR. Then F's remaining epochs under the cut schedule (flat 10 epochs, 60 in all).
Usage: python proj_F.py <c2_epoch_step_loss.txt> <F metrics.csv>..."""
import math, sys
ARGS = sys.argv[1:]
sys.argv = ["x", "--it"]
from plot_c2_pf import load, load_f, lr_frac, EPOCH_STEPS
P = load(ARGS[0]); F = load_f(*ARGS[1:])
NEW = [(7628, 0.1494, 1, "lin"), (87722, 1, 0.6, "cos"), (38140, 0.6, 0.6, "lin"), (95350, 0.6, 0.00016, "cos")]
def frac(pieces, s):
    for n, a, b, sh in pieces:
        if s < n:
            t = s / n
            if sh == "cos": t = (1 - math.cos(math.pi * t)) / 2
            return a + (b - a) * t
        s -= n
    return pieces[-1][2]
H = EPOCH_STEPS // 2
data = [(lr_frac(s - H) * 100, e, l, "P") for e, s, l in P if e > 29 and lr_frac(s - H) <= 0.30]
data += [(25 * frac(NEW, s - H), 497 + e, l, "F") for e, s, l in F if e >= 2]

def solve(X, y):
    n = len(X[0]); A = [[sum(r[i] * r[j] for r in X) for j in range(n)] for i in range(n)]
    b = [sum(r[i] * v for r, v in zip(X, y)) for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda k: abs(A[k][i])); A[i], A[p] = A[p], A[i]; b[i], b[p] = b[p], b[i]
        for k in range(i + 1, n):
            f = A[k][i] / A[i][i]
            for j in range(i, n): A[k][j] -= f * A[i][j]
            b[k] -= f * b[i]
    x = [0] * n
    for i in reversed(range(n)): x[i] = (b[i] - sum(A[i][j] * x[j] for j in range(i + 1, n))) / A[i][i]
    return x

for name, feats in (("linear in LR", lambda lr, e: [1, lr, e - 500]),
                    ("quadratic in LR", lambda lr, e: [1, lr, lr * lr, e - 500])):
    X = [feats(lr, e) for lr, e, _, _ in data]; y = [l for _, _, l, _ in data]
    c = solve(X, y)
    res = [v - sum(a * b for a, b in zip(c, r)) for r, v in zip(X, y)]
    sd = math.sqrt(sum(r * r for r in res) / (len(y) - len(c)))
    pred = lambda lr, e: sum(a * b for a, b in zip(c, feats(lr, e)))
    last = [(k, 25 * frac(NEW, k * EPOCH_STEPS + H)) for k in range(55, 60)]
    fin = sum(pred(lr, 497 + k) for k, lr in last) / len(last)
    print(f"{name}: residual sd {sd * 1e6:.2f}e-6, learning g {c[-1] * 1e6:+.3f}e-6/ep")
    for k in (32, 34, 40, 45, 50, 55, 59):
        lr = 25 * frac(NEW, k * EPOCH_STEPS + H)
        print(f"   F ep{k:2d} LR {lr:5.2f}%  {pred(lr, 497 + k):.6f}")
    print(f"   F mean ep55-59 {fin:.6f}; P continued to LR 0 at ep549 {pred(0.004, 549):.6f}; P at ep496 fit {pred(3.75, 496):.6f}")
pm = sum(l for e, _, l in P if e >= 492) / 5
print(f"P ep496 {P[-1][2]:.6f}, P mean ep492-496 {pm:.6f}")
