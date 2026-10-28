"""Non-linear fits of the Consilium Plebis Elo trajectory (fixed 100k nodes rows of consilium2_elo_points.csv).

Weighted least squares (weight = 1 / sigma^2, sigma = margin95 / 1.96). Each model has one non-linear parameter, found on a
grid, with the linear ones solved exactly for each grid value (numpy only, no scipy):
  log         Elo = a + b ln(epoch)
  saturating  Elo = A - B exp(-epoch / tau)          (A = asymptote)
  power       Elo = A - B epoch^(-k)                 (A = asymptote)
Prints, per model: parameters, weighted RMS of the residuals in sigmas, the epoch where Elo crosses 0 (if it does) and
the asymptote. Usage: python consilium2_elo_fit.py [--from EPOCH]
"""
import csv, math, sys
from pathlib import Path
import numpy as np

frm = int(sys.argv[sys.argv.index("--from") + 1]) if "--from" in sys.argv else 0
rows = [r for r in csv.DictReader(open(Path(__file__).with_name("consilium2_elo_points.csv"), encoding="utf-8"))
        if r["condition"].startswith("fixed 100k") and int(r["epoch"]) >= frm]
ep = np.array([float(r["epoch"]) for r in rows])
elo = np.array([float(r["elo"]) for r in rows])
w = 1.0 / (np.array([float(r["margin95"]) for r in rows]) / 1.96) ** 2


def wls(X, y):
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    res = (y - X @ coef) * sw
    return coef, math.sqrt(float(res @ res) / max(len(y) - X.shape[1] - 1, 1))


def crossing(f, lo=1, hi=5000):
    if f(hi) < 0:
        return None
    for e in range(lo, hi):
        if f(e) >= 0:
            return e
    return None


print(f"{len(ep)} points, epochs {int(ep.min())}-{int(ep.max())}")
c, s = wls(np.c_[np.ones_like(ep), np.log(ep)], elo)
f = lambda e: c[0] + c[1] * math.log(e)
print(f"log         a={c[0]:7.1f} b={c[1]:6.2f} (+{c[1]*math.log(2):.1f} per doubling)  rms={s:.2f}  zero at ep {crossing(f)}"
      f"  ep200 {f(200):+.1f}  ep550 {f(550):+.1f}")
for name, basis, grid in (("saturating", lambda t: np.exp(-ep / t), np.arange(5, 2000, 1.0)),
                          ("power", lambda k: ep ** (-k), np.arange(0.02, 3.0, 0.005))):
    best = None
    for g in grid:
        cc, ss = wls(np.c_[np.ones_like(ep), -basis(g)], elo)
        if cc[1] > 0 and (best is None or ss < best[2]):
            best = (g, cc, ss)
    g, cc, ss = best
    if name == "saturating":
        f = lambda e, g=g, cc=cc: cc[0] - cc[1] * math.exp(-e / g)
        par = f"tau={g:.0f}"
    else:
        f = lambda e, g=g, cc=cc: cc[0] - cc[1] * e ** (-g)
        par = f"k={g:.3f}"
    print(f"{name:11s} A={cc[0]:7.1f} B={cc[1]:8.1f} {par}  rms={ss:.2f}  zero at ep {crossing(f)}"
          f"  ep200 {f(200):+.1f}  ep550 {f(550):+.1f}  (asymptote {cc[0]:+.1f})")
