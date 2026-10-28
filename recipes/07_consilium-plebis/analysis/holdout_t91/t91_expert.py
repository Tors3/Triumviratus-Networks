"""t91 loss per Consilium Plebis expert (side to move's perspective), networks compared on the same positions
(26/10/2026). Reads the .npz files of c2_val_loss_fixed2.py --npz-dir: lw (loss * weight per lambda), w, score, us,
ew/eb (experts of White's and Black's perspective, Consilium-2 runs only). The expert comes from the first
Consilium-2 file. Differences are ratio estimates sum(lw2 - lw1) / sum(w) with the s.e. over batches of 8192.
Usage: python t91_expert.py <ref.npz> <net.npz> [<net.npz> ...]   (ref first, e.g. Consilium)
"""
import math
import os
import sys

import numpy as np

LAMBDAS = ("1.0", "0.75", "0")
TRAIN_SHARE = (8.16, 7.58, 12.33, 3.15, 13.05, 11.75, 22.44, 21.54)   # router_em.jsonl, epoch 64 (per perspective)
BATCH = 8192

files = sys.argv[1:]
z = [np.load(f) for f in files]
names = [os.path.splitext(os.path.basename(f))[0] for f in files]
for n, zz in zip(names[1:], z[1:]):
    assert np.array_equal(zz["score"], z[0]["score"]), f"{n}: not the same positions as {names[0]}"
src = next(zz for zz in z if "ew" in zz.files)
expert = np.where(src["us"] > 0, src["ew"], src["eb"])
w = z[0]["w"]
nb = len(w) // BATCH
blk = np.arange(len(w)) // BATCH


def ratio(num, den, mask):
    """sum(num) / sum(den) over mask, with the s.e. of the ratio estimate over the batches."""
    sn = np.bincount(blk[mask], weights=num[mask], minlength=nb)
    sd = np.bincount(blk[mask], weights=den[mask], minlength=nb)
    r = sn.sum() / sd.sum()
    se = math.sqrt(nb / (nb - 1) * ((sn - r * sd) ** 2).sum()) / sd.sum()
    return r, se


print(f"positions {len(w)} ({nb} batches); expert = side to move's perspective (opponent's material)")
for k, lam in enumerate(LAMBDAS):
    print(f"\n=== lambda {lam}: loss x1e-3, difference to {names[0]} in % (+- s.e.)")
    print(f"{'expert':>6} {'t91 %':>6} {'train %':>7} {names[0][:12]:>12}" +
          "".join(f" | {n[:14]:>14} {'diff%':>7} {'+-':>5}" for n in names[1:]))
    rows = []
    for e in list(range(8)) + ["all"]:
        mask = np.ones(len(w), bool) if e == "all" else expert == e
        ref = z[0]["lw"][k][mask].sum() / w[mask].sum()
        line = f"{e:>6} {mask.mean() * 100:6.1f} {('' if e == 'all' else f'{TRAIN_SHARE[e]:.1f}'):>7} {ref * 1e3:12.4f}"
        for zz in z[1:]:
            m2 = zz["lw"][k][mask].sum() / w[mask].sum()
            d, se = ratio(zz["lw"][k] - z[0]["lw"][k], w, mask)
            line += f" | {m2 * 1e3:14.4f} {d / ref * 100:+7.2f} {se / ref * 100:5.2f}"
        print(line)
    # share of the total improvement of the last network over the reference that each expert brings
    tot = (z[-1]["lw"][k] - z[0]["lw"][k]).sum()
    print("share of the total difference " + names[-1] + " - " + names[0] + ": " +
          " ".join(f"e{e} {((z[-1]['lw'][k] - z[0]['lw'][k])[expert == e].sum() / tot * 100):.0f}%" for e in range(8)))
    if len(z) >= 3:
        a, b = z[-2], z[-1]
        tot2 = (b["lw"][k] - a["lw"][k]).sum()
        print(f"{names[-2]} -> {names[-1]} by expert (diff % of {names[-2]}, share of the change): " +
              " ".join(f"e{e} {ratio(b['lw'][k] - a['lw'][k], w, expert == e)[0] / (a['lw'][k][expert == e].sum() / w[expert == e].sum()) * 100:+.2f}%"
                       f"/{((b['lw'][k] - a['lw'][k])[expert == e].sum() / tot2 * 100):.0f}%" for e in range(8)))
