"""Calibration of the binpack score against the game result, per piece band and per C2 expert of the side to move
(26/10/2026, question of the author: is a lambda per phase worth it?). The trainer's target is
pt = lam * pf(score) + (1 - lam) * outcome, pf as in calculate_sf_loss (tablebase remap, out_offset, out_scaling).
If E[outcome | pf] == pf in a group, the result adds only noise there on average; a systematic gap is the bias a
lower lambda would correct. VM version of lambda_calib.py: runs on Linux in the trainer folder, reads one binpack or a
glob of them (the whole training mix, drawn in proportion to size as in training), the routing table from the RTB1
section of a C2 network, no network evaluation. Prints, per group: positions, mean |E[r|pf] - pf| over 20 pf bins,
slope of E[r|pf] on pf through (0.5, 0.5) over all positions and over |pf - 0.5| < 0.2, and the 20-bin curve.
Usage: python lambda_calib_vm.py <binpack or glob> <c2 net> [batches] [workers]"""
import glob
import struct
import sys

import numpy as np
import torch

import data_loader
from data_loader._native import set_router_table
from model.nnue import remap_tablebase_score
import model as M

FEATURE_LOADER = "Full_Threats+HalfKAv2_hm_X8+PP_3Wide+PassedPawns"


def router_of(path):
    data = open(path, "rb").read()
    i = data.find(struct.pack("<III", 0x52544231, 2916, 8))
    if i < 0:
        raise SystemExit(f"{path}: no RTB1 section")
    return np.frombuffer(data[i + 12:i + 12 + 2916], dtype=np.uint8).copy()


files = sorted(glob.glob(sys.argv[1]))
net = sys.argv[2]
nb = int(sys.argv[3]) if len(sys.argv) > 3 else 200
workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
torch.set_num_threads(1)
lp = M.NNUELightningConfig().loss_params
print(f"files {len(files)}; loss params out_offset {lp.out_offset} out_scaling {lp.out_scaling} "
      f"tb_remap {lp.tb_remap_base}/{lp.tb_remap_scale}/{lp.tb_remap_decay}", flush=True)
router = router_of(net)
set_router_table(router)
table = torch.from_numpy(router.astype(np.int64))
prov = data_loader.SparseBatchProvider(feature_set=FEATURE_LOADER, filenames=files, cyclic=True,
                                       num_workers=workers, batch_size=16384,
                                       config=data_loader.DataloaderSkipConfig(random_fen_skipping=10, filtered=True))
PF, R, PC, EX = [], [], [], []
for i in range(nb):
    us, them, wi, bi, outcome, score, pc, *rest = next(prov)
    aux = rest[0]
    s = remap_tablebase_score(score, base=lp.tb_remap_base, scale=lp.tb_remap_scale, decay=lp.tb_remap_decay)
    a = (s - lp.out_offset) / lp.out_scaling
    b = (-s - lp.out_offset) / lp.out_scaling
    pf = 0.5 * (1.0 + a.sigmoid() - b.sigmoid())
    ew, eb = table[aux[:, 0].long()], table[aux[:, 1].long()]
    stm_white = us.view(-1) > 0.5
    PF.append(pf.view(-1).numpy()); R.append(outcome.view(-1).numpy()); PC.append(pc.view(-1).numpy())
    EX.append(torch.where(stm_white, ew, eb).numpy())
    if (i + 1) % 50 == 0:
        print(f"batch {i + 1}/{nb}", flush=True)
pf, r, pc, ex = (np.concatenate(x) for x in (PF, R, PC, EX))
print(f"positions {len(pf):,}; outcome values {np.unique(np.round(r, 2))[:5]}")


def slope(p, o):
    x = p - 0.5
    return (x * (o - 0.5)).sum() / max(1e-12, (x * x).sum())


def report(name, m):
    p, o = pf[m], r[m]
    n = len(p)
    if n < 5000:
        print(f"{name:18s} n {n:9,d}  (too few)")
        return
    bins = np.clip((p * 20).astype(int), 0, 19)
    ece, rows = 0.0, []
    for k in range(20):
        mk = bins == k
        if mk.sum() < 200:
            continue
        gap = o[mk].mean() - p[mk].mean()
        ece += mk.sum() * abs(gap)
        rows.append((p[mk].mean(), o[mk].mean(), int(mk.sum())))
    near = np.abs(p - 0.5) < 0.2
    print(f"{name:18s} n {n:9,d}  mean|gap| {ece / n:.4f}  slope {slope(p, o):.3f}  slope(|pf-.5|<.2) "
          f"{slope(p[near], o[near]):.3f}  mean(r-pf) {np.mean(o - p):+.4f}  E(r-pf)^2 {np.mean((o - p) ** 2):.4f}")
    print("    " + "  ".join(f"{a:.2f}>{b:.2f}" for a, b, _ in rows))


print("\n-- per piece band (pieces incl. kings)")
for lo, hi in ((3, 5), (6, 9), (10, 15), (16, 23), (24, 32)):
    report(f"pc {lo}-{hi}", (pc >= lo) & (pc <= hi))
print("\n-- per C2 expert of the side to move, more than 5 pieces")
for e in range(8):
    report(f"expert {e}", (ex == e) & (pc > 5))
print("\n-- all, more than 5 pieces")
report("all pc>5", pc > 5)
