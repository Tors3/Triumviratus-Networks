"""Best out_scaling per group (26/10/2026): with the trainer's mapping pf(s) = 0.5 * (1 + sig((s - 270) / k) -
sig((-s - 270) / k)), the k that minimises the cross-entropy of the game result, per piece band and per C2 expert of
the side to move (more than 5 pieces). k = 380 is the trainer's value; a smaller k means the results are more decisive
than the score. Usage: python scaling_fit_vm.py <glob> <c2 net> [batches]"""
import glob, struct, sys
import numpy as np, torch
import data_loader
from data_loader._native import set_router_table
from model.nnue import remap_tablebase_score
import model as M
files = sorted(glob.glob(sys.argv[1])); net = sys.argv[2]; nb = int(sys.argv[3]) if len(sys.argv) > 3 else 100
torch.set_num_threads(1)
lp = M.NNUELightningConfig().loss_params
d = open(net, "rb").read(); i = d.find(struct.pack("<III", 0x52544231, 2916, 8))
router = np.frombuffer(d[i + 12:i + 12 + 2916], dtype=np.uint8).copy(); set_router_table(router)
table = torch.from_numpy(router.astype(np.int64))
prov = data_loader.SparseBatchProvider(feature_set="Full_Threats+HalfKAv2_hm_X8+PP_3Wide+PassedPawns", filenames=files,
        cyclic=True, num_workers=6, batch_size=16384,
        config=data_loader.DataloaderSkipConfig(random_fen_skipping=10, filtered=True))
S, R, PC, EX = [], [], [], []
for _ in range(nb):
    us, them, wi, bi, outcome, score, pc, *rest = next(prov)
    aux = rest[0]
    s = remap_tablebase_score(score, base=lp.tb_remap_base, scale=lp.tb_remap_scale, decay=lp.tb_remap_decay)
    ew, eb = table[aux[:, 0].long()], table[aux[:, 1].long()]
    S.append(s.view(-1).numpy()); R.append(outcome.view(-1).numpy()); PC.append(pc.view(-1).numpy())
    EX.append(torch.where(us.view(-1) > 0.5, ew, eb).numpy())
s, r, pc, ex = (np.concatenate(x).astype(np.float64) for x in (S, R, PC, EX))
sig = lambda x: 1 / (1 + np.exp(-x))
def pf(k, ss): return 0.5 * (1 + sig((ss - 270) / k) - sig((-ss - 270) / k))
def xent(p, o):
    p = np.clip(p, 1e-6, 1 - 1e-6); return -np.mean(o * np.log(p) + (1 - o) * np.log(1 - p))
def best(m):
    ss, o = s[m], r[m]
    ks = np.arange(20, 601, 5)
    ce = [xent(pf(k, ss), o) for k in ks]
    j = int(np.argmin(ce)); return ks[j], ce[j], xent(pf(380, ss), o)
print(f"positions {len(s):,}")
for name, m in [(f"pc {a}-{b}", (pc >= a) & (pc <= b)) for a, b in ((6, 9), (10, 15), (16, 23), (24, 32))] + \
               [(f"expert {e}", (ex == e) & (pc > 5)) for e in range(8)] + [("all pc>5", pc > 5)]:
    k, ce, ce380 = best(m)
    print(f"{name:12s} n {int(m.sum()):9,d}  best k {k:3d}  xent {ce:.4f} (k=380: {ce380:.4f})")
