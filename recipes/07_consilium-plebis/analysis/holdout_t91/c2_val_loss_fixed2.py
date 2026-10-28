"""c2_val_loss_fixed.py with the data stream created BEFORE the network is loaded (Windows, 21/10/2026).

On this Windows machine the stream creation fails with an access violation for some exports (ep31, ep45: 8 out of 8
attempts) and not for others (ep22, ep27), with the same routing table: it depends on the state of the process after
the network is read, not on the network. Here the routing table is read straight from the RTB1 section of the file,
the stream is created, and only then the network is loaded. Same loss, same sampling as c2_val_loss_fixed.py.
Usage: python c2_val_loss_fixed2.py <holdout.binpack> <net.nnue> [<net.nnue> ...] [--batches 200] [--threads 8]
"""
import argparse
import os
import struct

import numpy as np
import torch

import data_loader
from data_loader._native import set_router_table
from model.nnue import calculate_sf_loss, remap_tablebase_score

import c2_val_loss_fixed as base


def per_position_loss(scorenet, score, outcome, lp, lam):
    """calculate_sf_loss before its reduction (26/10/2026): loss * weight and weight of every position, so that any
    grouping of the positions (expert, bucket) gives sum(loss * weight) / sum(weight) as the trainer does."""
    score = remap_tablebase_score(score, base=lp.tb_remap_base, scale=lp.tb_remap_scale, decay=lp.tb_remap_decay)
    q = (scorenet - lp.in_offset) / lp.in_scaling
    qm = (-scorenet - lp.in_offset) / lp.in_scaling
    qf = 0.5 * (1.0 + q.sigmoid() - qm.sigmoid())
    s = (score - lp.out_offset) / lp.out_scaling
    sm = (-score - lp.out_offset) / lp.out_scaling
    pf = 0.5 * (1.0 + s.sigmoid() - sm.sigmoid())
    pt = pf * lam + outcome * (1.0 - lam)
    loss = torch.pow(torch.abs(pt - qf), lp.pow_exp)
    if lp.qp_asymmetry != 0.0:
        loss = loss * ((qf > pt) * lp.qp_asymmetry + 1)
    w = 1 + (2.0 ** lp.w1 - 1) * torch.pow((pf - 0.5) ** 2 * pf * (1 - pf), lp.w2)
    return (loss * w).view(-1), w.view(-1)

FEATURE_LOADER = "Full_Threats+HalfKAv2_hm_X8+PP_3Wide+PassedPawns"   # loader name of the C2 feature set (no "^")
FEATURE_LOADER_C1 = "Full_Threats+HalfKAv2_hm_P4+PP_3Wide+PassedPawns"   # Consilium (25/10/2026: same stream, no router)


def router_of(path):
    """Routing table of a Consilium-2 network; None for a network without RTB1 (Consilium)."""
    data = open(path, "rb").read()
    i = data.find(struct.pack("<III", 0x52544231, 2916, 8))
    if i < 0:
        return None
    return np.frombuffer(data[i + 12:i + 12 + 2916], dtype=np.uint8).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("nets", nargs="+")
    ap.add_argument("--batches", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--threads", type=int, default=8)
    # 25/10/2026: every network reads the same positions (checked for Consilium against Consilium-2), so the per-batch
    # losses written here give paired differences between networks.
    ap.add_argument("--dump", default=None, help="file for the per-batch losses (network, batch, one per lambda)")
    # 26/10/2026: the loss of each layer-stack bucket ((pieces - 1) / 4) too, as extra dump lines
    # "network batch b<k> positions loss..." (the loss of the positions of that bucket in that batch).
    ap.add_argument("--by-bucket", action="store_true")
    # 26/10/2026: every position, in a compressed .npz per network: loss * weight per lambda, weight, piece count,
    # score (to check that two runs read the same positions) and, for a Consilium-2 network, the expert of the side to
    # move's perspective (routing cell of the aux block through the network's table).
    ap.add_argument("--npz-dir", default=None)
    a = ap.parse_args()
    dump = open(a.dump, "w") if a.dump else None
    # 22/10/2026: torch threads are set only after the stream exists; with 36 threads set before, the stream creation
    # fails (access violation, 8/8), with 4 it works.
    torch.set_num_threads(1)
    import model as M
    loss_params = M.NNUELightningConfig().loss_params
    rows = []
    for path in a.nets:
        router = router_of(path)
        if router is not None:
            set_router_table(router)
        for attempt in range(8):
            try:
                provider = data_loader.SparseBatchProvider(
                    feature_set=FEATURE_LOADER if router is not None else FEATURE_LOADER_C1, filenames=[a.data],
                    cyclic=True, num_workers=2,
                    batch_size=a.batch_size,
                    config=data_loader.DataloaderSkipConfig(random_fen_skipping=10, filtered=True))
                break
            except OSError as e:
                print(f"stream creation failed ({e}), attempt {attempt + 1}", flush=True)
        else:
            raise SystemExit("cannot create the data stream")
        torch.set_num_threads(a.threads)
        kind, m = base.load(path)
        m.eval()
        sums, sq = [0.0] * len(base.LAMBDAS), [0.0] * len(base.LAMBDAS)
        acc = {k: [] for k in ("lw", "w", "pc", "score", "us", "ew", "eb")}
        table = torch.from_numpy(router.astype(np.int64)) if router is not None else None
        with torch.no_grad():
            for i in range(a.batches):
                us, them, wi, bi, outcome, score, pc, *rest = next(provider)
                aux = rest[0] if rest else None
                y = m(us, them, wi, bi, pc, True, True, aux) * m.quantization.nnue2score
                vals = []
                for k, lam in enumerate(base.LAMBDAS):
                    v = calculate_sf_loss(y, score, outcome, loss_params, lam).item()
                    sums[k] += v
                    sq[k] += v * v
                    vals.append(v)
                if a.npz_dir:
                    pl = [per_position_loss(y, score, outcome, loss_params, lam) for lam in base.LAMBDAS]
                    if i == 0:
                        for k, (lw, w) in enumerate(pl):
                            r = (lw.sum() / w.sum()).item()
                            if abs(r - vals[k]) > 1e-5 * abs(vals[k]):
                                print(f"WARNING per-position loss {r} != {vals[k]} at lambda {base.LAMBDAS[k]}", flush=True)
                    acc["lw"].append(torch.stack([lw for lw, _ in pl]).numpy().astype(np.float32))
                    acc["w"].append(pl[0][1].numpy().astype(np.float32))
                    acc["pc"].append(pc.view(-1).numpy().astype(np.uint8))
                    acc["score"].append(score.view(-1).numpy().astype(np.float32))
                    acc["us"].append(us.view(-1).numpy().astype(np.uint8))
                    if table is not None and aux is not None:
                        acc["ew"].append(table[aux[:, 0].long()].numpy().astype(np.uint8))
                        acc["eb"].append(table[aux[:, 1].long()].numpy().astype(np.uint8))
                if dump:
                    dump.write(f"{os.path.basename(path)} {i} " + " ".join(f"{v:.9g}" for v in vals) + "\n")
                    if a.by_bucket:
                        bucket = ((pc.view(-1) - 1) // 4).long()
                        for b in range(8):
                            sel = bucket == b
                            nb = int(sel.sum())
                            if nb == 0:
                                continue
                            bv = [calculate_sf_loss(y[sel], score[sel], outcome[sel], loss_params, lam).item()
                                  for lam in base.LAMBDAS]
                            dump.write(f"{os.path.basename(path)} {i} b{b} {nb} " + " ".join(f"{v:.9g}" for v in bv) + "\n")
                    dump.flush()
                if (i + 1) % 25 == 0:
                    print(f"{os.path.basename(path)}: batch {i + 1}/{a.batches}", flush=True)
        n = a.batches
        mean = [s / n for s in sums]
        se = [((q / n - mu * mu) / max(1, n - 1)) ** 0.5 for q, mu in zip(sq, mean)]
        rows.append((os.path.basename(path), kind, mean, se))
        if a.npz_dir:
            np.savez_compressed(os.path.join(a.npz_dir, os.path.splitext(os.path.basename(path))[0] + ".npz"),
                                lw=np.concatenate(acc["lw"], axis=1), **{k: np.concatenate(v) for k, v in acc.items()
                                                                         if k != "lw" and v})
        del provider, m
    print(f"\n{a.batches} batches x {a.batch_size} positions of {os.path.basename(a.data)} per network (x1e-3, +- s.e.)")
    print(f"{'network':<40}{'kind':<11}" + "".join(f"lambda {lam:<14}" for lam in base.LAMBDAS))
    for name, kind, mean, se in rows:
        print(f"{name:<40}{kind:<11}" + "".join(f"{mu * 1e3:7.4f} +-{s * 1e3:6.4f}  " for mu, s in zip(mean, se)))


if __name__ == "__main__":
    main()
