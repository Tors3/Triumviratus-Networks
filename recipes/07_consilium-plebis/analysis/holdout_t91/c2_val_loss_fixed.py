"""Loss of Consilium and Consilium-2 networks on the same holdout with lambda FIXED (21/10/2026).

The log's train loss does not compare two runs: data mix, batch and lambda schedule differ. Here every network is
evaluated on batches of the same file with the same skip configuration and the default LossParams of both runs, at
lambda 1.0 (data eval only), 0.75 (the project's reference) and 0.0 (game result only). The two feature sets cannot
share one stream (the loader shuffles with a thread-local generator), so each network reads its own sample of the
same file: with a few million positions the sampling error of the mean stays far below the differences of interest.
For a Consilium-2 network the loader is given the network's routing table first (as train.py does).
Usage: python c2_val_loss_fixed.py <holdout.binpack> <net.nnue> [<net.nnue> ...] [--batches 400]
"""
import argparse
import os

import torch

import data_loader
import model as M
from data_loader._native import set_router_table
from model.nnue import calculate_sf_loss
from model.utils.serialize import NNUEReader

FEATURES = {"c2": "Full_Threats+HalfKAv2_hm_X8^+PP_3Wide+PassedPawns",
            "consilium": "Full_Threats+HalfKAv2_hm_P4^+PP_3Wide+PassedPawns"}
LAMBDAS = (1.0, 0.75, 0.0)


def load(path):
    """Tries the Consilium-2 layout first, then Consilium's."""
    for kind in ("c2", "consilium"):
        cfg = M.ModelConfig(L1=1024)
        if kind == "c2":
            cfg.pawn_king_hidden = 256
        try:
            with open(path, "rb") as f:
                m = NNUEReader(f, FEATURES[kind], cfg).model
            return kind, m
        except Exception:
            continue
    raise SystemExit(f"cannot read {path} as Consilium-2 or Consilium")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("nets", nargs="+")
    ap.add_argument("--batches", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    loss_params = M.NNUELightningConfig().loss_params
    rows = []
    for path in a.nets:
        kind, m = load(path)
        m.eval()
        for f in m.input.features:
            if hasattr(f, "router"):
                set_router_table(f.router.detach().cpu().numpy())
        # On Windows the stream creation fails now and then with an access violation (seen on 21/10 with every
        # feature set, never on the Linux VMs): a few attempts.
        for attempt in range(8):
            try:
                provider = data_loader.SparseBatchProvider(
                    feature_set=m.input_feature_name, filenames=[a.data], cyclic=True, num_workers=2,
                    batch_size=a.batch_size,
                    config=data_loader.DataloaderSkipConfig(random_fen_skipping=10, filtered=True))
                break
            except OSError as e:
                print(f"stream creation failed ({e}), attempt {attempt + 1}", flush=True)
        else:
            raise SystemExit("cannot create the data stream")
        sums, sq = [0.0] * len(LAMBDAS), [0.0] * len(LAMBDAS)
        with torch.no_grad():
            for i in range(a.batches):
                us, them, wi, bi, outcome, score, pc, *rest = next(provider)
                aux = rest[0] if rest else None
                y = m(us, them, wi, bi, pc, True, True, aux) * m.quantization.nnue2score
                for k, lam in enumerate(LAMBDAS):
                    v = calculate_sf_loss(y, score, outcome, loss_params, lam).item()
                    sums[k] += v
                    sq[k] += v * v
                if (i + 1) % 50 == 0:
                    print(f"{os.path.basename(path)}: batch {i + 1}/{a.batches}", flush=True)
        n = a.batches
        mean = [s / n for s in sums]
        se = [((q / n - mu * mu) / max(1, n - 1)) ** 0.5 for q, mu in zip(sq, mean)]   # over batch means
        rows.append((os.path.basename(path), kind, mean, se))
    print(f"\n{a.batches} batches x {a.batch_size} positions of {os.path.basename(a.data)} per network (x1e-3, +- s.e.)")
    print(f"{'network':<40}{'kind':<11}" + "".join(f"lambda {lam:<14}" for lam in LAMBDAS))
    for name, kind, mean, se in rows:
        print(f"{name:<40}{kind:<11}" + "".join(f"{mu * 1e3:7.4f} +-{s * 1e3:6.4f}  " for mu, s in zip(mean, se)))


if __name__ == "__main__":
    main()
