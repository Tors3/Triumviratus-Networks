#!/usr/bin/env python3
"""
make_random_net.py — una rete .nnue a pesi CASUALI della forma voluta (L1, feature set).

Serve a misurare nel motore il costo in NPS di una rete piu' larga PRIMA di allenarla:
si compila Triumviratus_8 con -DTRIUMV_L1=<n>, si carica questa rete con EvalFile e si
confronta con la 1024 (nps_ab_interleaved.py, PGO, macchina scarica). Con pesi casuali la
sparsita' di L1 e' peggiore di una rete allenata, quindi il numero e' un limite SUPERIORE.

USO (dalla venv del Training70, dopo setup_1060.sh):
    cd Training_NNUE/Training70_LegioSeptima/nnue-pytorch
    venv/bin/python ../../Training80/make_random_net.py --l1 2048 /tmp/random_l1_2048.nnue
    venv/bin/python ../../Training80/make_random_net.py --l1 1536 /tmp/random_l1_1536.nnue
Il feature set di default e' quello della 7.0; con --features si cambia (deve esistere nel
trainer: il blocco Mobility lato Python NON c'e' ancora).
"""
import argparse
import os
import sys

sys.path.insert(0, os.getcwd())
import torch  # noqa: E402

from model.config import NNUELightningConfig  # noqa: E402
from model.nnue import NNUE  # noqa: E402
from model.utils import NNUEWriter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="file .nnue da scrivere")
    ap.add_argument("--l1", type=int, default=1024)
    ap.add_argument("--features", default=None, help="default: quello del trainer (7.0)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--perturb-phases", action="store_true",
                    help="HalfKA a esperti (HalfKAv2_hm_P4): delta per fase casuali invece che zero, "
                         "altrimenti le 4 fasi hanno gli stessi pesi e cross_check_eval non vede un indice di fase sbagliato")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = NNUELightningConfig()
    # 🔴 28/09/2026: era `cfg.L1 = args.l1`, la stessa trappola di gpu_ceiling.py (Training80/README.md): L1 vive in
    # cfg.model_config, un attributo L1 su NNUELightningConfig NON ha effetto e la rete "2048" usciva a 1024.
    cfg.model_config.L1 = args.l1
    if args.features:
        cfg.features = args.features
    print(f"feature set: {cfg.features}   L1: {cfg.model_config.L1}")

    nnue = NNUE(config=cfg)
    nnue.eval()
    if args.perturb_phases:
        n = 0
        with torch.no_grad():
            for m in nnue.modules():
                if hasattr(m, "base_weight"):
                    s = 0.5 * float(m.base_weight.abs().max())
                    m.weight.uniform_(-s, s)
                    n += 1
        if n == 0:
            sys.exit("--perturb-phases: nessun modulo con base_weight (il feature set non ha fasi)")
        print(f"delta per fase casuali su {n} moduli")
    desc = f"random L1={args.l1} (misura di costo, NON giocare)"
    writer = NNUEWriter(nnue.model, desc, ft_compression="leb128")
    with open(args.target, "wb") as f:
        f.write(writer.buf)
    print(f"scritto {args.target}: {os.path.getsize(args.target) / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
