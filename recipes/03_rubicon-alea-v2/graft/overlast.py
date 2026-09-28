#!/usr/bin/env python3
r"""overlast.py — media dei pesi degli ultimi N checkpoint ("over_last").

Ricetta di CHIUSURA usata sia per rubicon-alea-v1 sia per v2: mediare i pesi
degli ultimi checkpoint riduce la varianza rispetto a prendere l'ultimo singolo
(che e' un campione rumoroso della traiettoria SGD). Per la v2 il gate ha dato
overlast vs ep459-singolo = +3.81±7.67 (parita' con lean positivo) -> si spedisce
la media, per principio di minor varianza a parita' statistica.

USO (dalla cartella nnue-pytorch della copia di lavoro, venv attivo):
  python ../graft/overlast.py <rundir> <out.pt> [--n 3] [--features "..."]

  <rundir>  es. ~/alea/runs/alea_v2_anneal   (contiene lightning_logs/*/checkpoints/)
  <out.pt>  modulo Lightning picklato, pronto per serialize.py

Prende gli ultimi N checkpoint periodici (epoch=*.ckpt, ordinati per epoca) e,
se presente, last.ckpt. NB: i .ckpt Lightning hanno le chiavi con prefisso
"model." -> si media direttamente lo state_dict completo.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.getcwd())  # la cartella nnue-pytorch (per i package model/)

import torch

import model as M
from model.config import NNUELightningConfig

DEFAULT_FEATURES = "Full_Threats+HalfKAv2_hm^+PawnPair"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir", help="run dir (contiene lightning_logs/*/checkpoints/)")
    ap.add_argument("out_pt", help="output .pt (modulo Lightning picklato)")
    ap.add_argument("--n", type=int, default=3, help="quanti checkpoint mediare (default 3)")
    ap.add_argument("--features", default=DEFAULT_FEATURES)
    a = ap.parse_args()

    ckdirs = glob.glob(os.path.join(os.path.expanduser(a.rundir), "lightning_logs", "*", "checkpoints"))
    if not ckdirs:
        sys.exit(f"nessun checkpoints/ sotto {a.rundir}")
    ckdir = sorted(ckdirs)[-1]

    def epoch_of(p: str) -> int:
        return int(os.path.basename(p).split("epoch=")[1].split("-")[0])

    cks = sorted(glob.glob(f"{ckdir}/epoch=*.ckpt"), key=epoch_of)[-(a.n - 1):]
    last = f"{ckdir}/last.ckpt"
    if os.path.exists(last):
        cks.append(last)
    if not cks:
        sys.exit(f"nessun checkpoint in {ckdir}")
    print("[i] medio:", [os.path.basename(c) for c in cks])

    sds = [torch.load(c, map_location="cpu", weights_only=False)["state_dict"] for c in cks]
    avg = {k: (sum(sd[k].float() for sd in sds) / len(sds)).to(sds[0][k].dtype) for k in sds[0]}

    cfg = NNUELightningConfig(features=a.features)
    nnue = M.NNUE.load_from_checkpoint(cks[-1], config=cfg, map_location="cpu")
    nnue.load_state_dict(avg)
    nnue.eval()

    out = os.path.expanduser(a.out_pt)
    torch.save(nnue, out)
    print(f"[OK] {out}")
    print("     prossimo passo: serialize.py <out.pt> <net.nnue> --features \"%s\" [--ft_optimize ...]" % a.features)


if __name__ == "__main__":
    main()
