#!/usr/bin/env python3
r"""graft_passedpawns.py — innesto zero-init del blocco PassedPawns sulla v2 finale.

Carica la v2 (3 blocchi: Full_Threats+HalfKAv2_hm^+PawnPair, tipicamente il
net SWA dell'anneal), costruisce il modello a 4 blocchi (…+PassedPawns), copia
TUTTI i parametri condivisi, azzera COMPLETAMENTE il blocco PassedPawns
(colonne L1 + PSQT) e salva il modulo Lightning picklato da passare a
`train.py --resume-from-model`.

Il feature transformer è lineare (accumulatore = somma delle colonne attive):
colonne a zero ⇒ output IDENTICO alla v2 al giorno zero. Gli indici globali dei
3 blocchi esistenti NON cambiano (PassedPawns è appeso in coda: 87808..87903).

USO (dalla cartella nnue-pytorch della copia di lavoro):
  python ..\graft\graft_passedpawns.py <v2_finale.nnue|.pt> ..\graft\alea_v3_grafted.pt
"""
import os
import sys

sys.path.insert(0, os.getcwd())  # la cartella nnue-pytorch (per i package model/data_loader)

import torch

import model as M
from model.config import NNUELightningConfig, ModelConfig
from model.utils.load_model import load_model

OLD_FEATURES = "Full_Threats+HalfKAv2_hm^+PawnPair"
NEW_FEATURES = "Full_Threats+HalfKAv2_hm^+PawnPair+PassedPawns"


def main(src: str, out_pt: str) -> None:
    print(f"[i] carico v2: {src}  ({OLD_FEATURES})")
    if src.endswith(".pt"):
        old_nnue = torch.load(src, weights_only=False, map_location="cpu")
        old_model = old_nnue.model
    else:
        old_model = load_model(src, OLD_FEATURES, ModelConfig())

    print(f"[i] costruisco il modello 4-blocchi ({NEW_FEATURES})")
    cfg = NNUELightningConfig(features=NEW_FEATURES)
    nnue = M.NNUE(config=cfg)

    # Copia tutti i parametri condivisi. Le chiavi mancanti DEVONO essere solo
    # il blocco nuovo (input.features.3.*); nessuna chiave inattesa ammessa.
    missing, unexpected = nnue.model.load_state_dict(old_model.state_dict(), strict=False)
    assert not unexpected, f"chiavi inattese dal modello v2: {unexpected}"
    bad = [k for k in missing if not k.startswith("input.features.3.")]
    assert not bad, f"chiavi condivise NON copiate: {bad}"
    print(f"[i] copiati {len(old_model.state_dict())} tensori; nuovi (blocco PassedPawns): {len(missing)}")

    # Zero-init del blocco innestato: TUTTE le colonne (L1 + PSQT).
    with torch.no_grad():
        pp = nnue.model.input.features[3]
        assert pp.FEATURE_NAME == "PassedPawns"
        pp.weight.zero_()
        assert pp.weight.abs().max().item() == 0.0

    nnue.train()
    torch.save(nnue, out_pt)
    print(f"[OK] graft salvato: {out_pt}")
    print(f"     NUM_INPUTS (training): {old_model.input.NUM_INPUTS} -> {nnue.model.input.NUM_INPUTS} (+96)")
    print(f"     NUM_REAL_FEATURES (export .nnue): {nnue.model.input.NUM_REAL_FEATURES} (atteso 87904)")
    print(f"     feature hash: {nnue.model.feature_hash:#010x}")
    print("     prossimo passo: python ..\\graft\\verify_passedpawns.py per l'identita' zero-init")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
