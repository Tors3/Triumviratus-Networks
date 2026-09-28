#!/usr/bin/env python3
r"""graft_pawnpair.py — innesto zero-init del blocco PawnPair su rubicon-alea-v1.

Carica la v1 (2 blocchi: Full_Threats+HalfKAv2_hm), costruisce il modello a
3 blocchi (…+PawnPair), copia TUTTI i parametri condivisi, azzera COMPLETAMENTE
il blocco PawnPair (colonne L1 + PSQT) e salva il modulo Lightning picklato da
passare a `train.py --resume_from_model`.

Il feature transformer è lineare (accumulatore = somma delle colonne attive):
colonne a zero ⇒ output IDENTICO alla v1 al giorno zero. Gli indici globali dei
2 blocchi esistenti NON cambiano (PawnPair è appeso in coda: 83248..87807).

USO (dalla cartella nnue-pytorch della copia di lavoro):
  python ..\graft\graft_pawnpair.py ..\..\..\Triumviratus_5\x64\Release\nn-rubicon-alea-v1.nnue ..\graft\alea_v2_grafted.pt
"""
import os
import sys

sys.path.insert(0, os.getcwd())  # la cartella nnue-pytorch (per i package model/data_loader)

import torch

import model as M
from model.config import NNUELightningConfig, ModelConfig
from model.utils.load_model import load_model

OLD_FEATURES = "Full_Threats+HalfKAv2_hm^"
NEW_FEATURES = "Full_Threats+HalfKAv2_hm^+PawnPair"


def main(src_nnue: str, out_pt: str) -> None:
    print(f"[i] carico v1: {src_nnue}  ({OLD_FEATURES})")
    old_model = load_model(src_nnue, OLD_FEATURES, ModelConfig())

    print(f"[i] costruisco il modello 3-blocchi ({NEW_FEATURES})")
    cfg = NNUELightningConfig(features=NEW_FEATURES)
    nnue = M.NNUE(config=cfg)

    # Copia tutti i parametri condivisi. Le chiavi mancanti DEVONO essere solo
    # il blocco nuovo (input.features.2.*); nessuna chiave inattesa ammessa.
    missing, unexpected = nnue.model.load_state_dict(old_model.state_dict(), strict=False)
    assert not unexpected, f"chiavi inattese dal modello v1: {unexpected}"
    bad = [k for k in missing if not k.startswith("input.features.2.")]
    assert not bad, f"chiavi condivise NON copiate: {bad}"
    print(f"[i] copiati {len(old_model.state_dict())} tensori; nuovi (blocco PawnPair): {len(missing)}")

    # Zero-init del blocco innestato: TUTTE le colonne (L1 + PSQT).
    with torch.no_grad():
        pp = nnue.model.input.features[2]
        assert pp.FEATURE_NAME == "PawnPair"
        pp.weight.zero_()
        assert pp.weight.abs().max().item() == 0.0

    nnue.train()
    torch.save(nnue, out_pt)
    print(f"[OK] graft salvato: {out_pt}")
    print(f"     NUM_INPUTS (training): {old_model.input.NUM_INPUTS} -> {nnue.model.input.NUM_INPUTS} (atteso 89856)")
    print(f"     NUM_REAL_FEATURES (export .nnue): {nnue.model.input.NUM_REAL_FEATURES} (atteso 87808)")
    print(f"     feature hash: {nnue.model.feature_hash:#010x}")
    print("     prossimo passo: python ..\\graft\\verify_graft.py per l'identita' zero-init")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
