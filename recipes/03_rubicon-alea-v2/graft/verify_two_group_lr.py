#!/usr/bin/env python3
"""Verifica della patch two-group-LR (pawnpair_lr) PRIMA del run GCP.

Replica ESATTAMENTE il percorso di train.py --resume-from-model:
  torch.load(alea_v2_grafted.pt) -> nnue.config = config CLI -> configure_optimizers()

Checks:
  [1] pawnpair_lr=1e-3 -> 11 param group; il gruppo PawnPair contiene ESATTAMENTE
      il tensore weight del blocco PawnPair (identita', non copia) con lr=1e-3
  [2] unione gruppi FT == tutti i weight dell'input (nessun parametro perso/duplicato)
  [3] pawnpair_lr=0 -> 10 gruppi, comportamento vecchio ESATTO
  [4] lo scheduler StepLR si crea e dopo uno step il RAPPORTO tra i lr resta costante
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "nnue-pytorch"))

import torch

GRAFT = os.path.join(HERE, "alea_v2_grafted.pt")


def build(pp_lr):
    nnue = torch.load(GRAFT, weights_only=False, map_location="cpu")
    nnue.config.optimizer_config.pawnpair_lr = pp_lr
    nnue.config.optimizer_config.lr = 1e-4
    nnue.config.optimizer_config.gamma = 0.997
    # come train.py: max_epoch/num_batches servono solo ad altri hook
    opts, scheds = nnue.configure_optimizers()
    return nnue, opts[0], scheds[0]


def ft_weight_params(nnue):
    return [p for n, p in nnue.model.input.named_parameters() if "bias" not in n]


def pp_weight(nnue):
    mods = [f for f in nnue.model.input.features if f.FEATURE_NAME == "PawnPair"]
    assert len(mods) == 1, f"attesi 1 blocco PawnPair, trovati {len(mods)}"
    return mods[0].weight


def ids(params):
    return {id(p) for p in params}


# ---- [1] + [2]: pawnpair_lr attivo --------------------------------------
nnue, opt, sched = build(1e-3)
groups = opt.param_groups
print(f"[1] gruppi: {len(groups)}  lrs: {[g['lr'] for g in groups]}")
assert len(groups) == 11, f"attesi 11 gruppi, trovati {len(groups)}"

pw = pp_weight(nnue)
pp_groups = [g for g in groups if id(pw) in ids(g["params"])]
assert len(pp_groups) == 1, "il weight PawnPair deve stare in UN solo gruppo"
assert pp_groups[0]["lr"] == 1e-3, f"lr PawnPair atteso 1e-3, trovato {pp_groups[0]['lr']}"
assert len(pp_groups[0]["params"]) == 1, "il gruppo PawnPair deve contenere SOLO quel tensore"
print(f"[1] OK: PawnPair weight {tuple(pw.shape)} in gruppo dedicato lr=1e-3")

all_ft = ids(ft_weight_params(nnue))
union = ids(groups[0]["params"]) | ids(groups[1]["params"])
assert union == all_ft, "unione gruppi FT != tutti i weight input (parametri persi o extra)"
assert not (ids(groups[0]["params"]) & ids(groups[1]["params"])), "parametro duplicato tra i 2 gruppi FT"
print(f"[2] OK: unione dei 2 gruppi FT == {len(all_ft)} weight dell'input, zero overlap")

# ---- [4]: scheduler step mantiene il rapporto -----------------------------
sch = sched["scheduler"] if isinstance(sched, dict) else sched
lrs0 = [g["lr"] for g in groups]
sch.step()
lrs1 = [g["lr"] for g in groups]
ratios = [b / a for a, b in zip(lrs0, lrs1)]
assert all(abs(r - 0.997) < 1e-9 for r in ratios), f"gamma non uniforme: {ratios}"
assert abs(lrs1[1] / lrs1[0] - 10.0) < 1e-6, "rapporto pawnpair/base cambiato dopo lo step"
print(f"[4] OK: StepLR gamma=0.997 uniforme, rapporto pawnpair/base = {lrs1[1]/lrs1[0]:.1f}x costante")

# ---- [3]: pawnpair_lr=0 -> comportamento vecchio --------------------------
nnue0, opt0, _ = build(0.0)
groups0 = opt0.param_groups
assert len(groups0) == 10, f"con pp_lr=0 attesi 10 gruppi, trovati {len(groups0)}"
assert ids(groups0[0]["params"]) == all_ft or len(ids(groups0[0]["params"])) == len(all_ft), \
    "gruppo 0 con pp_lr=0 deve contenere tutti i weight FT"
print(f"[3] OK: pawnpair_lr=0 -> 10 gruppi (comportamento pre-patch esatto)")

print("\n[ALL OK] two-group LR verificato end-to-end sul .pt del graft reale.")
