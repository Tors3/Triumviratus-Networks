#!/usr/bin/env bash
# train_v5.sh — SCREENING FROZEN-BASE del blocco CandidatePassers (v4-slot).
#
# SOLO FROZEN (lezione della sessione Outposts: il co-adapt su questa rete
# danneggia-e-pareggia): base CONGELATA (--lr 0; se il trainer protesta per
# lr=0 usare 1e-9), impara SOLO il blocco nuovo a 1e-3, gamma 0.99.
# Su v3 (PassedPawns) il segnale c'era gia' a ep3-4: go/no-go = gate ep3-ep5
# vs zerograft, ~2000 partite 12+0.12.
#
# ⚠️ FISSI BY DESIGN (richiesta esplicita utente, NON toccare):
#   - lambda: --start-lambda 0.75 --end-lambda 0.75 (COSTANTE, nessuno schedule)
#   - segnale WDL: NESSUN flag WDL esplicito = default del trainer invariato,
#     esattamente come i run v2/v3. Nessuna variazione del mix score/risultato.
#
# USO:  bash Training_NNUE/TrainingV5_CandidatePassers/train_v5.sh blackwell /home/USER/data
# Prerequisiti sulla VM: trainer patchato (candidate_passers.py + __init__ +
# optimizers/config + lightning_module + training_data_loader.cpp ricompilato)
# e graft gia' fatto:
#   python .../graft_candidates.py <v3_plain.nnue> .../alea_v5_grafted.pt
set -euo pipefail
GPU=${1:?l4|blackwell}; DATA=${2:?data dir}
OWN_REPEAT=${OWN_REPEAT:-10}
# WORKDIR = la cartella che contiene venv/ e nnue-pytorch/ (sulla VM di solito
# la copia di lavoro tipo ~/alea; override con WORKDIR=... se serve)
WORKDIR="${WORKDIR:-$(dirname "$0")/../TrainingAleaV2_Grafting}"
cd "$WORKDIR"; source venv/bin/activate; export TORCHDYNAMO_DISABLE=1
RUNDIR="$(pwd)/runs/alea_v5_screen"
GRAFT_PT="${GRAFT_PT:-../TrainingV5_CandidatePassers/graft/alea_v5_grafted.pt}"
cd nnue-pytorch
case "$GPU" in l4) BATCH=32768;WORKERS=8;; blackwell) BATCH=65536;WORKERS=12;; *) echo "gpu sconosciuta: $GPU";exit 1;; esac
mapfile -t T80 < <(find "$DATA/t80" -name "*.binpack"|sort)
mapfile -t OWN < <(find "$DATA/own" -name "*.binpack"|sort)
[ "${#T80[@]}" -gt 0 ] && [ "${#OWN[@]}" -gt 0 ] || { echo "!! binpack mancanti"; exit 1; }
BINPACKS=("${T80[@]}"); for _ in $(seq 1 "$OWN_REPEAT"); do BINPACKS+=("${OWN[@]}"); done
echo "[i] ${#BINPACKS[@]} binpack | frozen-base: lr 0 (base congelata), candidatepassers-lr 1e-3, gamma 0.99"
mkdir -p "$RUNDIR"
python train.py "${BINPACKS[@]}" \
  --features "Full_Threats+HalfKAv2_hm^+PawnPair+PassedPawns+CandidatePassers" \
  --resume-from-model "$GRAFT_PT" \
  --max-epochs 12 --epoch-size 100000000 --batch-size $BATCH --num-workers $WORKERS \
  --lr 0 --candidatepassers-lr 1e-3 --gamma 0.99 \
  --start-lambda 0.75 --end-lambda 0.75 --validation-size 1000000 \
  --network-save-period 2 --save-top-k -1 --default-root-dir "$RUNDIR" \
  2>&1 | tee -a "$RUNDIR.log"
# All'avvio DEVE stampare: [configure_optimizers] CandidatePassers block lr=0.001 (base lr=0.0)
# Se non lo stampa, il two-group non e' attivo e il blocco nuovo non impara.
# Se il trainer rifiuta --lr 0, rilanciare con --lr 1e-9 (base di fatto ferma).
#
# GO/NO-GO (senza aspettare le 12 epoche):
#   1. serializza ep3 e ep5 plain (NO ft_optimize):
#      python serialize.py <ckpt> vX.nnue --features "Full_Threats+HalfKAv2_hm^+PawnPair+PassedPawns+CandidatePassers"
#   2. gate net-isolated vs zerograft (STESSO binario tri-format su entrambi
#      i lati, ponder OFF), ~2000 partite 12+0.12 -> su v3 il segnale c'era
#      gia' a ep3-4; piatto a ep5 = feature scartata.
#   ⚠️ fastchess: opzioni spin con =1/=0, MAI true/false.
