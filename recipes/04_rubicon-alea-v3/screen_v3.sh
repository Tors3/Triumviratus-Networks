#!/usr/bin/env bash
# screen_v3.sh — SCREENING FROZEN-BASE del blocco PassedPawns (rubicon-alea-v3).
# USATO PER LA v3 (2026-07-17). Nato come sonda go/no-go economica: si e' rivelato
# essere IL training (il blocco satura in ~4 epoche).
#
# IDEA: la base pre-allenata (v2) e' CONGELATA (--lr 0) e allena SOLO il blocco
# nuovo (--passedpawns-lr 1e-3). Costa poche ore e dice se la feature porta segnale
# PRIMA di investire in un fine-tune two-group completo.
#
# ESITO: gate v3(ep4) vs v2 = +6.96±6.56 (LOS 98.1%, 2596 partite) -> feature
# CONFERMATA. Ma ep9 (parita', -0.32±10.29) ed ep14 (non meglio) mostrano che il
# blocco (99k param, 50x piu' piccolo del PawnPair) SATURA a ~ep4: il train pieno
# non e' mai servito. ep4 = la v3 spedita.
#
# ⚠️ --lr 0 congela davvero (Adam: step = lr * m_hat -> 0). Se il trainer si
# lamentasse di lr=0, usare --lr 1e-9 (equivalente in pratica).
#
# USO:  bash gcp/screen_v3.sh blackwell /home/USER/data
# Variante ARRICCHITA (esperimento, vedi README): aggiungere --min-passed-pawns 1
# per allenare/validare SOLO su posizioni con pedoni passati. Provato: NON migliora
# ep4 (il blocco era gia' convergiuto), ma il filtro funziona e costa solo ~3%.
set -euo pipefail
GPU=${1:?l4|blackwell}; DATA=${2:?data dir}
OWN_REPEAT=${OWN_REPEAT:-10}
cd "$(dirname "$0")/.."; source venv/bin/activate; export TORCHDYNAMO_DISABLE=1
RUNDIR="$(pwd)/runs/alea_v3_screen"
cd nnue-pytorch
case "$GPU" in l4) BATCH=32768;WORKERS=8;; blackwell) BATCH=65536;WORKERS=12;; *) echo "gpu sconosciuta: $GPU";exit 1;; esac
mapfile -t T80 < <(find "$DATA/t80" -name "*.binpack"|sort)
mapfile -t OWN < <(find "$DATA/own" -name "*.binpack"|sort)
[ "${#T80[@]}" -gt 0 ] && [ "${#OWN[@]}" -gt 0 ] || { echo "!! binpack mancanti"; exit 1; }
BINPACKS=("${T80[@]}"); for _ in $(seq 1 "$OWN_REPEAT"); do BINPACKS+=("${OWN[@]}"); done
echo "[i] ${#BINPACKS[@]} binpack | SCREENING frozen-base: lr 0 (base congelata), passedpawns-lr 1e-3"
mkdir -p "$RUNDIR"
python train.py "${BINPACKS[@]}" \
  --features "Full_Threats+HalfKAv2_hm^+PawnPair+PassedPawns" \
  --resume-from-model ../graft/alea_v3_grafted.pt \
  --max-epochs 30 --epoch-size 100000000 --batch-size $BATCH --num-workers $WORKERS \
  --lr 0 --passedpawns-lr 1e-3 --gamma 0.99 \
  --start-lambda 0.75 --end-lambda 0.75 --validation-size 1000000 \
  --network-save-period 5 --save-top-k -1 --default-root-dir "$RUNDIR" \
  2>&1 | tee -a "$RUNDIR.log"
# All'avvio DEVE stampare: [configure_optimizers] PassedPawns block lr=0.001 (base lr=0.0)
# Se non lo stampa, la base NON e' congelata e lo screening non isola la feature.
