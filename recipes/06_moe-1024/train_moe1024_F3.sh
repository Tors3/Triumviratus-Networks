#!/usr/bin/env bash
# Fase F3 della MoE-1024 (29/09/2026, decisione dell'utente): sostituisce la coda di P (epoche 370-449) e la F2.
#
# Perche': all'epoca 369 l'lr di P era gia' all'8,7% del picco (~6,9e-5) e la F2 prevista (picco 6,5e-5) avrebbe
# rifatto lo stesso raffreddamento due volte. Il problema della rete non e' "rifinire": ha passato ~180 epoche
# (112-292) a imparare verso un lambda che risaliva a 1,0, e dopo la correzione (lambda 0,75) la loss sul risultato
# delle partite e' ripresa a scendere finche' l'lr era al 13-16% (18,53 -> 18,30 su dati mai visti). F3 riapre
# un po' la rete e la raffredda di nuovo fino a zero, come la fase 2 di legio ma corta:
#   80 epoche, batch 524288, one-cycle con picco 1,6e-4 (20% del picco di P), riscaldamento minimo (1%, ~1 epoca:
#   --resume-from-model riparte con l'ottimizzatore azzerato e un salto secco al picco sarebbe a momenti nulli),
#   coseno fino a /1000, lambda 0,75 fisso. Ripresa dai PESI dell'ultimo checkpoint di P (P_ep369.pt).
# USO: nohup ~/tt/TrainingMoE1024/train_moe1024_F3.sh >> ~/train_moe1024.log 2>&1 &
# Rilanciandolo riprende da F3/last.ckpt (difetto noto: rifa' l'epoca salvata, vedi train_moe1024.sh).
set -euo pipefail

TRAINER="${TRAINER:-/root/moe/nnue-pytorch}"
DATA="${DATA:-/data/bt4}"
ROOT="${ROOT:-/root/moe/run}"
FEATURES="Full_Threats+HalfKAv2_hm_P4^+PP_3Wide+PassedPawns"
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/venv/main/bin/python3

FROM="${F3_FROM:?pesi di partenza: .pt serializzato dall ultimo checkpoint di P}"
BATCH="${F3_BATCH:-524288}"
EPOCHS="${F3_EPOCHS:-80}"
LR="${F3_LR:-1.6e-4}"
WARMUP="${F3_WARMUP:-0.01}"
EPOCH_SIZE=1000000000
WORKERS=46
export NNUE_WORKER_THREAD_RATIO=0.1

[ -f "$FROM" ] || { echo "F3: manca $FROM"; exit 1; }
MIN_GB=5
mapfile -t DATASETS < <(for f in "$DATA"/*.binpack; do
  sz=$(stat -c%s "$f"); [ "$sz" -ge $((MIN_GB * 1000000000)) ] && echo "$f"; done)
NGPU=$(nvidia-smi -L | grep -c .)
STEPS=$(( EPOCHS * EPOCH_SIZE / BATCH ))
DIR="$ROOT/F3"; mkdir -p "$DIR"
resume=(--resume-from-model "$FROM")
last=$(find "$DIR" -name last.ckpt -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
[ -n "$last" ] && { resume=(--resume-from-checkpoint "$last"); echo "[F3] RIPRESA da $last"; }

# export .nnue dei checkpoint periodici durante il run
( while true; do "$HERE/export_nets.sh" "$ROOT" >> "$ROOT/export.log" 2>&1; sleep 900; done ) &
EXPORTER=$!
trap 'kill $EXPORTER 2>/dev/null' EXIT

cd "$TRAINER"
echo "=== $(date -u)  [F3] da $FROM, $NGPU GPU, ${#DATASETS[@]} binpack, batch $BATCH, $EPOCHS epoche, $STEPS step, lr $LR, warmup $WARMUP"
"$PY" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" ddp_launcher.py train.py \
  "${DATASETS[@]}" \
  --features "$FEATURES" --l1 1024 --threads 1 --num-workers "$WORKERS" --accelerator cuda \
  --epoch-size "$EPOCH_SIZE" --random-fen-skipping 2 --early-fen-skipping -1 --soft-early-fen-skipping 20 \
  --pc-y0 -0.20 --pc-y1 0.45 --pc-y2 1.0 --pc-y3 0.95 --pc-y4 0.75 \
  --start-lambda 0.75 --end-lambda 0.75 --lambda-cycle-delta 0 \
  --one-cycle-warmup-pct "$WARMUP" --one-cycle-final-div 1000 \
  --batch-size "$BATCH" --max-epochs "$EPOCHS" --lr "$LR" --one-cycle-steps "$STEPS" \
  --validation-size 1000000 --check-val-every-n-epoch 10 --save-last-network True --seed 42 \
  --network-save-period 10 --save-top-k -1 \
  --default-root-dir "$DIR" "${resume[@]}"
touch "$ROOT/F3.done"
"$HERE/export_nets.sh" "$ROOT"
echo "=== $(date -u)  F3 FINITA"
