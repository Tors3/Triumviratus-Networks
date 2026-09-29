#!/usr/bin/env bash
# Fase F "dolce" della MoE-1024 (29/09/2026, decisione dell'utente), al posto della F di train_moe1024.sh.
#
# Perche': la F della ricetta SFNNv16 riparte con un nuovo one-cycle a 5,66e-4 (70% del picco di P) subito dopo che
# P ha portato l'lr a ~0: un riscaldamento forte che sbalza la rete dal minimo di P. Qui invece:
#   30 epoche (non 22), batch 262144, picco 6,5e-5 (~8% del picco di P), riscaldamento lento sul 10% (3 epoche)
#   e coseno fino a /1000; lambda 0,75 FISSO come la fine di P; ripresa dai pesi finali di P (ottimizzatore nuovo).
#
# Come si innesta senza toccare lo script in corso (bash lo legge a pezzi, modificarlo mentre gira lo rompe):
#   - $ROOT/F.done creato a mano -> train_moe1024.sh, finita P, serializza P_final.pt, salta la sua F, esporta, esce;
#   - questo script, lanciato con WAIT_PID=<pid di train_moe1024.sh>, aspetta quel pid e poi parte.
# USO: WAIT_PID=46042 nohup ~/tt/TrainingMoE1024/train_moe1024_F2.sh >> ~/train_moe1024.log 2>&1 &
# Rilanciandolo riprende da solo da F2/last.ckpt (con il difetto noto: rifa' l'epoca salvata, vedi train_moe1024.sh).
set -euo pipefail

TRAINER="${TRAINER:-/root/moe/nnue-pytorch}"
DATA="${DATA:-/data/bt4}"
ROOT="${ROOT:-/root/moe/run}"
FEATURES="Full_Threats+HalfKAv2_hm_P4^+PP_3Wide+PassedPawns"
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/venv/main/bin/python3

BATCH=262144
EPOCHS="${F2_EPOCHS:-30}"
LR="${F2_LR:-6.5e-5}"
EPOCH_SIZE=1000000000
WORKERS=46
export NNUE_WORKER_THREAD_RATIO=0.1

if [ -n "${WAIT_PID:-}" ]; then
  echo "=== $(date -u)  F2: aspetto la fine di train_moe1024.sh (pid $WAIT_PID)"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi

P_PT="$ROOT/P_final.pt"
[ -f "$P_PT" ] || { echo "F2: manca $P_PT (P non finita?)"; exit 1; }
[ -f "$ROOT/P.done" ] || { echo "F2: manca P.done"; exit 1; }

MIN_GB=5
mapfile -t DATASETS < <(for f in "$DATA"/*.binpack; do
  sz=$(stat -c%s "$f"); [ "$sz" -ge $((MIN_GB * 1000000000)) ] && echo "$f"; done)
NGPU=$(nvidia-smi -L | grep -c .)
STEPS=$(( EPOCHS * EPOCH_SIZE / BATCH ))
DIR="$ROOT/F2"; mkdir -p "$DIR"
resume=(--resume-from-model "$P_PT")
last=$(find "$DIR" -name last.ckpt -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
[ -n "$last" ] && { resume=(--resume-from-checkpoint "$last"); echo "[F2] RIPRESA da $last"; }

cd "$TRAINER"
echo "=== $(date -u)  [F2] $NGPU GPU, ${#DATASETS[@]} binpack, batch $BATCH, $EPOCHS epoche, $STEPS step, lr $LR"
"$PY" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" ddp_launcher.py train.py \
  "${DATASETS[@]}" \
  --features "$FEATURES" --l1 1024 --threads 1 --num-workers "$WORKERS" --accelerator cuda \
  --epoch-size "$EPOCH_SIZE" --random-fen-skipping 2 --early-fen-skipping -1 --soft-early-fen-skipping 20 \
  --pc-y0 -0.20 --pc-y1 0.45 --pc-y2 1.0 --pc-y3 0.95 --pc-y4 0.75 \
  --start-lambda 0.75 --end-lambda 0.75 --lambda-cycle-delta 0 \
  --one-cycle-warmup-pct 0.10 --one-cycle-final-div 1000 \
  --batch-size "$BATCH" --max-epochs "$EPOCHS" --lr "$LR" --one-cycle-steps "$STEPS" \
  --validation-size 1000000 --check-val-every-n-epoch 5 --save-last-network True --seed 42 \
  --network-save-period 5 --save-top-k -1 \
  --default-root-dir "$DIR" "${resume[@]}"
touch "$ROOT/F2.done"
"$HERE/export_nets.sh" "$ROOT"
echo "=== $(date -u)  F2 FINITA"
