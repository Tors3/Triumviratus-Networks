#!/usr/bin/env bash
# Fase F4 della MoE-1024 (29/09/2026, decisione dell'utente): secondo piccolo riavvio dopo la F3, per usare il credito
# vast.ai rimasto (~15,8 $ alle 19:00 locali, ~5 h a 2,98 $/h).
#
# Perche': a fine F3 (lr ~0) la loss di VALIDAZIONE, confrontabile perche' lambda e' fisso a 0,75, scendeva ancora e
# sempre piu' in fretta (ep 49 0,00342, 59 0,00340, 69 0,00337): la rete non e' satura. Un secondo ciclo corto le da'
# un altro po' di spazio e poi lo incassa: 60 epoche, batch 524288, picco 6e-5 (~40% del picco F3, ~7,5% di P),
# riscaldamento 1%, coseno fino a /1000, lambda 0,75. ~3,3 h: resta ~1,5 h di credito per esportare e scaricare.
# Salva ogni 10 epoche: se il credito finisse prima, restano le reti intermedie (esportate ogni 15 minuti).
#
# Parte da sola alla fine della F3: WAIT_PID = pid dello script F3; poi serializza F3/last.ckpt in F3_final.pt.
# USO: WAIT_PID=<pid F3> nohup ~/tt/TrainingMoE1024/train_moe1024_F4.sh >> ~/train_moe1024.log 2>&1 &
set -euo pipefail

TRAINER="${TRAINER:-/root/moe/nnue-pytorch}"
DATA="${DATA:-/data/bt4}"
ROOT="${ROOT:-/root/moe/run}"
FEATURES="Full_Threats+HalfKAv2_hm_P4^+PP_3Wide+PassedPawns"
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/venv/main/bin/python3

BATCH="${F4_BATCH:-524288}"
EPOCHS="${F4_EPOCHS:-60}"
LR="${F4_LR:-6e-5}"
WARMUP="${F4_WARMUP:-0.01}"
EPOCH_SIZE=1000000000
WORKERS=46
export NNUE_WORKER_THREAD_RATIO=0.1

if [ -n "${WAIT_PID:-}" ]; then
  echo "=== $(date -u)  F4: aspetto la fine della F3 (pid $WAIT_PID)"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi
[ -f "$ROOT/F3.done" ] || { echo "F4: manca F3.done (F3 non finita?)"; exit 1; }

FROM="$ROOT/F3_final.pt"
if [ ! -f "$FROM" ]; then
  L=$(find "$ROOT/F3" -name last.ckpt -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
  cd "$TRAINER"
  CUDA_VISIBLE_DEVICES="" "$PY" serialize.py "$L" "$FROM" --features "$FEATURES" --l1 1024
fi

MIN_GB=5
mapfile -t DATASETS < <(for f in "$DATA"/*.binpack; do
  sz=$(stat -c%s "$f"); [ "$sz" -ge $((MIN_GB * 1000000000)) ] && echo "$f"; done)
NGPU=$(nvidia-smi -L | grep -c .)
STEPS=$(( EPOCHS * EPOCH_SIZE / BATCH ))
DIR="$ROOT/F4"; mkdir -p "$DIR"
resume=(--resume-from-model "$FROM")
last=$(find "$DIR" -name last.ckpt -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
[ -n "$last" ] && { resume=(--resume-from-checkpoint "$last"); echo "[F4] RIPRESA da $last"; }

( while true; do "$HERE/export_nets.sh" "$ROOT" >> "$ROOT/export.log" 2>&1; sleep 900; done ) &
EXPORTER=$!
trap 'kill $EXPORTER 2>/dev/null' EXIT

cd "$TRAINER"
echo "=== $(date -u)  [F4] da $FROM, $NGPU GPU, ${#DATASETS[@]} binpack, batch $BATCH, $EPOCHS epoche, $STEPS step, lr $LR, warmup $WARMUP"
"$PY" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" ddp_launcher.py train.py \
  "${DATASETS[@]}" \
  --features "$FEATURES" --l1 1024 --threads 1 --num-workers "$WORKERS" --accelerator cuda \
  --epoch-size "$EPOCH_SIZE" --random-fen-skipping 2 --early-fen-skipping -1 --soft-early-fen-skipping 20 \
  --pc-y0 -0.20 --pc-y1 0.45 --pc-y2 1.0 --pc-y3 0.95 --pc-y4 0.75 \
  --start-lambda 0.75 --end-lambda 0.75 --lambda-cycle-delta 0 \
  --one-cycle-warmup-pct "$WARMUP" --one-cycle-final-div 1000 \
  --batch-size "$BATCH" --max-epochs "$EPOCHS" --lr "$LR" --one-cycle-steps "$STEPS" \
  --validation-size 1000000 --check-val-every-n-epoch 10 --save-last-network True --seed 43 \
  --network-save-period 10 --save-top-k -1 \
  --default-root-dir "$DIR" "${resume[@]}"
touch "$ROOT/F4.done"
"$HERE/export_nets.sh" "$ROOT"
echo "=== $(date -u)  F4 FINITA"
