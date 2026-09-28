#!/usr/bin/env bash
# Training della rete MoE-1024 (28/09/2026): mix unico BT4, ricetta SFNNv16 scalata sul budget di 28 ore.
#
#   fase P (pretraining, ~95% delle posizioni): batch BATCH, one-cycle su tutti gli step, lambda con ciclo e jitter;
#   fase F (finetuning, ~5%): ripresa dalla rete di P (.pt) a batch dimezzato, stesse opzioni (come SF: 4500 -> 4750).
#
# USO (sulla VM, in nohup o tmux):
#   nohup ~/tt/TrainingMoE1024/train_moe1024.sh > ~/train_moe1024.log 2>&1 &
# Rilanciandolo riprende da solo: P da last.ckpt, F idem; P finita e' segnata da $ROOT/P.done.
#
# Throughput misurato il 28/09 (LOG_RUN.md): 4x5090, NNUE_WORKER_THREAD_RATIO 0.25 (lo 0.05 della legio lasciava un solo
# thread di feature per rank: 1,4M invece di 3,1-4,6M pos/s), 46 worker/rank, skip 3.
set -euo pipefail

TRAINER="${TRAINER:-/root/moe/nnue-pytorch}"
DATA="${DATA:-/data/bt4}"
ROOT="${ROOT:-/root/moe/run}"
FEATURES="Full_Threats+HalfKAv2_hm_P4^+PP_3Wide+PassedPawns"
HERE=$(cd "$(dirname "$0")" && pwd)

BATCH="${BATCH:-262144}"             # scelta dell'utente (28/09); F usa BATCH/2
EPOCH_SIZE="${EPOCH_SIZE:-1000000000}" # 1 G posizioni: ~4 minuti, last.ckpt (~2,7 GB) non troppo spesso
P_EPOCHS="${P_EPOCHS:?numero di epoche di P (1 G ciascuna): dal throughput misurato e dalle ore}"
F_EPOCHS="${F_EPOCHS:-$(( (P_EPOCHS + 19) / 20 ))}"   # ~5% di P
WORKERS="${WORKERS:-46}"
SKIP="${SKIP:-2}"   # 28/09, scelta dell'utente (skip 3 -> 4,79M, skip 0 -> 4,93M pos/s: differenze piccole)
# 28/09, decisione dell'utente: niente skip DURO delle aperture (SF usa hard 18 + soft 32), ma soft 20 (default del
# trainer): le aperture restano per l'esperto MoE >= 24 pezzi, campionate dal 10% alla semimossa 0 al 75% alla 18, cosi'
# le posizioni di libro ripetute in milioni di partite pesano meno.
EARLY_SKIP="${EARLY_SKIP:--1}"
SOFT_EARLY_SKIP="${SOFT_EARLY_SKIP:-20}"
# quota di thread di feature: 0.05 -> 1,4M (un thread per rank), 0.25 -> 4,62M, 0.15 -> 4,70M, 0.1 -> 4,79M (28/09,
# batch 524288, skip 3, 46 worker/rank)
export NNUE_WORKER_THREAD_RATIO="${NNUE_WORKER_THREAD_RATIO:-0.1}"
# lr: 0,4e-3 a batch 131072 nella ricetta SFNNv16, scalata con sqrt(batch/131072)
LR_P=$(python3 -c "print(f'{0.4e-3 * ($BATCH / 131072) ** 0.5:.4e}')")
LR_F=$(python3 -c "print(f'{0.4e-3 * ($BATCH / 2 / 131072) ** 0.5:.4e}')")

PY=""
for c in /venv/main/bin/python3 /usr/bin/python3 python3; do
  command -v "$c" >/dev/null 2>&1 && "$c" -c "import torch" >/dev/null 2>&1 && { PY="$c"; break; }
done
[ -n "$PY" ] || { echo "nessun python con torch"; exit 1; }
NGPU="${NGPU:-$(nvidia-smi -L | grep -c .)}"

# 🔴 Il loader sceglie il FILE uniformemente a caso, non in proporzione alla dimensione (parallel_dataloader.h): i
# binpack piccoli verrebbero riletti molte volte. Sotto MIN_GB si escludono (vedi Training70 train_phase2.sh).
MIN_GB="${MIN_BINPACK_GB:-5}"
mapfile -t DATASETS < <(for f in "$DATA"/*.binpack; do
  sz=$(stat -c%s "$f"); if [ "$sz" -ge $((MIN_GB * 1000000000)) ]; then echo "$f"
  else echo "  [escluso: $((sz / 1000000000)) GB < $MIN_GB] $(basename "$f")" >&2; fi; done)
[ "${#DATASETS[@]}" -gt 0 ] || { echo "nessun binpack in $DATA"; exit 1; }

mkdir -p "$ROOT"
cd "$TRAINER"
echo "=== $(date -u)  interprete $PY, $NGPU GPU, ${#DATASETS[@]} binpack, ratio $NNUE_WORKER_THREAD_RATIO, $WORKERS worker/rank"

COMMON=(--features "$FEATURES" --l1 1024 --threads 1 --num-workers "$WORKERS" --accelerator cuda
  --epoch-size "$EPOCH_SIZE" --random-fen-skipping "$SKIP"
  --early-fen-skipping "$EARLY_SKIP" --soft-early-fen-skipping "$SOFT_EARLY_SKIP"
  --pc-y0 -0.20 --pc-y1 0.45 --pc-y2 1.0 --pc-y3 0.95 --pc-y4 0.75
  --start-lambda 1.0 --end-lambda 1.0 --lambda-cycle-delta -0.3 --lambda-cycle-warmup-pct 0.25 --lambda-cycle-jitter
  --jitter-lambda-sample 0.0035 --jitter-lambda-batch 0.0070 --jitter-decay-lambda-batch 0.999
  --one-cycle-warmup-pct 0.05 --one-cycle-final-div 1000
  --validation-size 1000000 --check-val-every-n-epoch 20 --save-last-network True --seed 42)

run() {  # $1 = fase, $2 = batch, $3 = epoche, $4 = lr, resto = argomenti extra
  local phase="$1" batch="$2" epochs="$3" lr="$4"; shift 4
  local dir="$ROOT/$phase" steps=$(( epochs * EPOCH_SIZE / batch ))
  mkdir -p "$dir"
  local resume=() last
  last=$(find "$dir" -name last.ckpt -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
  [ -n "$last" ] && { resume=(--resume-from-checkpoint "$last"); echo "[$phase] RIPRESA da $last"; }
  echo "[$phase] batch $batch, $epochs epoche da $EPOCH_SIZE, $steps step, lr $lr"
  "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" ddp_launcher.py train.py \
    "${DATASETS[@]}" "${COMMON[@]}" \
    --batch-size "$batch" --max-epochs "$epochs" --lr "$lr" \
    --one-cycle-steps "$steps" --lambda-schedule-steps "$steps" \
    --network-save-period $(( epochs / 10 > 0 ? epochs / 10 : 1 )) --save-top-k -1 \
    --default-root-dir "$dir" "${resume[@]}" "$@"
}

# export .nnue di ogni checkpoint nuovo, in background per tutto il run
( while true; do "$HERE/export_nets.sh" "$ROOT" >> "$ROOT/export.log" 2>&1; sleep 900; done ) &
EXPORTER=$!
trap 'kill $EXPORTER 2>/dev/null' EXIT

if [ ! -f "$ROOT/P.done" ]; then
  run P "$BATCH" "$P_EPOCHS" "$LR_P"
  touch "$ROOT/P.done"
fi

P_LAST=$(find "$ROOT/P" -name last.ckpt -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
P_PT="$ROOT/P_final.pt"
[ -f "$P_PT" ] || "$PY" serialize.py "$P_LAST" "$P_PT" --features "$FEATURES" --l1 1024
if [ ! -f "$ROOT/F.done" ]; then
  if find "$ROOT/F" -name last.ckpt -type f 2>/dev/null | grep -q .; then
    run F $(( BATCH / 2 )) "$F_EPOCHS" "$LR_F"
  else
    run F $(( BATCH / 2 )) "$F_EPOCHS" "$LR_F" --resume-from-model "$P_PT"
  fi
  touch "$ROOT/F.done"
fi
"$HERE/export_nets.sh" "$ROOT"
echo "=== FINE $(date -u)"
