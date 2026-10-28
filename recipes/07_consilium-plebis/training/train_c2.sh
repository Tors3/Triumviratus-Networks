#!/usr/bin/env bash
# Consilium-2 full run (21/10/2026). Architecture: docs/rete/ARCHITETTURA_CONSILIUM2_2026-10-21.md (engine repo side)
#   features Full_Threats+HalfKAv2_hm_X8^+PP_3Wide+PassedPawns, L1 1024, pawn-king subnetwork 256, int8 HalfKA,
#   learned routing of the 8 experts (rounds after epochs 8,16,24,32,48,64; budget 0.20 expert changes per capture; initial table 0.209, today 0.251).
#
#   P  pretraining: batch NGPU x 131072, lr 0.4e-3 x sqrt(batch/131072), one-cycle, lambda 1.0 -> 0.75 over the first
#      30% of P then 0.75 fixed (lambda 0.75 is the project's reference; the 29/09 lesson: a cycle back to 1.0 stalled).
#   F  finetune as Consilium's F3, the phase that gave the Elo (P ep343 level with legio -> F3 +14.5/+24.7 at 90+0.9):
#      same batch as P, peak lr = 20% of P's, warmup 1%, lambda exactly 0.75; length set after P (author, 21/10).
#   Data: every .binpack of $DATA (701 GB BT4 + tb5dtm), drawn in proportion to size (each position weighs the same).
#   pc-y: endgames sampled more than Consilium (-0.20 -> 0.0 at 0 pieces, 0.45 -> 0.60 at 8 pieces), the rest unchanged.
#   Skipping: random 2 as Consilium (consecutive positions of one game are near-duplicates); no hard early skip; soft
#   early 20 made softer than Consilium's so the opening expert sees more openings (author, 21/10): kept fraction at
#   ply 0/6/10/18 = 0.25/0.40/0.60/0.90 (Consilium 0.10/0.15/0.25/0.75), all from ply 20. Quiet filters on.
#
# Usage (tmux or nohup): P_EPOCHS=500 nohup bash train_c2.sh > /root/train_c2.log 2>&1 &     (then F: F_EPOCHS=80 ...)
# Re-running resumes: P from its last.ckpt, F likewise; a finished phase is marked by $ROOT/P.done, $ROOT/F.done.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TRAINER="${TRAINER:-/root/c2/nnue-pytorch}"
DATA="${DATA:-/data/bt4}"
ROOT="${ROOT:-/root/c2/run}"
FEATURES="Full_Threats+HalfKAv2_hm_X8^+PP_3Wide+PassedPawns"
PY=""
for c in /venv/main/bin/python3 /usr/bin/python3 python3; do
  command -v "$c" >/dev/null 2>&1 && "$c" -c "import torch" >/dev/null 2>&1 && { PY="$c"; break; }
done
[ -n "$PY" ] || { echo "no python with torch"; exit 1; }
NGPU="${NGPU:-$(nvidia-smi -L | grep -c .)}"
BATCH="${BATCH:-$(( NGPU * 131072 ))}"
EPOCH_SIZE="${EPOCH_SIZE:-1000000000}"
P_EPOCHS="${P_EPOCHS:?P epochs of 1 G positions}"
F_EPOCHS="${F_EPOCHS:-0}"   # 0 = P only; F is set later (author, 21/10: after the cost of P is known)
STOP_P="${STOP_P:-0}"       # 1 = P was stopped by hand: mark it done and go on to F from its last checkpoint
# Workers: all the CPU the instance is given (cgroup quota when present: vast shows more threads than it assigns),
# minus 2 per rank for the training loop (author, 21/10: use the CPU to the maximum).
CPUS=$(awk '{ if ($1 != "max") print int($1 / $2) }' /sys/fs/cgroup/cpu.max 2>/dev/null)
CPUS="${CPUS:-$(nproc)}"
WORKERS="${WORKERS:-$(( CPUS / NGPU - 2 ))}"
export NNUE_WORKER_THREAD_RATIO="${NNUE_WORKER_THREAD_RATIO:-0.25}"
# 21/10/2026: the first validation (end of epoch 19) left the 24 GB 4090s ~1 GB short at the next training step
# (CUDA out of memory, run stopped at 10:20). Less fragmentation, and no validation inside the run (below).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
LR_P=$(python3 -c "print(f'{0.4e-3 * ($BATCH / 131072) ** 0.5:.4e}')")
# F_LR_FRAC: F's --lr as a fraction of P's peak (27/10: 0.25, the top of the author's NNUE_LR_PIECES schedule).
LR_F=$(python3 -c "print(f'{0.4e-3 * ($BATCH / 131072) ** 0.5 * ${F_LR_FRAC:-0.2}:.4e}')")
# +3%: a resume redoes the epoch stored in last.ckpt (trainer behaviour noted on 29/09); with exact steps OneCycleLR
# would stop with an error at the end of the phase after a resume.
P_STEPS=$(( P_EPOCHS * EPOCH_SIZE / BATCH * 103 / 100 ))
F_STEPS=$(( F_EPOCHS * EPOCH_SIZE / BATCH * 103 / 100 ))
LAMBDA_P=(--start-lambda 1.0 --end-lambda 0.75 --lambda-cycle-delta 0 --lambda-schedule-steps $(( P_STEPS * 3 / 10 )))
# lambda-schedule-steps 1: the jitter is scaled by the (zero) cycle factor after the schedule, so F trains at exactly
# 0.75 (without it the jitter would stay on, +-0.01 around 0.75). In P lambda reaches 0.75 at 30% and stays there
# (min(1, step/steps) in model/lambda_utils.py; cycle delta 0: it never goes back up, the 29/09 Consilium error).
LAMBDA_F=(--start-lambda 0.75 --end-lambda 0.75 --lambda-cycle-delta 0 --lambda-schedule-steps 1)

mapfile -t DATASETS < <(ls "$DATA"/*.binpack)
[ "${#DATASETS[@]}" -gt 0 ] || { echo "no binpack in $DATA"; exit 1; }
mkdir -p "$ROOT"
cd "$TRAINER"
echo "=== $(date -u) $NGPU GPU, ${#DATASETS[@]} binpacks ($(du -sh "$DATA" | cut -f1)), batch $BATCH, workers/rank $WORKERS,"\
     "P $P_EPOCHS ep ($P_STEPS steps, lr $LR_P), F $F_EPOCHS ep ($F_STEPS steps, lr $LR_F)"

COMMON=(--features "$FEATURES" --pawn-king-hidden 256 --l1 1024 --threads 1 --num-workers "$WORKERS"
  --accelerator cuda --epoch-size "$EPOCH_SIZE"
  --random-fen-skipping 2 --early-fen-skipping -1 --soft-early-fen-skipping 20
  --ply-x1 0 --ply-y1 0.25 --ply-x2 6 --ply-y2 0.40 --ply-x3 10 --ply-y3 0.60 --ply-x4 18 --ply-y4 0.90
  --pc-y0 0.0 --pc-y1 0.60 --pc-y2 1.0 --pc-y3 0.95 --pc-y4 0.75
  --lambda-cycle-jitter --jitter-lambda-sample 0.0035 --jitter-lambda-batch 0.0070 --jitter-decay-lambda-batch 0.999
  --one-cycle-final-div 1000
  --validation-size 1000000 --check-val-every-n-epoch 100000 --save-last-network True --seed 42
  --router-em-transitions "$TRAINER/c2_transitions.npz")

run() {  # $1 phase, $2 batch, $3 epochs, $4 lr, $5 steps, $6 warmup pct, rest = extra arguments
  local phase="$1" batch="$2" epochs="$3" lr="$4" steps="$5" warm="$6"; shift 6
  local dir="$ROOT/$phase"
  mkdir -p "$dir"
  local resume=() last
  last=$(find "$dir" -name last.ckpt -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
  [ -n "$last" ] && { resume=(--resume-from-checkpoint "$last"); echo "[$phase] resuming from $last"; }
  "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" ddp_launcher.py train.py \
    "${DATASETS[@]}" "${COMMON[@]}" \
    --batch-size "$batch" --max-epochs "$epochs" --lr "$lr" --one-cycle-steps "$steps" \
    --one-cycle-warmup-pct "$warm" \
    --network-save-period 10 --save-top-k -1 \
    --default-root-dir "$dir" "${resume[@]}" "$@"
}

( while true; do bash "$HERE/export_c2.sh" "$ROOT" >> "$ROOT/export.log" 2>&1; sleep 900; done ) &
EXPORTER=$!
trap 'kill $EXPORTER 2>/dev/null' EXIT

[ "$STOP_P" = 1 ] && touch "$ROOT/P.done"
if [ ! -f "$ROOT/P.done" ]; then
  run P "$BATCH" "$P_EPOCHS" "$LR_P" "$P_STEPS" 0.05 "${LAMBDA_P[@]}" \
    --router-em-epochs 8,16,24,32,48,64 --router-em-budget 0.20
  touch "$ROOT/P.done"
fi
if [ "$F_EPOCHS" -le 0 ]; then
  bash "$HERE/export_c2.sh" "$ROOT"
  echo "=== P finished $(date -u); F not set (F_EPOCHS=0)"
  exit 0
fi
P_LAST=$(find "$ROOT/P" -name last.ckpt -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
P_PT="$ROOT/P_final.pt"
[ -f "$P_PT" ] || "$PY" serialize.py "$P_LAST" "$P_PT" --features "$FEATURES" --pawn-king-hidden 256 --l1 1024
if [ ! -f "$ROOT/F.done" ]; then
  # --router-em-epochs 0: no routing round in F (epoch numbers start at 1), but the callback still loads the learned
  # table into the loader at train start; after a checkpoint resume the model is restored only after the loaders start.
  if find "$ROOT/F" -name last.ckpt -type f 2>/dev/null | grep -q .; then
    run F "$BATCH" "$F_EPOCHS" "$LR_F" "$F_STEPS" 0.01 "${LAMBDA_F[@]}" --router-em-epochs 0
  else
    run F "$BATCH" "$F_EPOCHS" "$LR_F" "$F_STEPS" 0.01 "${LAMBDA_F[@]}" --router-em-epochs 0 --resume-from-model "$P_PT"
  fi
  touch "$ROOT/F.done"
fi
bash "$HERE/export_c2.sh" "$ROOT"
echo "=== END $(date -u)"
