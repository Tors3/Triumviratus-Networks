#!/usr/bin/env bash
# Consilium-2 speed probe (21/10/2026): the real run's settings (features, pawn-king, workers, loader ratio) for a
# short epoch at each batch size; prints positions per second measured between the 40% and 100% progress lines
# (after the torch.compile warm-up). No checkpoints, no routing rounds.
# Usage: bash probe_c2.sh [batch ...]        (default: 524288 1048576)
set -uo pipefail
TRAINER="${TRAINER:-/root/c2/nnue-pytorch}"
DATA="${DATA:-/data/bt4}"
PY=/venv/main/bin/python3
NGPU=$(nvidia-smi -L | grep -c .)
WORKERS="${WORKERS:-$(( $(nproc) / NGPU - 2 ))}"
export NNUE_WORKER_THREAD_RATIO="${NNUE_WORKER_THREAD_RATIO:-0.1}"
mapfile -t DATASETS < <(ls "$DATA"/*.binpack)
cd "$TRAINER"
for B in "${@:-524288 1048576}"; do
  for b in $B; do
    STEPS=100
    LOG=/root/probe_$b.log
    rm -rf /root/c2/probe
    "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" ddp_launcher.py train.py "${DATASETS[@]}" \
      --features "Full_Threats+HalfKAv2_hm_X8^+PP_3Wide+PassedPawns" --pawn-king-hidden 256 --l1 1024 --threads 1 \
      --num-workers "$WORKERS" --accelerator cuda --batch-size "$b" --epoch-size $(( b * STEPS )) --max-epochs 1 \
      --lr 8e-4 --one-cycle-steps 1000 --random-fen-skipping 0 --early-fen-skipping -1 --soft-early-fen-skipping 20 \
      --pc-y0 0.0 --pc-y1 0.60 --pc-y2 1.0 --pc-y3 0.95 --pc-y4 0.75 --start-lambda 0.75 --end-lambda 0.75 \
      --lambda-cycle-delta 0 --network-save-period 1000 --save-last-network False --validation-size 65536 \
      --check-val-every-n-epoch 1000 --default-root-dir /root/c2/probe > "$LOG" 2>&1
    # progress lines: "NN/100 [mm:ss<..." -> seconds at 40% and at 100%
    t40=$(grep -o " 40/$STEPS *\[[0-9:]*" "$LOG" | tail -1 | grep -o "[0-9]*:[0-9]*$" | awk -F: '{print $1*60+$2}')
    t100=$(grep -o "$STEPS/$STEPS *\[[0-9:]*" "$LOG" | tail -1 | grep -o "[0-9]*:[0-9]*$" | awk -F: '{print $1*60+$2}')
    if [ -n "$t40" ] && [ -n "$t100" ] && [ "$t100" -gt "$t40" ]; then
      echo "batch $b: $(( 60 * b / (t100 - t40) / 1000 )) k positions/s (60 steps in $(( t100 - t40 )) s), workers/rank $WORKERS"
    else
      echo "batch $b: no measure (see $LOG)"; tail -5 "$LOG"
    fi
  done
done
