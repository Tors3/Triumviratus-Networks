#!/usr/bin/env bash
# Tail of F's second cycle (author, 27/10/2026): from the second cycle's final checkpoint, 5 epochs at a fixed LR of
# 0.004% of P's peak (the value the cooling ends at, and P's own one-cycle floor, lr/25000), one checkpoint per epoch
# kept; the final network is the average of these five epochs (avg_ckpt.py, after_tail.sh).
# Run by after_tail.sh once the second cycle has exited and its final checkpoint is copied (after_cycle2.sh).
set -euo pipefail
ROOT=/root/c2/run; V=/root/valfix; PY=/venv/main/bin/python3
C2_STEPS_DONE=324190
TAIL_STEPS=19070
[ -f "$ROOT/F.done" ] || { echo "the second cycle has not finished"; exit 1; }
[ -f "$V/F2_final_exported" ] || { echo "no export of the second cycle's final checkpoint (after_cycle2.sh)"; exit 1; }
CK=$(find "$ROOT/F" -name last.ckpt -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
read -r EP GS < <(CUDA_VISIBLE_DEVICES="" "$PY" -c "
import torch
c = torch.load('$CK', map_location='cpu', weights_only=False)
print(c['epoch'], c['global_step'])")
[ "$GS" = "$C2_STEPS_DONE" ] || { echo "last.ckpt is at step $GS, not $C2_STEPS_DONE"; exit 1; }
mv "$ROOT/F.done" "$ROOT/F.done.cycle2"
export P_EPOCHS=550 F_LR_FRAC=0.25 SAVE_PERIOD=1 EXPORT_SH=export_c2_keep.sh
export F_EPOCHS=$(( EP + TAIL_STEPS / 3814 ))
export NNUE_LR_PIECES="7628:0.1494:1:lin;87722:1:0.6:cos;38140:0.6:0.6:lin;95350:0.6:0.00016:cos;7628:0.00016:0.4:lin;11442:0.4:0.16:cos;19070:0.16:0.16:lin;57210:0.16:0.00016:cos;19070:0.00016:0.00016:lin"
echo "=== F TAIL $(date -u) checkpoint epoch $EP step $GS, F_EPOCHS=$F_EPOCHS NNUE_LR_PIECES=$NNUE_LR_PIECES"
cd /root/c2pkg; exec bash /root/c2pkg/train_c2_tail.sh
