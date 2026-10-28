#!/usr/bin/env bash
# Consilium Plebis F, second cycle (author, 27/10/2026), resumed from F's final checkpoint. In % of P's peak: from
# P's final LR up to 10% in 2 epochs (linear), half-cosine down to 4% in 3, flat 4% for 5, half-cosine down to P's
# final LR in 15: 25 epochs. In fractions of --lr (25% of P's peak): 0.4, 0.16, 0.00016. The scheduler keeps counting
# from F's 228,840 steps, so F's four segments stay in front of the new ones.
# A resume repeats the epoch label of its checkpoint while the step count goes on: max epochs = checkpoint epoch +
# remaining steps / 3814.
# Launch only on the author's order: nohup bash /root/valfix/start_cycle2.sh >> /root/train_c2.log 2>&1 < /dev/null &
set -euo pipefail
ROOT=/root/c2/run; V=/root/valfix; PY=/venv/main/bin/python3
F_STEPS_DONE=228840
TOTAL=$(( F_STEPS_DONE + 7628 + 11442 + 19070 + 57210 ))
[ -f "$ROOT/F.done" ] || { echo "F has not finished"; exit 1; }
[ -f "$V/F_final.ckpt" ] || { echo "no copy of F's final checkpoint (after_F.sh)"; exit 1; }
CK=$(find "$ROOT/F" -name last.ckpt -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
read -r EP GS < <(cd /root/c2/nnue-pytorch && CUDA_VISIBLE_DEVICES="" "$PY" -c "
import torch
c = torch.load('$CK', map_location='cpu', weights_only=False)
print(c['epoch'], c['global_step'])")
[ "$GS" = "$F_STEPS_DONE" ] || { echo "last.ckpt is at step $GS, not $F_STEPS_DONE"; exit 1; }
mv "$ROOT/F.done" "$ROOT/F.done.cycle1"
export P_EPOCHS=550 F_LR_FRAC=0.25
export F_EPOCHS=$(( EP + (TOTAL - GS) / 3814 ))
export NNUE_LR_PIECES="7628:0.1494:1:lin;87722:1:0.6:cos;38140:0.6:0.6:lin;95350:0.6:0.00016:cos;7628:0.00016:0.4:lin;11442:0.4:0.16:cos;19070:0.16:0.16:lin;57210:0.16:0.00016:cos"
echo "=== F CYCLE 2 $(date -u) checkpoint epoch $EP step $GS, F_EPOCHS=$F_EPOCHS NNUE_LR_PIECES=$NNUE_LR_PIECES"
cd /root/c2pkg
exec bash /root/c2pkg/train_c2.sh
