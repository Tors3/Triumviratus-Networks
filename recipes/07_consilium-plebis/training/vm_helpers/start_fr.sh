#!/usr/bin/env bash
# Restart from F's final checkpoint (author, 27/10/2026 evening, after the second cycle was stopped): one epoch of
# linear rise from 0.004% to 0.5% of P's peak, then 10 epochs of half-cosine back to 0.004%, one checkpoint per epoch
# kept (train_c2_tail.sh with SAVE_PERIOD=1 and export_c2_keep.sh). The author's reading: the loss was still falling at
# the end of F, so a small LR lets it go on falling, while a large one (the second cycle's 10%) lifts it. 0.5% is the
# LR of F's epoch 57, when the loss was still falling. The run has a folder of its own, so that the trainer resumes
# from F's final checkpoint and not from the second cycle's newest last.ckpt.
# Usage: setsid -f nohup bash start_fr.sh >> /root/train_c2_fr.log 2>&1 < /dev/null
set -euo pipefail
SRC=/root/c2/run; ROOT=/root/c2/run_fr; V=/root/valfix; PY=/venv/main/bin/python3
F_STEPS_DONE=228840
NEW_STEPS=$(( 3814 + 38140 ))
[ -e "$ROOT" ] && { echo "$ROOT exists already, stop"; exit 1; }
[ -f "$V/F_final.ckpt" ] || { echo "no $V/F_final.ckpt, stop"; exit 1; }
mkdir -p "$ROOT/P" "$ROOT/F/lightning_logs/version_0/checkpoints" "$ROOT/nets"
touch "$ROOT/P.done"
ln -s "$SRC/P_final.pt" "$ROOT/P_final.pt"   # train_c2.sh only checks that it exists: F resumes from its checkpoint
cp "$V/F_final.ckpt" "$ROOT/F/lightning_logs/version_0/checkpoints/last.ckpt"
read -r EP GS < <(CUDA_VISIBLE_DEVICES="" "$PY" -c "
import torch
c = torch.load('$ROOT/F/lightning_logs/version_0/checkpoints/last.ckpt', map_location='cpu', weights_only=False)
print(c['epoch'], c['global_step'])")
[ "$GS" = "$F_STEPS_DONE" ] || { echo "F's final checkpoint is at step $GS, not $F_STEPS_DONE, stop"; exit 1; }
export ROOT P_EPOCHS=550 F_LR_FRAC=0.25 SAVE_PERIOD=1 EXPORT_SH=export_c2_keep.sh
export F_EPOCHS=$(( EP + NEW_STEPS / 3814 ))
export NNUE_LR_PIECES="7628:0.1494:1:lin;87722:1:0.6:cos;38140:0.6:0.6:lin;95350:0.6:0.00016:cos;3814:0.00016:0.02:lin;38140:0.02:0.00016:cos"
echo "=== F RESTART $(date -u) from F's end (epoch label $EP, step $GS), ROOT=$ROOT, F_EPOCHS=$F_EPOCHS NNUE_LR_PIECES=$NNUE_LR_PIECES"
cd /root/c2pkg; exec bash /root/c2pkg/train_c2_tail.sh
