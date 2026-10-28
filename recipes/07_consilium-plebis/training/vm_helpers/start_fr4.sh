#!/usr/bin/env bash
# 5 more epochs after F's continuation (author, 28/10/2026 00:20 Italian): from the continuation's final checkpoint
# (step 240,282), 2 epochs at a flat 0.0003% of P's peak (about the mean LR of its last epoch), then 3 epochs of
# half-cosine down to 0.00001%, one checkpoint per epoch kept, in a run folder of its own.
# Started by chain_fr4.sh once the continuation has ended and its final checkpoint is exported.
set -euo pipefail
SRC=/root/c2/run_fr3; ROOT=/root/c2/run_fr4; V=/root/valfix; PY=/venv/main/bin/python3
DONE_STEPS=240282
NEW_STEPS=19070
[ -e "$ROOT" ] && { echo "$ROOT exists already, stop"; exit 1; }
CK=$(find "$SRC/F" -name last.ckpt -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
mkdir -p "$ROOT/P" "$ROOT/F/lightning_logs/version_0/checkpoints" "$ROOT/nets"
touch "$ROOT/P.done"
ln -s /root/c2/run/P_final.pt "$ROOT/P_final.pt"
cp "$CK" "$ROOT/F/lightning_logs/version_0/checkpoints/last.ckpt"
read -r EP GS < <(CUDA_VISIBLE_DEVICES="" "$PY" -c "
import torch
c = torch.load('$ROOT/F/lightning_logs/version_0/checkpoints/last.ckpt', map_location='cpu', weights_only=False)
print(c['epoch'], c['global_step'])")
[ "$GS" = "$DONE_STEPS" ] || { echo "the continuation's last checkpoint is at step $GS, not $DONE_STEPS, stop"; exit 1; }
export ROOT P_EPOCHS=550 F_LR_FRAC=0.25 SAVE_PERIOD=1 EXPORT_SH=export_c2_keep.sh
export F_EPOCHS=$(( EP + NEW_STEPS / 3814 ))
export NNUE_LR_PIECES="7628:0.1494:1:lin;87722:1:0.6:cos;38140:0.6:0.6:lin;95350:0.6:0.00016:cos;11442:0.00016:0:cos;7628:0.000012:0.000012:lin;11442:0.000012:0.0000004:cos"
echo "=== F 2 EP AT 0.0003% + 3 EP TO 0.00001% $(date -u) from the continuation's end (epoch label $EP, step $GS), ROOT=$ROOT, F_EPOCHS=$F_EPOCHS NNUE_LR_PIECES=$NNUE_LR_PIECES"
cd /root/c2pkg; exec bash /root/c2pkg/train_c2_tail.sh
