#!/usr/bin/env bash
# Consilium Plebis F, resumed with the flat 15% stretch cut from 20 to 10 epochs (author, 27/10/2026).
# Schedule by optimizer step (3814 steps = 1 epoch): rise 2 epochs, cosine 25% -> 15% in 23, flat 15% for 10 (was 20),
# cosine down to P's final lr in 25: 60 epochs of steps. The scheduler restores its step count from last.ckpt and
# takes the new pieces from this variable (the lambda is not stored in the checkpoint).
# F_EPOCHS is passed by cut_flat.sh: a resume repeats the epoch label stored in last.ckpt while the step count goes on,
# so max epochs = checkpoint epoch + remaining steps / 3814.
export P_EPOCHS=550 F_LR_FRAC=0.25
export F_EPOCHS="${F_EPOCHS:?set by cut_flat.sh}"
export NNUE_LR_PIECES="7628:0.1494:1:lin;87722:1:0.6:cos;38140:0.6:0.6:lin;95350:0.6:0.00016:cos"
echo "=== F RESUME $(date -u) F_EPOCHS=$F_EPOCHS NNUE_LR_PIECES=$NNUE_LR_PIECES"
cd /root/c2pkg
exec bash /root/c2pkg/train_c2.sh
