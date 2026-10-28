#!/usr/bin/env bash
# Consilium Plebis F (27/10/2026, the author's schedule): P stopped by hand after epoch 496 (last.ckpt, LR 3.73% of P's
# peak; copy in /root/valfix/P_ep496_last.ckpt). F = 70 epochs from P's weights, --lr = 25% of P's peak, piecewise LR:
#   2 epochs linear from P's last LR (0.1494 of F's --lr) up to 25%, 23 epochs half-cosine down to 15%,
#   20 epochs flat at 15%, 25 epochs half-cosine down to P's final LR (0.004% of P's peak).
# 3814 steps per epoch (1 G positions / batch 262144). Lambda 0.75 fixed, same BT4 data, routing table frozen.
export STOP_P=1 P_EPOCHS=550 F_EPOCHS=70 F_LR_FRAC=0.25
export NNUE_LR_PIECES="7628:0.1494:1:lin;87722:1:0.6:cos;76280:0.6:0.6:lin;95350:0.6:0.00016:cos"
echo "=== F START $(date -u) NNUE_LR_PIECES=$NNUE_LR_PIECES"
cd /root/c2pkg
exec bash /root/c2pkg/train_c2.sh
