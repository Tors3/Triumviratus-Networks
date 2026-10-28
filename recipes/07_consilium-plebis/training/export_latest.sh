#!/usr/bin/env bash
# Exports the newest last.ckpt to .nnue (CPU), then its fixed-lambda loss on t91 (Consilium-2 only; Consilium is in ep9.txt).
set -uo pipefail
L=$(find /root/c2/run/P -name last.ckpt -printf "%T@ %p\n" | sort -n | tail -1 | cut -d" " -f2-)
cp "$L" /root/valfix/latest.ckpt
cd /root/c2/nnue-pytorch
EP=$(CUDA_VISIBLE_DEVICES="" /venv/main/bin/python3 -c "import torch; print(torch.load(\"/root/valfix/latest.ckpt\", map_location=\"cpu\", weights_only=False)[\"epoch\"])")
OUT=/root/c2/run/nets/P_last_ep${EP}.nnue
CUDA_VISIBLE_DEVICES="" nice -n 10 /venv/main/bin/python3 serialize.py /root/valfix/latest.ckpt "$OUT" --features "Full_Threats+HalfKAv2_hm_X8^+PP_3Wide+PassedPawns" --pawn-king-hidden 256 --l1 1024 --ft-compression leb128 > /root/valfix/export_latest.log 2>&1
rm -f /root/valfix/latest.ckpt
echo "EXPORTED $OUT" >> /root/valfix/export_latest.log
CUDA_VISIBLE_DEVICES="" nice -n 19 /venv/main/bin/python3 c2_val_loss_fixed.py /root/valfix/t91_sample.binpack "$OUT" --batches 200 --batch-size 8192 --threads 4 > /root/valfix/loss_ep${EP}.txt 2>&1
echo "LOSS DONE ep${EP}" >> /root/valfix/export_latest.log
