#!/usr/bin/env bash
# Exports the newest last.ckpt to .nnue (CPU) only, without the t91 loss (21/10/2026). Since 27/10 it looks in F and
# P and names the net after the phase of the checkpoint (F_last_epN / P_last_epN; F counts its epochs from 0).
set -uo pipefail
L=$(find /root/c2/run/F /root/c2/run/P -name last.ckpt -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d" " -f2-)
PH=$(echo "$L" | sed -n 's|^/root/c2/run/\([A-Z]*\)/.*|\1|p')
cp "$L" /root/valfix/latest.ckpt
cd /root/c2/nnue-pytorch
EP=$(CUDA_VISIBLE_DEVICES="" /venv/main/bin/python3 -c "import torch; print(torch.load(\"/root/valfix/latest.ckpt\", map_location=\"cpu\", weights_only=False)[\"epoch\"])")
OUT=/root/c2/run/nets/${PH}_last_ep${EP}.nnue
CUDA_VISIBLE_DEVICES="" nice -n 10 /venv/main/bin/python3 serialize.py /root/valfix/latest.ckpt "$OUT" --features "Full_Threats+HalfKAv2_hm_X8^+PP_3Wide+PassedPawns" --pawn-king-hidden 256 --l1 1024 --ft-compression leb128 > /root/valfix/export_latest.log 2>&1
rm -f /root/valfix/latest.ckpt
echo "EXPORTED $OUT" >> /root/valfix/export_latest.log
