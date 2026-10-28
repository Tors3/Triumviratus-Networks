#!/usr/bin/env bash
# Permuted copy of an export on GPU 1 (ftperm via serialize.py --ft-optimize, cupy), with a watchdog that kills it
# if GPU 1 goes above 22000 MiB, so the training never runs out of memory (23/10/2026). Usage: perm_gpu.sh <in.nnue>
set -uo pipefail
IN=${1:?}
OUT=${IN%.nnue}_perm.nnue
cd /root/c2/nnue-pytorch
/venv/main/bin/python3 serialize.py "$IN" "$OUT" --features 'Full_Threats+HalfKAv2_hm_X8^+PP_3Wide+PassedPawns' --pawn-king-hidden 256 --l1 1024 --ft-compression leb128 --ft-optimize --ft-optimize-data /root/valfix/t91_sample.binpack --ft-optimize-count 1000000 --cupy --device 1 > /root/valfix/perm_latest.log 2>&1 &
P=$!
peak=0
while kill -0 $P 2>/dev/null; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
  [ "$m" -gt "$peak" ] && peak=$m
  if [ "$m" -gt 22000 ]; then kill $P; echo "WATCHDOG killed at $m MiB" >> /root/valfix/perm_latest.log; fi
  sleep 2
done
wait $P; rc=$?
echo "PERM_DONE $OUT rc=$rc peak_gpu1=${peak}MiB" >> /root/valfix/perm_latest.log
