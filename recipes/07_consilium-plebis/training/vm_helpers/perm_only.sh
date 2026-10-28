#!/usr/bin/env bash
# Permuted copy of an export (ftperm via serialize.py --ft-optimize), CPU only, nice 19 (23/10/2026). Usage: perm_only.sh <in.nnue>
set -uo pipefail
IN=${1:?}
OUT=${IN%.nnue}_perm.nnue
cd /root/c2/nnue-pytorch
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
nice -n 5 /venv/main/bin/python3 serialize.py "$IN" "$OUT" --features 'Full_Threats+HalfKAv2_hm_X8^+PP_3Wide+PassedPawns' --pawn-king-hidden 256 --l1 1024 --ft-compression leb128 --ft-optimize --ft-optimize-data /root/valfix/t91_sample.binpack --ft-optimize-count 500000 --no-cupy --device cpu > /root/valfix/perm_latest.log 2>&1
rc=$?
echo "PERM_DONE $OUT rc=$rc" >> /root/valfix/perm_latest.log
