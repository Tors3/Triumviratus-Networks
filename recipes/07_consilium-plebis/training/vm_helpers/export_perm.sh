#!/usr/bin/env bash
# Exports the newest last.ckpt (export_only.sh), then writes a copy with the feature transformer permuted for
# sparsity (ftperm through serialize.py --ft-optimize). CPU only (the GPUs belong to the run), nice 19 (23/10/2026).
set -uo pipefail
/root/valfix/export_only.sh
IN=$(grep -o '/root/c2/run/nets/P_last_ep[0-9]*.nnue' /root/valfix/export_latest.log | tail -1)
OUT=${IN%.nnue}_perm.nnue
cd /root/c2/nnue-pytorch
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
nice -n 19 /venv/main/bin/python3 serialize.py "$IN" "$OUT" --features 'Full_Threats+HalfKAv2_hm_X8^+PP_3Wide+PassedPawns'   --pawn-king-hidden 256 --l1 1024 --ft-compression leb128 --ft-optimize --ft-optimize-data /root/valfix/t91_sample.binpack   --ft-optimize-count 200000 --no-cupy --device cpu > /root/valfix/perm_latest.log 2>&1
rc=$?
echo "PERM_DONE $OUT rc=$rc" >> /root/valfix/perm_latest.log
