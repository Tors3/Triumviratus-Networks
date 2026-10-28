#!/usr/bin/env bash
# Permuted copy of the final network (average of the continuation's last 3 epochs) on 2 million positions (author,
# 28/10/2026: "2 milioni di posizioni o più con tutta la cpu post compressione"), after compress_all.sh. Positions
# from the BT4 training data, which has far more distinct positions than the 70 MB t91 sample used by the earlier
# copies (the loader keeps about 1 position in 11). Activation gathering on GPU 0 (cupy, ~18 GB at 2M; the 1M run
# took 9 GB), the evaluation of the permutation on the CPU with all the quota's threads. Then the archive of it.
# Usage: setsid -f nohup bash perm2m.sh >> /root/valfix/perm2m.log 2>&1 < /dev/null
set -uo pipefail
V=/root/valfix; A=/root/arch
IN=/root/c2/run_fr3/nets/FR3_avg3_step240282.nnue
OUT=/root/c2/run_fr3/nets/FR3_avg3_step240282_perm2m.nnue
DATA=/data/bt4/T60T70wIsRightFarseerT60T74T75T76.split_0.relabel-BT4-tf13tune.binpack
log() { echo "[perm2m $(date -u +%H:%M:%S)] $*"; }
until [ -f "$V/compress_done" ]; do sleep 20; done
log "compression done, permuting $IN on 2,000,000 positions"
cd /root/c2/nnue-pytorch
export OMP_NUM_THREADS=56 MKL_NUM_THREADS=56
/venv/main/bin/python3 serialize.py "$IN" "$OUT" --features 'Full_Threats+HalfKAv2_hm_X8^+PP_3Wide+PassedPawns' \
  --pawn-king-hidden 256 --l1 1024 --ft-compression leb128 --ft-optimize --ft-optimize-data "$DATA" \
  --ft-optimize-count 2000000 --cupy --device 0 > "$V/perm2m_serialize.log" 2>&1
rc=$?
log "serialize rc=$rc; $(grep -E 'Combined zeros|Total improvement' "$V/perm2m_serialize.log" | tail -2 | tr '\n' ' ')"
[ "$rc" = 0 ] && [ -f "$OUT" ] || { log "no permuted copy"; exit 1; }
md5sum "$OUT"
xz -9e -T8 -c "$OUT" > "$A/FR3_avg3_step240282_perm2m.nnue.xz"
[ "$(xz -dc "$A/FR3_avg3_step240282_perm2m.nnue.xz" | md5sum | cut -d' ' -f1)" = "$(md5sum "$OUT" | cut -d' ' -f1)" ] \
  && log "archive OK" || log "archive MISMATCH"
touch "$V/perm2m_done"
