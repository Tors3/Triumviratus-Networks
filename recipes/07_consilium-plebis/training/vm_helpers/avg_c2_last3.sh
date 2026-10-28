#!/usr/bin/env bash
# Weight average of the last three epochs of F's second cycle (27/10/2026; step-epochs 82-84, mid-epoch LR 1.5e-6,
# 5.8e-7 and 8.5e-8, 0.27% to 0.015% of P's peak), the closest to the LR range of Consilium's F4 avg5 (8.9e-7 to
# 7.0e-8): the second cycle's cooling is 15 epochs long against F4's 60, so its last five epochs reach five times
# higher LRs. From the copies of snap_c2_last5.sh, after avg_c2_last5.sh (its t91 dumps are reused: the VM's t91
# stream is deterministic); export, paired t91 with the five-epoch average and the second cycle's end, FT-permuted
# copy. CPU only.
# Usage: setsid -f nohup bash avg_c2_last3.sh >> /root/valfix/avg_c2_last3.log 2>&1 < /dev/null
set -uo pipefail
ROOT=/root/c2/run; V=/root/valfix; T=/root/c2/nnue-pytorch; PY=/venv/main/bin/python3
FEAT="Full_Threats+HalfKAv2_hm_X8^+PP_3Wide+PassedPawns"
log() { echo "[avg_c2_last3 $(date -u +%H:%M:%S)] $*"; }

log "waiting for the t91 of the five-epoch average ($V/t91_c2avg5_done)"
until [ -f "$V/t91_c2avg5_done" ]; do sleep 60; done
CKS=()
for s in 316562 320376 324190; do
  c="$V/c2_last5/c2_step$s.ckpt"
  [ -f "$c" ] || { log "missing $c, stop"; exit 1; }
  CKS+=("$c")
done
cd "$T"
CUDA_VISIBLE_DEVICES="" nice -n 5 "$PY" "$V/avg_ckpt.py" "$V/F2_avg3.ckpt" "${CKS[@]}" || { log "averaging failed"; exit 1; }
FA="$ROOT/nets/F2_avg3_step324190.nnue"
CUDA_VISIBLE_DEVICES="" nice -n 5 "$PY" serialize.py "$V/F2_avg3.ckpt" "$FA" --features "$FEAT" --pawn-king-hidden 256 \
  --l1 1024 --ft-compression leb128 > "$V/F2_avg3.ser.log" 2>&1 || { log "serialization failed"; exit 1; }
md5sum "$FA" "$V/F2_avg3.ckpt"
echo "$FA" > "$V/c2_avg3_exported"
log "average exported: $FA"

OUT="$V/t91_c2avg3"; mkdir -p "$OUT"
for t in "$V"/t91_c2avg5/*.txt; do case "$t" in */all_dump.txt|*/paired_*) ;; *) cp "$t" "$OUT/" ;; esac; done
b=$(basename "$FA" .nnue)
for try in 1 2 3; do
  CUDA_VISIBLE_DEVICES="" "$PY" c2_val_loss_fixed2.py "$V/t91_sample.binpack" "$FA" --batches 300 --threads 16 \
    --dump "$OUT/$b.txt" > "$OUT/$b.log" 2>&1
  [ "$(wc -l < "$OUT/$b.txt" 2>/dev/null)" = 300 ] && break   # complete: one dump line per batch
done
DUMPS=()
for t in "$OUT"/*.txt; do case "$t" in */all_dump.txt|*/paired_*) ;; *) DUMPS+=("$t") ;; esac; done
cat "${DUMPS[@]}" > "$OUT/all_dump.txt"
python3 "$V/t91_paired.py" "$OUT/all_dump.txt" --ref P_last_ep496.nnue > "$OUT/paired_vs_P496.txt"
python3 "$V/t91_paired.py" "$OUT/all_dump.txt" --ref "$(basename "$(cat "$V/F2_final_exported")")" > "$OUT/paired_vs_F2_final.txt"
cat "$OUT/paired_vs_P496.txt" "$OUT/paired_vs_F2_final.txt"
touch "$V/t91_c2avg3_done"
log "t91 done"

for i in $(seq 1 240); do [ -f "$V/perm_c2avg5_done" ] && break; sleep 30; done   # one permutation at a time
bash "$V/perm_fast.sh" "$FA"
tail -1 "$V/perm_latest.log"
touch "$V/perm_c2avg3_done"
log "permuted copy done"
