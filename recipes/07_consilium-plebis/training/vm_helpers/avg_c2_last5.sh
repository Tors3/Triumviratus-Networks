#!/usr/bin/env bash
# Weight average of the last five epochs of F's second cycle (27/10/2026; step-epochs 80-84, LR 1.0% -> 0.004% of
# P's peak), as Consilium's final network (F4 avg5), from the copies of snap_c2_last5.sh; its export; the paired t91
# against the second cycle's end, F's end, P's epoch 496 and Consilium; the FT-permuted copy. CPU only.
# Usage: setsid -f nohup bash avg_c2_last5.sh >> /root/valfix/avg_c2_last5.log 2>&1 < /dev/null
set -uo pipefail
ROOT=/root/c2/run; V=/root/valfix; T=/root/c2/nnue-pytorch; PY=/venv/main/bin/python3
FEAT="Full_Threats+HalfKAv2_hm_X8^+PP_3Wide+PassedPawns"
log() { echo "[avg_c2_last5 $(date -u +%H:%M:%S)] $*"; }

log "waiting for the five copies ($V/c2_last5_done)"
until [ -f "$V/c2_last5_done" ]; do sleep 60; done
CKS=()
for s in 308934 312748 316562 320376 324190; do
  c="$V/c2_last5/c2_step$s.ckpt"
  [ -f "$c" ] || { log "missing $c, stop"; exit 1; }
  CKS+=("$c")
done
cd "$T"
CUDA_VISIBLE_DEVICES="" nice -n 5 "$PY" "$V/avg_ckpt.py" "$V/F2_avg5.ckpt" "${CKS[@]}" || { log "averaging failed"; exit 1; }
FA="$ROOT/nets/F2_avg5_step324190.nnue"
CUDA_VISIBLE_DEVICES="" nice -n 5 "$PY" serialize.py "$V/F2_avg5.ckpt" "$FA" --features "$FEAT" --pawn-king-hidden 256 \
  --l1 1024 --ft-compression leb128 > "$V/F2_avg5.ser.log" 2>&1 || { log "serialization failed"; exit 1; }
md5sum "$FA" "$V/F2_avg5.ckpt"
echo "$FA" > "$V/c2_avg5_exported"
log "average exported: $FA"

log "waiting for the export of the second cycle's end ($V/F2_final_exported)"
until [ -f "$V/F2_final_exported" ]; do sleep 60; done
OUT="$V/t91_c2avg5"; mkdir -p "$OUT"
for f in nn-consilium P_last_ep496 F_final_step228840; do cp "$V/t91_F/$f.txt" "$OUT/"; done   # deterministic stream
PIDS=()
for n in "$FA" "$(cat "$V/F2_final_exported")"; do
  b=$(basename "$n" .nnue)
  ( for try in 1 2 3; do
      CUDA_VISIBLE_DEVICES="" "$PY" c2_val_loss_fixed2.py "$V/t91_sample.binpack" "$n" --batches 300 --threads 16 \
        --dump "$OUT/$b.txt" > "$OUT/$b.log" 2>&1
      [ "$(wc -l < "$OUT/$b.txt" 2>/dev/null)" = 300 ] && break   # complete: one dump line per batch
    done ) &
  PIDS+=($!)
done
wait "${PIDS[@]}"
DUMPS=()
for t in "$OUT"/*.txt; do case "$t" in */all_dump.txt|*/paired_*) ;; *) DUMPS+=("$t") ;; esac; done
cat "${DUMPS[@]}" > "$OUT/all_dump.txt"
python3 "$V/t91_paired.py" "$OUT/all_dump.txt" --ref P_last_ep496.nnue > "$OUT/paired_vs_P496.txt"
python3 "$V/t91_paired.py" "$OUT/all_dump.txt" --ref "$(basename "$(cat "$V/F2_final_exported")")" > "$OUT/paired_vs_F2_final.txt"
cat "$OUT/paired_vs_P496.txt" "$OUT/paired_vs_F2_final.txt"
touch "$V/t91_c2avg5_done"
log "t91 done"

for i in $(seq 1 240); do [ -f "$V/perm_F2_done" ] && break; sleep 30; done   # one permutation at a time
bash "$V/perm_fast.sh" "$FA"
tail -1 "$V/perm_latest.log"
touch "$V/perm_c2avg5_done"
log "permuted copy done"
