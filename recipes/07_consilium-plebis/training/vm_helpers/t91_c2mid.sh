#!/usr/bin/env bash
# Paired t91 of two exports of F's second cycle (author, 27/10/2026), to choose the LR of the tail: label 69 (step
# 274,608, LR 3.83% of P's peak, the closest to P's epoch 496 at 3.73%) and label 79 (step 312,748, LR 0.39%).
# The VM's t91 stream is deterministic (two processes read identical batches), so the dumps of t91_F are reused for
# the reference networks. Below the training's priority.
# Usage: setsid -f nohup bash t91_c2mid.sh >> /root/valfix/t91_c2mid.log 2>&1 < /dev/null
set -uo pipefail
ROOT=/root/c2/run; V=/root/valfix; T=/root/c2/nnue-pytorch; PY=/venv/main/bin/python3
OUT="$V/t91_c2mid"; mkdir -p "$OUT"
log() { echo "[t91_c2mid $(date -u +%H:%M:%S)] $*"; }
for f in nn-consilium P_last_ep496 P_last_ep486 P_epoch=479-step=1842162 F_epoch=49-step=194514 F_final_step228840; do
  cp "$V/t91_F/$f.txt" "$OUT/"
done
cd "$T"
for tag in 69-step=274608 79-step=312748; do
  n="$ROOT/nets/F_epoch=$tag.nnue"; b=$(basename "$n" .nnue)
  log "waiting for $n"
  until [ -f "$n" ]; do sleep 60; done
  s0=-1; s1=$(stat -c %s "$n")
  until [ "$s0" = "$s1" ] && [ "$s1" -gt 250000000 ]; do sleep 30; s0=$s1; s1=$(stat -c %s "$n"); done
  for try in 1 2 3; do
    CUDA_VISIBLE_DEVICES="" nice -n 10 "$PY" c2_val_loss_fixed2.py "$V/t91_sample.binpack" "$n" --batches 300 \
      --threads 8 --dump "$OUT/$b.txt" > "$OUT/$b.log" 2>&1
    [ "$(wc -l < "$OUT/$b.txt" 2>/dev/null)" = 300 ] && break   # complete: one dump line per batch
  done
  DUMPS=()
  for t in "$OUT"/*.txt; do case "$t" in */all_dump.txt|*/paired_*) ;; *) DUMPS+=("$t") ;; esac; done
  cat "${DUMPS[@]}" > "$OUT/all_dump.txt"
  python3 "$V/t91_paired.py" "$OUT/all_dump.txt" --ref P_last_ep496.nnue > "$OUT/paired_vs_P496.txt"
  cat "$OUT/paired_vs_P496.txt"
  touch "$V/t91_c2_${tag%%-*}_done"
  log "t91 of $b done"
done
