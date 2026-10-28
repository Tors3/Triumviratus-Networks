#!/usr/bin/env bash
# Step 3 of after_F_auto.sh run on its own (27/10/2026). after_F_auto.sh started the second cycle as its own background
# child, so its `wait` for the t91 processes also waited for the whole second cycle; it was stopped by PID once the
# t91 dumps were complete. This script writes the paired comparisons and the FT-permuted copies of the final F and of
# P's epoch 496.
# Usage: setsid -f nohup bash finish_F_t91.sh >> /root/valfix/after_F.log 2>&1 < /dev/null
set -uo pipefail
ROOT=/root/c2/run; V=/root/valfix; OUT=/root/valfix/t91_F
log() { echo "[finish_F $(date -u +%H:%M:%S)] $*"; }

DUMPS=()
for t in "$OUT"/*.txt; do
  case "$t" in */all_dump.txt|*/paired_*) continue ;; esac
  n=$(wc -l < "$t")
  [ "$n" -eq 300 ] || { log "$t has $n lines instead of 300, stop"; exit 1; }
  DUMPS+=("$t")
done
log "t91: ${#DUMPS[@]} complete dumps"
cat "${DUMPS[@]}" > "$OUT/all_dump.txt"
python3 "$V/t91_paired.py" "$OUT/all_dump.txt" --ref nn-consilium.nnue > "$OUT/paired_vs_consilium.txt"
python3 "$V/t91_paired.py" "$OUT/all_dump.txt" --ref P_last_ep496.nnue > "$OUT/paired_vs_P496.txt"
cat "$OUT/paired_vs_P496.txt"
touch "$V/t91_F_done"
log "t91 done"

nice -n 10 bash "$V/perm_fast.sh" "$(cat "$V/F_final_exported")"
nice -n 10 bash "$V/perm_fast.sh" "$ROOT/nets/P_last_ep496.nnue"
tail -1 "$V/perm_latest.log"
touch "$V/perm_F_done"
log "permuted copies done"
