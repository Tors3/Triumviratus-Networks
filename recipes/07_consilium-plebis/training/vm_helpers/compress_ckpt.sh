#!/usr/bin/env bash
# Third pass for the three 5.9 GB checkpoints (28/10/2026): the first bytesplit.py wrote its planes in one call and a
# write of more than 2 GB to a pipe came out truncated (md5 mismatch, archives deleted); bytesplit.py now writes in
# 256 MB pieces (round trip of F_final.ckpt checked: same md5). Appends to MANIFEST.md5, then compress2_done.
# Usage: setsid -f nohup bash compress_ckpt.sh >> /root/valfix/compress_all.log 2>&1 < /dev/null
set -uo pipefail
V=/root/valfix; A=/root/arch; BS="$V/bytesplit.py"
log() { echo "[compress $(date -u +%H:%M:%S)] $*"; }
FL=/root/c2/run_fr3/F/lightning_logs/version_2/checkpoints
rm -f "$A/P_ep496_last.ckpt.bsplit.xz" "$A/F_final.ckpt.bsplit.xz" "$A/FR3_ep62_last.ckpt.bsplit.xz"
LIST="P_ep496_last.ckpt|$V/P_ep496_last.ckpt
F_final.ckpt|$V/F_final.ckpt
FR3_ep62_last.ckpt|$FL/epoch=60-step=240282.ckpt"
PIDS=()
while IFS='|' read -r name src; do
  ( out="$A/$name.bsplit.xz"
    python3 "$BS" split "$src" - | xz -9e -T18 -c > "$out"
    got=$(xz -dc -T8 "$out" | python3 "$BS" join - - | md5sum | cut -d' ' -f1)
    want=$(md5sum "$src" | cut -d' ' -f1)
    if [ "$got" = "$want" ]; then
      echo "$want $name $(md5sum "$out" | cut -d' ' -f1) $(basename "$out") $(stat -c %s "$src") $(stat -c %s "$out")" >> "$A/MANIFEST.md5"
      log "OK $name $(( $(stat -c %s "$out") * 100 / $(stat -c %s "$src") ))%"
    else
      log "MISMATCH $name (orig $want, restored $got)"
    fi ) &
  PIDS+=($!)
done <<< "$LIST"
wait "${PIDS[@]}"
cp "$BS" "$A/"
log "checkpoints done"
touch "$V/compress2_done"
