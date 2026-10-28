#!/usr/bin/env bash
# Second pass of compress_all.sh for the six float files (28/10/2026): its byte-plane pipelines stalled for minutes
# because the container sat at its memory limit with 357 GB of page cache of the training data (memory pressure 98%,
# every allocation reclaiming); after evicting that cache (posix_fadvise DONTNEED) a 1.18 GB split takes 4.5 s.
# Same archives and checks as compress_all.sh, then the manifest, RESTORE.txt and the compress_done flag.
# Usage: setsid -f nohup bash compress_big.sh >> /root/valfix/compress_all.log 2>&1 < /dev/null
set -uo pipefail
V=/root/valfix; A=/root/arch; BS="$V/bytesplit.py"
log() { echo "[compress $(date -u +%H:%M:%S)] $*"; }
FL=/root/c2/run_fr3/F/lightning_logs/version_2/checkpoints
LIST="P_ep496_last.ckpt|$V/P_ep496_last.ckpt|20
F_final.ckpt|$V/F_final.ckpt|20
FR3_ep62_last.ckpt|$FL/epoch=60-step=240282.ckpt|20
P_final.pt|/root/c2/run/P_final.pt|6
F_final.pt|$V/F_final.pt|6
FR3_avg3.pt|$V/FR3_avg3.pt|6"
PIDS=()
while IFS='|' read -r name src thr; do
  ( out="$A/$name.bsplit.xz"
    python3 "$BS" split "$src" - | xz -9e -T"$thr" -c > "$out"
    got=$(xz -dc -T8 "$out" | python3 "$BS" join - - | md5sum | cut -d' ' -f1)
    want=$(md5sum "$src" | cut -d' ' -f1)
    if [ "$got" = "$want" ]; then
      echo "$want $name $(md5sum "$out" | cut -d' ' -f1) $(basename "$out") $(stat -c %s "$src") $(stat -c %s "$out")" >> "$A/MANIFEST.tmp"
      log "OK $name $(( $(stat -c %s "$out") * 100 / $(stat -c %s "$src") ))%"
    else
      log "MISMATCH $name (orig $want, restored $got)"
    fi ) &
  PIDS+=($!)
done <<< "$LIST"
wait "${PIDS[@]}"
rm -f "$A/sprt_FR3avg3_vs_P496.tar"
{ echo "# original_md5 name archive_md5 archive original_bytes archive_bytes"; sort -k2 "$A/MANIFEST.tmp"; } > "$A/MANIFEST.md5"
cp "$BS" "$A/"
cat > "$A/RESTORE.txt" << 'EOF'
Restore (28/10/2026 archives of the Consilium Plebis run):
  *.bsplit.xz (checkpoints, .pt):  xz -dc NAME.bsplit.xz | python3 bytesplit.py join - NAME
  *.xz (networks, PGN tar):        xz -dc NAME.xz > NAME
bytesplit.py needs numpy and about three times the file size in RAM. MANIFEST.md5 lists the md5 of every original.
EOF
log "done: $(grep -vc '^#' "$A/MANIFEST.md5") archives"
touch "$V/compress_done"
