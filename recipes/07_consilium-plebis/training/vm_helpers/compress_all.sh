#!/usr/bin/env bash
# Max-ratio archives of the run's files before the VM is released (author, 28/10/2026 01:20 Italian: "tutte le
# compressioni al massimo sulla vm", networks too). Float checkpoints and .pt: byte planes (bytesplit.py), then
# xz -9e (63-73% of the original on the 28/10 test, against 68-79% for plain xz and 71-79% for zstd -22); networks,
# PGN and logs: xz -9e (the planes do not help them). Every archive is checked by decompressing it and comparing the
# md5 with the original's. Output /root/arch with MANIFEST.md5 (original md5 + archive md5) and RESTORE.txt.
# Usage: setsid -f nohup bash compress_all.sh >> /root/valfix/compress_all.log 2>&1 < /dev/null
set -uo pipefail
V=/root/valfix; A=/root/arch; BS="$V/bytesplit.py"; mkdir -p "$A"
log() { echo "[compress $(date -u +%H:%M:%S)] $*"; }
FL=/root/c2/run_fr3/F/lightning_logs/version_2/checkpoints
# name in the archive | source | kind (s = byte planes + xz, x = xz) | xz threads
LIST="P_ep496_last.ckpt|$V/P_ep496_last.ckpt|s|20
F_final.ckpt|$V/F_final.ckpt|s|20
FR3_ep62_last.ckpt|$FL/epoch=60-step=240282.ckpt|s|20
P_final.pt|/root/c2/run/P_final.pt|s|6
F_final.pt|$V/F_final.pt|s|6
FR3_avg3.pt|$V/FR3_avg3.pt|s|6
P_last_ep486_perm.nnue|/root/c2/run/nets/P_last_ep486_perm.nnue|x|2
P_last_ep496.nnue|/root/c2/run/nets/P_last_ep496.nnue|x|2
P_last_ep496_perm.nnue|/root/c2/run/nets/P_last_ep496_perm.nnue|x|2
F_final_step228840.nnue|/root/c2/run/nets/F_final_step228840.nnue|x|2
F_final_step228840_perm.nnue|/root/c2/run/nets/F_final_step228840_perm.nnue|x|2
FR3_final_step240282.nnue|/root/c2/run_fr3/nets/F_epoch=60-step=240282.nnue|x|2
FR3_avg2_step240282.nnue|/root/c2/run_fr3/nets/FR3_avg2_step240282.nnue|x|2
FR3_avg3_step240282.nnue|/root/c2/run_fr3/nets/FR3_avg3_step240282.nnue|x|2
FR3_avg3_step240282_perm1m.nnue|/root/c2/run_fr3/nets/FR3_avg3_step240282_perm.nnue|x|2"
tar -C /root/linux_kit -cf "$A/sprt_FR3avg3_vs_P496.tar" results
LIST+=$'\n'"sprt_FR3avg3_vs_P496.tar|$A/sprt_FR3avg3_vs_P496.tar|x|2"
PIDS=()
while IFS='|' read -r name src kind thr; do
  [ -f "$src" ] || { log "MISSING $src"; continue; }
  ( if [ "$kind" = s ]; then
      out="$A/$name.bsplit.xz"; python3 "$BS" split "$src" - | xz -9e -T"$thr" -c > "$out"
      got=$(xz -dc -T8 "$out" | python3 "$BS" join - - | md5sum | cut -d' ' -f1)
    else
      out="$A/$name.xz"; xz -9e -T"$thr" -c "$src" > "$out"
      got=$(xz -dc -T8 "$out" | md5sum | cut -d' ' -f1)
    fi
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
rm -f "$A/MANIFEST.tmp"
cp "$BS" "$A/"
cat > "$A/RESTORE.txt" << 'EOF'
Restore (28/10/2026 archives of the Consilium Plebis run):
  *.bsplit.xz (checkpoints, .pt):  xz -dc NAME.bsplit.xz | python3 bytesplit.py join - NAME
  *.xz (networks, PGN tar):        xz -dc NAME.xz > NAME
bytesplit.py needs numpy and about twice the file size in RAM. MANIFEST.md5 lists the md5 of every original.
EOF
log "done: $(grep -vc '^#' "$A/MANIFEST.md5") archives"
touch "$V/compress_done"
