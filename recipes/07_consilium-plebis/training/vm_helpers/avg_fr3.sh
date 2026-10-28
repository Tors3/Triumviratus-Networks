#!/usr/bin/env bash
# Weight averages of the last 2 and 3 epochs of F's continuation (author, 28/10/2026 00:35 Italian: "se la nuova fase
# non serve prendiamo media ultime 2 o 3 e basta"), their exports and their paired t91 against F's end, P's epoch 496,
# Consilium and the continuation's last epoch (dumps reused: deterministic stream). CPU only, after the t91 of the last
# epoch (after_fr3b.sh).
# Usage: setsid -f nohup bash avg_fr3.sh >> /root/valfix/avg_fr3.log 2>&1 < /dev/null
set -uo pipefail
ROOT=/root/c2/run_fr3; V=/root/valfix; T=/root/c2/nnue-pytorch; PY=/venv/main/bin/python3
FEAT="Full_Threats+HalfKAv2_hm_X8^+PP_3Wide+PassedPawns"
log() { echo "[avg_fr3 $(date -u +%H:%M:%S)] $*"; }
until [ -f "$V/t91_fr3_done" ]; do sleep 30; done
mapfile -t CKS < <(find "$ROOT/F" -name 'epoch=*-step=*.ckpt' | awk -F'step=' \
  '{s = $NF; sub(/\.ckpt$/, "", s); print s " " $0}' | sort -n | cut -d' ' -f2-)
log "epoch checkpoints: ${#CKS[@]}"; printf '  %s\n' "${CKS[@]}"
[ "${#CKS[@]}" -eq 3 ] || { log "expected 3 checkpoints, stop"; exit 1; }
cd "$T"
OUT="$V/t91_fr3"; NETS=()
for k in 2 3; do
  CUDA_VISIBLE_DEVICES="" "$PY" "$V/avg_ckpt.py" "$V/FR3_avg$k.ckpt" "${CKS[@]: -$k}" || { log "average of $k failed"; exit 1; }
  N="$ROOT/nets/FR3_avg${k}_step240282.nnue"
  CUDA_VISIBLE_DEVICES="" "$PY" serialize.py "$V/FR3_avg$k.ckpt" "$N" --features "$FEAT" --pawn-king-hidden 256 \
    --l1 1024 --ft-compression leb128 > "$V/FR3_avg$k.ser.log" 2>&1 || { log "serialization of $k failed"; exit 1; }
  NETS+=("$N")
done
md5sum "${NETS[@]}"
printf '%s\n' "${NETS[@]}" > "$V/fr3_avg_exported"
PIDS=()
for n in "${NETS[@]}"; do
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
python3 "$V/t91_paired.py" "$OUT/all_dump.txt" --ref F_final_step228840.nnue > "$OUT/paired_vs_F_final.txt"
python3 "$V/t91_paired.py" "$OUT/all_dump.txt" --ref P_last_ep496.nnue > "$OUT/paired_vs_P496.txt"
cat "$OUT/paired_vs_F_final.txt"
touch "$V/t91_fr3avg_done"
log "done"
