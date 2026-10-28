#!/usr/bin/env bash
# End of F's continuation without rise (start_fr3.sh, 27/10/2026): weight average of its 3 epochs (avg_ckpt.py) and
# its export; paired t91 of the average and of the three epochs against F's end, P's epoch 496 and Consilium (their
# dumps reused from t91_F: the VM's t91 stream is deterministic); FT-permuted copies of the final epoch and of the
# average.
# Usage: setsid -f nohup bash after_fr3.sh >> /root/valfix/after_fr3.log 2>&1 < /dev/null
set -uo pipefail
ROOT=/root/c2/run_fr3; V=/root/valfix; T=/root/c2/nnue-pytorch; PY=/venv/main/bin/python3
FEAT="Full_Threats+HalfKAv2_hm_X8^+PP_3Wide+PassedPawns"
END=240282
log() { echo "[after_fr3 $(date -u +%H:%M:%S)] $*"; }
ser() { CUDA_VISIBLE_DEVICES="" "$PY" serialize.py "$1" "$2" --features "$FEAT" --pawn-king-hidden 256 --l1 1024 \
          --ft-compression leb128 > "$V/$(basename "$2" .nnue).ser.log" 2>&1; }

log "waiting for the end of the continuation ($ROOT/F.done)"
until [ -f "$ROOT/F.done" ]; do sleep 60; done
for i in $(seq 1 80); do pgrep -f "bash /root/c2pkg/train_c2_tail.sh" > /dev/null || break; sleep 15; done
mapfile -t CKS < <(find "$ROOT/F" -name 'epoch=*-step=*.ckpt' | awk -F'step=' \
  '{s = $NF; sub(/\.ckpt$/, "", s); print s " " $0}' | sort -n | cut -d' ' -f2-)
log "epoch checkpoints: ${#CKS[@]}"; printf '  %s\n' "${CKS[@]}"
[ "${#CKS[@]}" -eq 3 ] || { log "expected 3 epoch checkpoints, stop"; exit 1; }
case "${CKS[-1]}" in *"step=$END.ckpt") ;; *) log "the last checkpoint is not at step $END, stop"; exit 1 ;; esac
cd "$T"
CUDA_VISIBLE_DEVICES="" "$PY" "$V/avg_ckpt.py" "$V/FR3_avg3.ckpt" "${CKS[@]}" || { log "average failed"; exit 1; }
FA="$ROOT/nets/FR3_avg3_step$END.nnue"
ser "$V/FR3_avg3.ckpt" "$FA" || { log "serialization of the average failed"; exit 1; }
FL="$ROOT/nets/F_$(basename "${CKS[-1]}" .ckpt).nnue"
[ -f "$FL" ] || ser "${CKS[-1]}" "$FL"
md5sum "$FL" "$FA" "${CKS[-1]}" "$V/FR3_avg3.ckpt"
printf '%s\n' "$FL" "$FA" "${CKS[-1]}" > "$V/fr3_exported"
log "exports done"

OUT="$V/t91_fr3"; mkdir -p "$OUT"
for f in nn-consilium P_last_ep496 F_final_step228840; do cp "$V/t91_F/$f.txt" "$OUT/"; done
NETS=("$FA")
for ck in "${CKS[@]}"; do NETS+=("$ROOT/nets/F_$(basename "$ck" .ckpt).nnue"); done
log "t91: ${#NETS[@]} networks, 300 batches, 16 threads each"
PIDS=()
for n in "${NETS[@]}"; do
  [ -f "$n" ] || { log "missing $n"; continue; }
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
touch "$V/t91_fr3_done"
log "t91 done"

for n in "$FL" "$FA"; do
  bash "$V/perm_fast.sh" "$n"; tail -1 "$V/perm_latest.log"
done
touch "$V/perm_fr3_done"
log "permuted copies done"
