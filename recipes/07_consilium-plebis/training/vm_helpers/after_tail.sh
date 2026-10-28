#!/usr/bin/env bash
# Tail of F's second cycle (author, 27/10/2026), armed while the second cycle runs:
#   1. once after_cycle2.sh has copied and exported the second cycle's final checkpoint, the second cycle's driver has
#      exited and the GPUs are free: start_tail.sh in a session of its own (5 epochs at 0.004% of P's peak, one
#      checkpoint per epoch kept);
#   2. at the end of the tail: the average of its five epoch checkpoints (avg_ckpt.py), its export, and a copy of the
#      tail's last checkpoint;
#   3. paired t91 of the average, the tail's five epochs, the second cycle's end, F's end, P's epoch 496 and Consilium;
#      then the FT-permuted copy of the average (after the second cycle's own permuted copy).
# Usage: setsid -f nohup bash after_tail.sh >> /root/valfix/after_tail.log 2>&1 < /dev/null
set -uo pipefail
ROOT=/root/c2/run; V=/root/valfix; T=/root/c2/nnue-pytorch; PY=/venv/main/bin/python3
FEAT="Full_Threats+HalfKAv2_hm_X8^+PP_3Wide+PassedPawns"
C2_END=324190; TAIL_END=343260
log() { echo "[after_tail $(date -u +%H:%M:%S)] $*"; }

# 1. start of the tail
log "waiting for the export of the second cycle's final checkpoint ($V/F2_final_exported)"
until [ -f "$V/F2_final_exported" ]; do sleep 60; done
for i in $(seq 1 120); do pgrep -f "bash /root/c2pkg/train_c2.sh" > /dev/null || break; sleep 15; done
for i in $(seq 1 40); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
  [ "$used" -lt 1000 ] && break; sleep 15
done
if pgrep -f "bash /root/c2pkg/train_c2.sh" > /dev/null || [ "$used" -ge 1000 ]; then
  log "second cycle still alive or GPUs busy ($used MiB): tail NOT started"; exit 1
fi
setsid -f nohup bash "$V/start_tail.sh" >> /root/train_c2.log 2>&1 < /dev/null
for i in $(seq 1 40); do [ -f "$ROOT/F.done.cycle2" ] && break; sleep 15; done
[ -f "$ROOT/F.done.cycle2" ] || { log "start_tail.sh did not start the tail (see /root/train_c2.log), stop"; exit 1; }
log "tail started; waiting for its end ($ROOT/F.done)"

# 2. average of the five epochs
until [ -f "$ROOT/F.done" ]; do sleep 60; done
for i in $(seq 1 80); do pgrep -f "bash /root/c2pkg/train_c2_tail.sh" > /dev/null || break; sleep 15; done
mapfile -t CKS < <(find "$ROOT/F" -name 'epoch=*-step=*.ckpt' | awk -F'step=' -v lo="$C2_END" \
  '{s = $NF; sub(/\.ckpt$/, "", s); if (s + 0 > lo) print s " " $0}' | sort -n | cut -d' ' -f2-)
log "tail checkpoints: ${#CKS[@]}"; printf '  %s\n' "${CKS[@]}"
[ "${#CKS[@]}" -eq 5 ] || { log "expected 5 tail checkpoints, stop before averaging"; exit 1; }
case "${CKS[4]}" in *"step=$TAIL_END.ckpt") ;; *) log "the last tail checkpoint is not at step $TAIL_END, stop"; exit 1 ;; esac
cd "$T"
CUDA_VISIBLE_DEVICES="" "$PY" "$V/avg_ckpt.py" "$V/F_avg5.ckpt" "${CKS[@]}" || { log "averaging failed"; exit 1; }
FA="$ROOT/nets/F_avg5_step${TAIL_END}.nnue"
CUDA_VISIBLE_DEVICES="" "$PY" serialize.py "$V/F_avg5.ckpt" "$FA" --features "$FEAT" --pawn-king-hidden 256 --l1 1024 \
  --ft-compression leb128 > "$V/F_avg5.ser.log" 2>&1 || { log "serialization of the average failed"; exit 1; }
cp "${CKS[4]}" "$V/F_tail_last.ckpt"
md5sum "$FA" "$V/F_avg5.ckpt" "$V/F_tail_last.ckpt"
echo "$FA" > "$V/tail_avg_exported"
log "average exported: $FA"

# 3. t91 and the permuted copy
OUT="$V/t91_tail"; mkdir -p "$OUT"
NETS=("$V/nn-consilium.nnue" "$ROOT/nets/P_last_ep496.nnue" "$(cat "$V/F_final_exported")" "$(cat "$V/F2_final_exported")" "$FA")
for ck in "${CKS[@]}"; do NETS+=("$ROOT/nets/F_$(basename "$ck" .ckpt).nnue"); done
log "t91: ${#NETS[@]} networks, 300 batches, one process each (8 threads)"
PIDS=()
for n in "${NETS[@]}"; do
  [ -f "$n" ] || { log "missing $n"; continue; }
  b=$(basename "$n" .nnue)
  ( for try in 1 2 3; do
      CUDA_VISIBLE_DEVICES="" "$PY" c2_val_loss_fixed2.py "$V/t91_sample.binpack" "$n" --batches 300 --threads 8 \
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
cat "$OUT/paired_vs_P496.txt"
touch "$V/t91_tail_done"
log "t91 done"

for i in $(seq 1 240); do [ -f "$V/perm_F2_done" ] && break; sleep 30; done   # one permutation at a time
bash "$V/perm_fast.sh" "$FA"
tail -1 "$V/perm_latest.log"
touch "$V/perm_tail_done"
log "permuted copy done"
