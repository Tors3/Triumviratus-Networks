#!/usr/bin/env bash
# Starts start_fr4.sh once F's continuation has ended: after_fr3.sh has exported its final checkpoint (fr3_exported),
# the continuation's driver has exited and the GPUs are free. Then, at the end of the 4 flat epochs: export of each
# epoch and paired t91 against F's end, P's epoch 496 and Consilium (dumps reused: deterministic stream).
# Usage: setsid -f nohup bash chain_fr4.sh >> /root/valfix/chain_fr4.log 2>&1 < /dev/null
set -uo pipefail
V=/root/valfix; ROOT=/root/c2/run_fr4; T=/root/c2/nnue-pytorch; PY=/venv/main/bin/python3
FEAT="Full_Threats+HalfKAv2_hm_X8^+PP_3Wide+PassedPawns"
END=255538
log() { echo "[chain_fr4 $(date -u +%H:%M:%S)] $*"; }

log "waiting for the continuation's export ($V/fr3_exported)"
until [ -f "$V/fr3_exported" ]; do sleep 30; done
for i in $(seq 1 120); do pgrep -f "bash /root/c2pkg/train_c2_tail.sh" > /dev/null || break; sleep 15; done
for i in $(seq 1 40); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
  [ "$used" -lt 1000 ] && break; sleep 15
done
if pgrep -f "bash /root/c2pkg/train_c2_tail.sh" > /dev/null || [ "$used" -ge 1000 ]; then
  log "continuation still alive or GPUs busy ($used MiB): 4 flat epochs NOT started"; exit 1
fi
setsid -f nohup bash "$V/start_fr4.sh" >> /root/train_c2_fr4.log 2>&1 < /dev/null
log "4 flat epochs started"

until [ -f "$ROOT/F.done" ]; do sleep 60; done
for i in $(seq 1 80); do pgrep -f "bash /root/c2pkg/train_c2_tail.sh" > /dev/null || break; sleep 15; done
mapfile -t CKS < <(find "$ROOT/F" -name 'epoch=*-step=*.ckpt' | awk -F'step=' \
  '{s = $NF; sub(/\.ckpt$/, "", s); print s " " $0}' | sort -n | cut -d' ' -f2-)
log "epoch checkpoints: ${#CKS[@]}"; printf '  %s\n' "${CKS[@]}"
cd "$T"
NETS=()
for ck in "${CKS[@]}"; do
  n="$ROOT/nets/F_$(basename "$ck" .ckpt).nnue"
  [ -f "$n" ] || CUDA_VISIBLE_DEVICES="" "$PY" serialize.py "$ck" "$n" --features "$FEAT" --pawn-king-hidden 256 \
    --l1 1024 --ft-compression leb128 > "$V/$(basename "$n" .nnue).ser.log" 2>&1
  NETS+=("$n")
done
md5sum "${NETS[@]}"
printf '%s\n' "${NETS[-1]}" "${CKS[-1]}" > "$V/fr4_exported"
OUT="$V/t91_fr4"; mkdir -p "$OUT"
for f in nn-consilium P_last_ep496 F_final_step228840; do cp "$V/t91_F/$f.txt" "$OUT/"; done
[ -d "$V/t91_fr3" ] && for t in "$V"/t91_fr3/F_epoch=*.txt "$V"/t91_fr3/FR3_avg3*.txt; do [ -f "$t" ] && cp "$t" "$OUT/"; done
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
touch "$V/t91_fr4_done"
log "t91 done"
