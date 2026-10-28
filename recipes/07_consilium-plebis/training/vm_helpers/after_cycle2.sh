#!/usr/bin/env bash
# After the end of F's second cycle (27/10/2026): copy and export of its final checkpoint (not a periodic one), the
# paired t91 loss of the final networks of both cycles, P's epoch 496, Consilium and the periodic exports of the
# second cycle, then the FT-permuted copy of the final network. start_cycle2.sh renames F.done to F.done.cycle1, and
# train_c2.sh writes F.done again at the end of the second cycle.
# Armed by after_F_auto.sh: setsid -f nohup bash after_cycle2.sh >> /root/valfix/after_cycle2.log 2>&1 < /dev/null
set -uo pipefail
ROOT=/root/c2/run; V=/root/valfix; T=/root/c2/nnue-pytorch; PY=/venv/main/bin/python3
FEAT="Full_Threats+HalfKAv2_hm_X8^+PP_3Wide+PassedPawns"
END_STEP=324190
log() { echo "[after_cycle2 $(date -u +%H:%M:%S)] $*"; }

log "waiting for the start of the second cycle ($ROOT/F.done.cycle1)"
for i in $(seq 1 120); do [ -f "$ROOT/F.done.cycle1" ] && break; sleep 15; done
[ -f "$ROOT/F.done.cycle1" ] || { log "the second cycle did not start, stop"; exit 1; }
log "second cycle running; waiting for its end ($ROOT/F.done)"
until [ -f "$ROOT/F.done" ]; do sleep 60; done
CK=$(find "$ROOT/F" -name last.ckpt -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
cp "$CK" "$V/F2_final.ckpt"
cd "$T"
read -r EP GS < <(CUDA_VISIBLE_DEVICES="" "$PY" -c "
import torch
c = torch.load('$V/F2_final.ckpt', map_location='cpu', weights_only=False)
print(c['epoch'], c['global_step'])")
log "second cycle done: final checkpoint epoch label $EP, global step $GS (expected $END_STEP)"
FN2="$ROOT/nets/F2_final_step${GS}.nnue"
CUDA_VISIBLE_DEVICES="" "$PY" serialize.py "$V/F2_final.ckpt" "$FN2" --features "$FEAT" --pawn-king-hidden 256 \
  --l1 1024 --ft-compression leb128 > "$V/F2_final.ser.log" 2>&1
md5sum "$FN2"
echo "$FN2" > "$V/F2_final_exported"
log "export done"

# the run's own last export pass (periodic checkpoints of the second cycle) ends within minutes
for i in $(seq 1 60); do pgrep -f "bash /root/c2pkg/train_c2.sh" > /dev/null || break; sleep 15; done
OUT="$V/t91_F2"; mkdir -p "$OUT"
NETS=("$V/nn-consilium.nnue" "$ROOT/nets/P_last_ep496.nnue" "$(cat "$V/F_final_exported")" "$FN2")
while read -r n; do NETS+=("$n"); done < <(ls "$ROOT"/nets/F_epoch=*-step=*.nnue | awk -F'step=' '{s=$2; sub(/\.nnue$/,"",s); if (s+0 > 228840) print}')
log "t91: ${#NETS[@]} networks, 300 batches, one process each"
for n in "${NETS[@]}"; do
  [ -f "$n" ] || { log "missing $n"; continue; }
  b=$(basename "$n" .nnue)
  ( for try in 1 2 3; do
      CUDA_VISIBLE_DEVICES="" "$PY" c2_val_loss_fixed2.py "$V/t91_sample.binpack" "$n" --batches 300 --threads 8 \
        --dump "$OUT/$b.txt" > "$OUT/$b.log" 2>&1
      grep -q "batches x" "$OUT/$b.log" && break
    done ) &
done
wait
cat "$OUT"/*.txt > "$OUT/all_dump.txt"
python3 "$V/t91_paired.py" "$OUT/all_dump.txt" --ref nn-consilium.nnue > "$OUT/paired_vs_consilium.txt"
python3 "$V/t91_paired.py" "$OUT/all_dump.txt" --ref "$(basename "$(cat "$V/F_final_exported")")" > "$OUT/paired_vs_F_final.txt"
cat "$OUT/paired_vs_F_final.txt"
touch "$V/t91_F2_done"
log "t91 done"

bash "$V/perm_fast.sh" "$FN2"
tail -1 "$V/perm_latest.log"
touch "$V/perm_F2_done"
log "permuted copy done"
