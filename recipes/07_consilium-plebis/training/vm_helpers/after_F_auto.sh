#!/usr/bin/env bash
# After the end of F (27/10/2026), with the second cycle started automatically (author: travelling at the end of F).
#   1. copy and export of F's final checkpoint (the run's export skips last.ckpt, and after the resume of 08:22 the
#      final epoch label is not a periodic checkpoint) and of P's epoch 496;
#   2. once the F run has exited and the GPUs are free: the second cycle (start_cycle2.sh);
#   3. at lower priority than the training: the paired t91 loss of the final F, P's epoch 496, Consilium and earlier
#      exports, then the FT-permuted copies of the final F and of P's epoch 496.
# The copy that ran on 27/10 started the second cycle as its own background child, so the `wait` of step 3 also
# waited for the whole second cycle; it was stopped by PID once the t91 dumps were complete and finish_F_t91.sh did
# the rest. This copy starts the second cycle in a session of its own and waits only for the t91 processes.
# Usage: setsid -f nohup bash after_F_auto.sh >> /root/valfix/after_F.log 2>&1 < /dev/null
set -uo pipefail
ROOT=/root/c2/run; V=/root/valfix; T=/root/c2/nnue-pytorch; PY=/venv/main/bin/python3
FEAT="Full_Threats+HalfKAv2_hm_X8^+PP_3Wide+PassedPawns"
log() { echo "[after_F $(date -u +%H:%M:%S)] $*"; }
ser() { CUDA_VISIBLE_DEVICES="" nice -n 5 "$PY" serialize.py "$1" "$2" --features "$FEAT" --pawn-king-hidden 256 \
          --l1 1024 --ft-compression leb128 > "$V/$(basename "$2" .nnue).ser.log" 2>&1; }

log "waiting for $ROOT/F.done"
until [ -f "$ROOT/F.done" ]; do sleep 30; done
CK=$(find "$ROOT/F" -name last.ckpt -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
cp "$CK" "$V/F_final.ckpt"
cd "$T"
read -r EP GS < <(CUDA_VISIBLE_DEVICES="" "$PY" -c "
import torch
c = torch.load('$V/F_final.ckpt', map_location='cpu', weights_only=False)
print(c['epoch'], c['global_step'])")
log "F done: final checkpoint epoch label $EP, global step $GS (copy $V/F_final.ckpt)"
FN="$ROOT/nets/F_final_step${GS}.nnue"
P496="$ROOT/nets/P_last_ep496.nnue"
ser "$V/F_final.ckpt" "$FN" &
[ -f "$P496" ] || ser "$V/P_ep496_last.ckpt" "$P496" &
wait
md5sum "$FN" "$P496"
echo "$FN" > "$V/F_final_exported"
log "exports done"

# 2. the second cycle: the F run must have exited (its last export pass, the end line, the exporter loop) and the
# GPUs must be free
for i in $(seq 1 120); do
  pgrep -f "bash /root/c2pkg/train_c2.sh" > /dev/null || break
  sleep 15
done
for i in $(seq 1 40); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
  [ "$used" -lt 1000 ] && break; sleep 15
done
if pgrep -f "bash /root/c2pkg/train_c2.sh" > /dev/null || [ "$used" -ge 1000 ]; then
  log "F run still alive or GPUs busy ($used MiB): second cycle NOT started"
else
  setsid -f nohup bash "$V/start_cycle2.sh" >> /root/train_c2.log 2>&1 < /dev/null
  log "second cycle started"
  setsid -f nohup bash "$V/after_cycle2.sh" >> "$V/after_cycle2.log" 2>&1 < /dev/null
  log "after_cycle2.sh armed"
fi

# 3. t91 and permutations below the training's priority
F49="$ROOT/nets/F_epoch=49-step=194514.nnue"
for i in $(seq 1 60); do [ -f "$F49" ] && break; sleep 30; done
OUT="$V/t91_F"; mkdir -p "$OUT"
NETS=("$V/nn-consilium.nnue" "$P496" "$FN" "$ROOT/nets/P_epoch=399-step=1533228.nnue"
      "$ROOT/nets/P_epoch=439-step=1685788.nnue" "$ROOT/nets/P_epoch=479-step=1842162.nnue"
      "$ROOT/nets/P_last_ep486.nnue" "$ROOT/nets/F_epoch=19-step=76280.nnue"
      "$ROOT/nets/F_epoch=39-step=156374.nnue" "$F49")
log "t91: ${#NETS[@]} networks, 300 batches, one process each (5 threads, nice 10)"
PIDS=()
for n in "${NETS[@]}"; do
  [ -f "$n" ] || { log "missing $n"; continue; }
  b=$(basename "$n" .nnue)
  ( for try in 1 2 3; do
      CUDA_VISIBLE_DEVICES="" nice -n 10 "$PY" c2_val_loss_fixed2.py "$V/t91_sample.binpack" "$n" --batches 300 \
        --threads 5 --dump "$OUT/$b.txt" > "$OUT/$b.log" 2>&1
      [ "$(wc -l < "$OUT/$b.txt" 2>/dev/null)" = 300 ] && break   # complete: one dump line per batch
    done ) &
  PIDS+=($!)
done
wait "${PIDS[@]}"
cat "$OUT"/*.txt > "$OUT/all_dump.txt"
python3 "$V/t91_paired.py" "$OUT/all_dump.txt" --ref nn-consilium.nnue > "$OUT/paired_vs_consilium.txt"
python3 "$V/t91_paired.py" "$OUT/all_dump.txt" --ref P_last_ep496.nnue > "$OUT/paired_vs_P496.txt"
cat "$OUT/paired_vs_P496.txt"
touch "$V/t91_F_done"
log "t91 done"

nice -n 10 bash "$V/perm_fast.sh" "$FN"
nice -n 10 bash "$V/perm_fast.sh" "$P496"
tail -1 "$V/perm_latest.log"
touch "$V/perm_F_done"
log "permuted copies done"
