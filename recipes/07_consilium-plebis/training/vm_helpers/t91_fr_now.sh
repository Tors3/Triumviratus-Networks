#!/usr/bin/env bash
# Paired t91 of the restart's epochs exported so far (author, 27/10/2026 23:10 Italian), against F's end, P's epoch 496
# and Consilium (their dumps reused from t91_F: the VM's t91 stream is deterministic). Exports the newest epoch
# checkpoint first if the run's exporter has not reached it yet. Below the training's priority.
# Usage: setsid -f nohup bash t91_fr_now.sh >> /root/valfix/t91_fr_now.log 2>&1 < /dev/null
set -uo pipefail
ROOT=/root/c2/run_fr; V=/root/valfix; T=/root/c2/nnue-pytorch; PY=/venv/main/bin/python3
FEAT="Full_Threats+HalfKAv2_hm_X8^+PP_3Wide+PassedPawns"
OUT="$V/t91_fr_now"; mkdir -p "$OUT"
log() { echo "[t91_fr_now $(date -u +%H:%M:%S)] $*"; }
cd "$T"
for ck in "$ROOT"/F/lightning_logs/version_2/checkpoints/epoch=*-step=*.ckpt; do
  n="$ROOT/nets/F_$(basename "$ck" .ckpt).nnue"
  [ -f "$n" ] && continue
  log "exporting $(basename "$ck")"
  CUDA_VISIBLE_DEVICES="" nice -n 10 "$PY" serialize.py "$ck" "$V/tmp_fr_now.nnue" --features "$FEAT" \
    --pawn-king-hidden 256 --l1 1024 --ft-compression leb128 > "$V/tmp_fr_now.ser.log" 2>&1 && mv "$V/tmp_fr_now.nnue" "$n"
done
for f in nn-consilium P_last_ep496 F_final_step228840; do cp "$V/t91_F/$f.txt" "$OUT/"; done
PIDS=()
for n in "$ROOT"/nets/F_epoch=*-step=*.nnue; do
  case "$n" in *_perm.nnue) continue ;; esac
  b=$(basename "$n" .nnue)
  [ -f "$OUT/$b.txt" ] && [ "$(wc -l < "$OUT/$b.txt")" = 300 ] && continue
  ( for try in 1 2 3; do
      CUDA_VISIBLE_DEVICES="" nice -n 10 "$PY" c2_val_loss_fixed2.py "$V/t91_sample.binpack" "$n" --batches 300 \
        --threads 16 --dump "$OUT/$b.txt" > "$OUT/$b.log" 2>&1
      [ "$(wc -l < "$OUT/$b.txt" 2>/dev/null)" = 300 ] && break   # complete: one dump line per batch
    done ) &
  PIDS+=($!)
done
log "t91 of ${#PIDS[@]} networks"
wait "${PIDS[@]}"
DUMPS=()
for t in "$OUT"/*.txt; do case "$t" in */all_dump.txt|*/paired_*) ;; *) DUMPS+=("$t") ;; esac; done
cat "${DUMPS[@]}" > "$OUT/all_dump.txt"
python3 "$V/t91_paired.py" "$OUT/all_dump.txt" --ref F_final_step228840.nnue > "$OUT/paired_vs_F_final.txt"
python3 "$V/t91_paired.py" "$OUT/all_dump.txt" --ref P_last_ep496.nnue > "$OUT/paired_vs_P496.txt"
cat "$OUT/paired_vs_F_final.txt"
touch "$V/t91_fr_now_done"
log "done"
