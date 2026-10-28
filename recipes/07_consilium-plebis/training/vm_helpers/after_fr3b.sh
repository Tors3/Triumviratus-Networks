#!/usr/bin/env bash
# End of F's continuation, reduced (author, 28/10/2026 00:23 Italian: t91 of the last epoch only, no average): export
# of the final epoch, the fr3_exported flag that starts the 5 more epochs (chain_fr4.sh), paired t91 of the final epoch
# against F's end, P's epoch 496 and Consilium (their dumps reused: deterministic stream). Replaces after_fr3.sh.
# Usage: setsid -f nohup bash after_fr3b.sh >> /root/valfix/after_fr3.log 2>&1 < /dev/null
set -uo pipefail
ROOT=/root/c2/run_fr3; V=/root/valfix; T=/root/c2/nnue-pytorch; PY=/venv/main/bin/python3
FEAT="Full_Threats+HalfKAv2_hm_X8^+PP_3Wide+PassedPawns"
END=240282
log() { echo "[after_fr3b $(date -u +%H:%M:%S)] $*"; }

log "waiting for the end of the continuation ($ROOT/F.done)"
until [ -f "$ROOT/F.done" ]; do sleep 30; done
for i in $(seq 1 80); do pgrep -f "bash /root/c2pkg/train_c2_tail.sh" > /dev/null || break; sleep 15; done
CK=$(find "$ROOT/F" -name "epoch=*-step=$END.ckpt" | head -1)
[ -n "$CK" ] || { log "no checkpoint at step $END, stop"; exit 1; }
cd "$T"
FL="$ROOT/nets/F_$(basename "$CK" .ckpt).nnue"
[ -f "$FL" ] || CUDA_VISIBLE_DEVICES="" "$PY" serialize.py "$CK" "$FL" --features "$FEAT" --pawn-king-hidden 256 \
  --l1 1024 --ft-compression leb128 > "$V/$(basename "$FL" .nnue).ser.log" 2>&1
md5sum "$FL" "$CK"
printf '%s\n' "$FL" "$CK" > "$V/fr3_exported"
log "final epoch exported: $FL"

OUT="$V/t91_fr3"; mkdir -p "$OUT"
for f in nn-consilium P_last_ep496 F_final_step228840; do cp "$V/t91_F/$f.txt" "$OUT/"; done
b=$(basename "$FL" .nnue)
for try in 1 2 3; do
  CUDA_VISIBLE_DEVICES="" "$PY" c2_val_loss_fixed2.py "$V/t91_sample.binpack" "$FL" --batches 300 --threads 32 \
    --dump "$OUT/$b.txt" > "$OUT/$b.log" 2>&1
  [ "$(wc -l < "$OUT/$b.txt" 2>/dev/null)" = 300 ] && break   # complete: one dump line per batch
done
DUMPS=()
for t in "$OUT"/*.txt; do case "$t" in */all_dump.txt|*/paired_*) ;; *) DUMPS+=("$t") ;; esac; done
cat "${DUMPS[@]}" > "$OUT/all_dump.txt"
python3 "$V/t91_paired.py" "$OUT/all_dump.txt" --ref F_final_step228840.nnue > "$OUT/paired_vs_F_final.txt"
python3 "$V/t91_paired.py" "$OUT/all_dump.txt" --ref P_last_ep496.nnue > "$OUT/paired_vs_P496.txt"
cat "$OUT/paired_vs_F_final.txt"
touch "$V/t91_fr3_done"
log "t91 done"
