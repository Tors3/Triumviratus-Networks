#!/usr/bin/env bash
# Copies of last.ckpt at the end of the last five epochs of F's second cycle (27/10/2026): step-epochs 80-84, global
# steps 308,934 to 324,190, LR from 1.0% to 0.004% of P's peak. They are the inputs of the weight average of the
# cooling's last five epochs, the procedure of Consilium's final network (F4 avg5: epochs 55-59 of its one-cycle,
# avg_perm_F4.sh). The run saves a checkpoint every 10 epochs only, and the exporter removes it after the export.
# last.ckpt is rewritten at every epoch end: each new version is read once it is 60 s old, and copied when its global
# step is the next target.
# Usage: setsid -f nohup bash snap_c2_last5.sh >> /root/valfix/snap_c2_last5.log 2>&1 < /dev/null
set -uo pipefail
V=/root/valfix; PY=/venv/main/bin/python3
CK=/root/c2/run/F/lightning_logs/version_4/checkpoints/last.ckpt
OUT="$V/c2_last5"; mkdir -p "$OUT"
log() { echo "[snap_c2_last5 $(date -u +%H:%M:%S)] $*"; }
step_of() { CUDA_VISIBLE_DEVICES="" "$PY" -c "
import torch
print(torch.load('$1', map_location='cpu', weights_only=False)['global_step'])"; }

seen=$(stat -c %Y "$CK")
for target in 308934 312748 316562 320376 324190; do
  log "waiting for step $target"
  while true; do
    m=$(stat -c %Y "$CK")
    if [ "$m" != "$seen" ] && [ $(( $(date +%s) - m )) -ge 60 ]; then
      seen=$m
      gs=$(step_of "$CK")
      [ -n "$gs" ] || { log "could not read last.ckpt, retrying"; seen=""; sleep 20; continue; }
      [ "$gs" = "$target" ] && break
      if [ "$gs" -gt "$target" ]; then log "missed step $target (last.ckpt already at $gs)"; continue 2; fi
    fi
    sleep 20
  done
  cp "$CK" "$OUT/part.ckpt" && mv "$OUT/part.ckpt" "$OUT/c2_step$target.ckpt"
  log "copied step $target: $(step_of "$OUT/c2_step$target.ckpt")"
done
touch "$V/c2_last5_done"
log "done"
