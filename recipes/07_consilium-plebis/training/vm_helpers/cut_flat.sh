#!/usr/bin/env bash
# 27/10/2026: cut 10 epochs from F's flat 15% stretch (author). Wait for F's next end-of-epoch checkpoint, stop F
# (torchrun by PID; train_c2.sh then exits under set -e and its EXIT trap stops the exporter), keep a copy of the
# checkpoint, resume with start_F2.sh.
# Usage: nohup setsid bash cut_flat.sh <torchrun pid> <train_c2.sh pid> <exporter pid> >> /root/valfix/cut_flat.log 2>&1 &
TR=$1; SH=$2; EX=$3
CK=/root/c2/run/F/lightning_logs/version_0/checkpoints/last.ckpt
TOTAL=228840   # 60 epochs of 3814 steps, the sum of the NNUE_LR_PIECES segments of start_F2.sh
log() { echo "[cut_flat $(date -u +%H:%M:%S)] $*"; }

grep -q "torch.distributed.run" /proc/$TR/cmdline 2>/dev/null || { log "pid $TR is not torchrun, stop"; exit 1; }
T0=$(stat -c %Y $CK)
log "waiting for a new $CK (now $(date -u -d @$T0 +%H:%M:%S))"
while [ "$(stat -c %Y $CK)" = "$T0" ]; do sleep 5; done
s1=0; s2=$(stat -c %s $CK)
while [ "$s1" != "$s2" ]; do sleep 10; s1=$s2; s2=$(stat -c %s $CK); done
sleep 20
log "checkpoint written ($s2 bytes), stopping torchrun $TR"
kill -TERM "$TR"
for i in $(seq 1 60); do kill -0 "$TR" 2>/dev/null || break; sleep 5; done
kill -0 "$TR" 2>/dev/null && { log "torchrun still alive after 300 s, stop"; exit 1; }
sleep 10
kill -0 "$SH" 2>/dev/null && { log "train_c2.sh $SH still alive, stopping it"; kill -TERM "$SH"; }
kill -0 "$EX" 2>/dev/null && { log "exporter $EX still alive, stopping it"; kill -TERM "$EX"; }
for i in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
  [ "$used" -lt 1000 ] && break; sleep 5
done
log "GPU memory used: $used MiB"
[ "$used" -lt 1000 ] || { log "GPUs not free, stop"; exit 1; }

read EP GS < <(cd /root/c2/nnue-pytorch && /venv/main/bin/python3 -c "
import torch
c = torch.load('$CK', map_location='cpu', weights_only=False)
print(c['epoch'], c['global_step'])")
log "checkpoint epoch $EP, global step $GS"
[ $(( (TOTAL - GS) % 3814 )) -eq 0 ] || { log "steps left $((TOTAL - GS)) not a whole number of epochs, stop"; exit 1; }
F_EPOCHS=$(( EP + (TOTAL - GS) / 3814 ))
free=$(df --output=avail -BG /root | tail -1 | tr -dc 0-9)
if [ "$free" -gt 30 ]; then cp "$CK" "/root/valfix/F_last_step${GS}.ckpt" && log "copy: /root/valfix/F_last_step${GS}.ckpt"; fi
log "resuming F with F_EPOCHS=$F_EPOCHS"
F_EPOCHS=$F_EPOCHS nohup bash /root/valfix/start_F2.sh >> /root/train_c2.log 2>&1 < /dev/null &
log "started pid $!"
