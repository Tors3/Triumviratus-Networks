#!/usr/bin/env bash
# Consilium-2 run guard (21/10/2026): restarts P from its last checkpoint when the tmux session "train" is gone (the
# script exited on an error) or the GPUs stay idle for 15 minutes (a hang). At most 3 restarts, and none within
# 30 minutes of the previous one (a crash that repeats needs a person). Stops when P.done exists.
LOG=/root/guard_c2.log; ROOT=/root/c2/run; restarts=0; last=0; idle=0
CMD="P_EPOCHS=550 bash /root/c2pkg/train_c2.sh 2>&1 | tee -a /root/train_c2.log"
say() { echo "[$(date -u +%F\ %H:%M:%S)] $*" >> "$LOG"; }
say "guard started"
while true; do
  sleep 60
  [ -f "$ROOT/P.done" ] && { say "P.done: guard ends"; exit 0; }
  if tmux has-session -t train 2>/dev/null; then
    u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sort -n | tail -1)
    if [ "${u:-0}" -lt 5 ]; then idle=$((idle + 1)); else idle=0; fi
    [ "$idle" -lt 15 ] && continue
    say "GPUs idle for 15 minutes: killing the train session"; tmux kill-session -t train; sleep 30
  fi
  now=$(date +%s)
  if [ "$restarts" -ge 3 ] || [ $((now - last)) -lt 1800 ]; then say "not restarting (restarts=$restarts, last $((now - last)) s ago)"; exit 1; fi
  restarts=$((restarts + 1)); last=$now; idle=0
  echo "=== RESUME by guard $(date -u) (restart $restarts)" >> /root/train_c2.log
  say "restart $restarts"; tmux new -d -s train "$CMD"
done
