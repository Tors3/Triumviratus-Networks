#!/usr/bin/env bash
# SPRT of the average of the continuation's last 3 epochs against P's epoch 496 (author, 28/10/2026 00:50 Italian:
# "sprt fra fine fase P e media delle 3"), 30k nodes as the F-vs-P tests of the main chat, on the VM's CPU quota once
# the averages' t91 has ended (avg_fr3.sh). Benches checked by hand at 23:55 UTC: P ep496 135026, avg3 245702.
# Usage: setsid -f nohup bash launch_sprt_avg3.sh >> /root/valfix/sprt_avg3.log 2>&1 < /dev/null
set -uo pipefail
V=/root/valfix; KIT=/root/linux_kit
log() { echo "[sprt_avg3 $(date -u +%H:%M:%S)] $*"; }
until [ -f "$V/t91_fr3avg_done" ]; do sleep 20; done
log "averages' t91 done, launching the SPRT"
cd "$KIT"
M=C2Fut6=22,C2Lmr6=-72,C2Nmp6=13,C2Rfp6=4,C2Rfp7=0,C2Fut7=-1,C2Lmr7=5,C2Nmp7=-3,C2Rfp5=-1,C2Fut5=2,C2Lmr5=-6,C2Nmp5=2
CAND=bin/triumviratus_pgo CAND_BENCH=245702 CAND_OPTS="EvalFile=$KIT/nets/FR3_avg3_step240282.nnue,$M" \
BASE=bin/triumviratus_pgo BASE_BENCH=135026 BASE_OPTS="EvalFile=$KIT/nets/P_last_ep496.nnue,$M" \
TAG=c2FR3avg3_vs_P496 NODES=30000 SEED=50211 ./run_sprt_two.sh
