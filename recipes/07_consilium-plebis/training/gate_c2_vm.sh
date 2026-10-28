#!/usr/bin/env bash
# Consilium-2 gates on the VM (21/10/2026), before the real run:
#   gate 2  loader indices (cells, experts, pawn-king) against python-chess          (c2_gate_loader.py)
#   gate 1  integer evaluation of an exported random network against the trainer    (c2_gate_int.py)
#   gate 3  GPU smoke run on all GPUs: 3 short epochs, one learned-routing round at the end of epoch 2, checkpoints,
#           resume for a 4th epoch, export of the last checkpoint, gate 1 on the TRAINED network.
# Usage: bash gate_c2_vm.sh [DEST] [BINPACK]     (BINPACK: one downloaded file for the smoke run)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DEST="${1:-/root/c2}"
BIN="${2:?a downloaded .binpack for the smoke run}"
cd "$DEST/nnue-pytorch"
PY=""
for c in /venv/main/bin/python3 /usr/bin/python3 python3; do
  command -v "$c" >/dev/null 2>&1 && "$c" -c "import torch" >/dev/null 2>&1 && { PY="$c"; break; }
done
NGPU="${NGPU:-$(nvidia-smi -L | grep -c .)}"
FEATURES="Full_Threats+HalfKAv2_hm_X8^+PP_3Wide+PassedPawns"

echo "=== gate 2"
"$PY" c2_gate_loader.py "$HERE/gate_uho600.pgn" 4000 | tail -2
"$PY" c2_gate_loader.py "$HERE/gate_eg400.pgn" 3000 | tail -2
echo "=== gate 1 (random network)"
"$PY" c2_gate_int.py "$HERE/gate_uho600.pgn" 2000 --out "$DEST/random_c2.nnue" | tail -3

echo "=== gate 3 (smoke run, $NGPU GPU)"
SM="$DEST/smoke"; rm -rf "$SM"; mkdir -p "$SM"
COMMON=(--features "$FEATURES" --pawn-king-hidden 256 --l1 1024 --threads 1 --num-workers 16 --accelerator cuda
  --batch-size 65536 --epoch-size 6553600 --lr 4e-4 --one-cycle-steps 1000
  --random-fen-skipping 3 --start-lambda 0.75 --end-lambda 0.75 --lambda-cycle-delta 0
  --router-em-epochs 2 --router-em-sample-batches 8 --router-em-sample-size 8192
  --router-em-transitions "$DEST/nnue-pytorch/c2_transitions.npz"
  --network-save-period 1 --save-last-network True --validation-size 65536 --check-val-every-n-epoch 100)
"$PY" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" ddp_launcher.py train.py "$BIN" \
  "${COMMON[@]}" --max-epochs 3 --default-root-dir "$SM/a" 2>&1 | tail -25
LAST=$(find "$SM/a" -name last.ckpt | head -1)
[ -n "$LAST" ] || { echo "no checkpoint"; exit 1; }
echo "--- router rounds"; find "$SM/a" -name router_em.jsonl -exec python3 -c "
import json,sys
for l in open(sys.argv[1]): d=json.loads(l); d.pop('table'); print(d)" {} \;
echo "--- resume"
"$PY" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" ddp_launcher.py train.py "$BIN" \
  "${COMMON[@]}" --max-epochs 4 --default-root-dir "$SM/b" --resume-from-checkpoint "$LAST" 2>&1 | tail -8
LAST2=$(find "$SM/b" -name last.ckpt | head -1)
echo "--- export + gate 1 on the trained network"
"$PY" serialize.py "$LAST2" "$SM/smoke.nnue" --features "$FEATURES" --pawn-king-hidden 256 --l1 1024 | tail -2
"$PY" c2_gate_int.py "$HERE/gate_uho600.pgn" 2000 --net "$SM/smoke.nnue" | tail -3
echo "=== gates done"
