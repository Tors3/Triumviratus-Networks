#!/usr/bin/env bash
# Serializza il last.ckpt CORRENTE del run in un .nnue (leb128), senza disturbare il training (28/09/2026).
# last.ckpt viene riscritto a ogni epoca (~3,4 min): si aspetta che sia fermo da 20 s, se ne fa una copia e si
# serializza la copia sulla CPU (CUDA_VISIBLE_DEVICES vuoto). Stampa il nome del .nnue creato in $ROOT/nets.
set -euo pipefail
ROOT="${ROOT:-/root/moe/run}"
TRAINER="${TRAINER:-/root/moe/nnue-pytorch}"
PY="${PY:-/venv/main/bin/python3}"
PHASE="${1:-P}"
L=$(ls -t "$ROOT/$PHASE"/lightning_logs/*/checkpoints/last.ckpt | head -1)
while [ $(( $(date +%s) - $(stat -c %Y "$L") )) -lt 20 ]; do sleep 5; done
TMP=/root/moe/last_copy.ckpt
cp "$L" "$TMP"
E=$(date -u +%m%d_%H%M)
OUT="$ROOT/nets/last_${PHASE}_$E.nnue"
mkdir -p "$ROOT/nets"
cd "$TRAINER"
CUDA_VISIBLE_DEVICES="" "$PY" serialize.py "$TMP" "$OUT" --features "Full_Threats+HalfKAv2_hm_P4^+PP_3Wide+PassedPawns" \
  --l1 1024 --ft-compression leb128 > /root/moe/serialize_last.log 2>&1
rm -f "$TMP"
echo "$(basename "$OUT")"
