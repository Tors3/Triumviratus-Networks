#!/usr/bin/env bash
# Converte in .nnue (leb128) ogni checkpoint di $1 che non ha ancora il suo .nnue accanto. Idempotente.
# USO: export_nets.sh /root/moe/run
set -uo pipefail
ROOT="${1:?cartella del run}"
TRAINER="${TRAINER:-/root/moe/nnue-pytorch}"
PY="${PY:-/venv/main/bin/python3}"
FEATURES="Full_Threats+HalfKAv2_hm_P4^+PP_3Wide+PassedPawns"
mkdir -p "$ROOT/nets"
cd "$TRAINER"
find "$ROOT" -name '*.ckpt' -type f ! -name last.ckpt | sort | while read -r ck; do
  rel="${ck#"$ROOT"/}"; phase="${rel%%/*}"   # P o F: la prima cartella sotto ROOT
  out="$ROOT/nets/${phase:-X}_$(basename "$ck" .ckpt).nnue"
  [ -f "$out" ] && continue
  CUDA_VISIBLE_DEVICES="" "$PY" serialize.py "$ck" "$out" --features "$FEATURES" --l1 1024 --ft-compression leb128 \
    && echo "$(date -u +%H:%M) $out" || echo "$(date -u +%H:%M) FALLITO $ck"
done
