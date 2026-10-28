#!/usr/bin/env bash
# Consilium-2: export_c2.sh without the removal of exported checkpoints (27/10/2026, tail of F's second cycle: the
# final network is the average of the last five epoch checkpoints, so they must stay). Every checkpoint of $1 without
# its .nnue gets one (int16 parts leb128, int8 parts raw). Idempotent.
# Usage: export_c2_keep.sh /root/c2/run
set -uo pipefail
ROOT="${1:?run folder}"
TRAINER="${TRAINER:-/root/c2/nnue-pytorch}"
PY="${PY:-$(command -v /venv/main/bin/python3 || command -v python3)}"
FEATURES="Full_Threats+HalfKAv2_hm_X8^+PP_3Wide+PassedPawns"
mkdir -p "$ROOT/nets"
cd "$TRAINER"
find "$ROOT" -name '*.ckpt' -type f ! -name last.ckpt | sort | while read -r ck; do
  rel="${ck#"$ROOT"/}"; phase="${rel%%/*}"
  out="$ROOT/nets/${phase:-X}_$(basename "$ck" .ckpt).nnue"
  [ -f "$out" ] && continue
  if CUDA_VISIBLE_DEVICES="" "$PY" serialize.py "$ck" "$out" --features "$FEATURES" --pawn-king-hidden 256 --l1 1024 \
      --ft-compression leb128; then
    echo "$(date -u +%H:%M) $out (checkpoint kept)"
  else
    echo "$(date -u +%H:%M) FAILED $ck"
  fi
done
