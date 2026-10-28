#!/usr/bin/env bash
# Consilium-2: every checkpoint of $1 without its .nnue gets one (int16 parts leb128, int8 parts raw). Idempotent.
# Usage: export_c2.sh /root/c2/run
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
    echo "$(date -u +%H:%M) $out"
    # 1 TB disk, ~5 GB per checkpoint: once exported, keep the checkpoint only every 100 epochs (last.ckpt is kept
    # by the trainer). The .nnue stays for every saved epoch.
    ep=$(basename "$ck" | sed -n 's/^epoch=\([0-9]*\)-.*/\1/p')
    if [ -n "$ep" ] && [ $(( (ep + 1) % 100 )) -ne 0 ]; then rm -f "$ck" && echo "  removed $ck"; fi
  else
    echo "$(date -u +%H:%M) FAILED $ck"
  fi
done
