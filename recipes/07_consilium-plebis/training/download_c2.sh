#!/usr/bin/env bash
# Consilium-2 data (21/10/2026): the whole BT4-relabelled corpus of Consilium (701 GB, download_bt4.sh, includes the
# DFRC file dfrc_n5000.relabel-BT4-tf13tune) plus vondele/rescored/tb5dtm.binpack (0.14 GB, 5-man DTM mates, in
# Stockfish's recipe). Research of 21/10: no other BT4/BT5 relabelled data exists on Hugging Face; raw (non-BT4)
# data is left out (SF nettest #403: relabel-only was better).
# The loader draws files in proportion to their size (parallel_dataloader.h): every position weighs the same,
# small files are not read more often. No file is listed twice.
# Usage: bash download_c2.sh [DEST] [PARALLEL]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DEST="${1:-/data/bt4}"
JOBS="${2:-8}"
mkdir -p "$DEST"
export HF_HUB_ENABLE_HF_TRANSFER=1
python3 -c "import hf_transfer" 2>/dev/null || export HF_HUB_ENABLE_HF_TRANSFER=0
HF=""; for c in hf huggingface-cli; do command -v $c >/dev/null && { HF=$c; break; }; done
[ -n "$HF" ] || { echo "missing hf: pip install -U 'huggingface_hub[hf_transfer]'"; exit 1; }
echo "=== tb5dtm"
$HF download vondele/rescored tb5dtm.binpack --repo-type dataset --local-dir "$DEST/_tb" >/dev/null
mv -f "$DEST/_tb/tb5dtm.binpack" "$DEST/tb5dtm.binpack" && rm -rf "$DEST/_tb"
ls -la "$DEST/tb5dtm.binpack"
echo "=== BT4 corpus (701 GB)"
bash "$HERE/download_bt4.sh" "$DEST" 850 "$JOBS"
echo "=== total"; du -sh "$DEST"; ls "$DEST"/*.binpack | wc -l
