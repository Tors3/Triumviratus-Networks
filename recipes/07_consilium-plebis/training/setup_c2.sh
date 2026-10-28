#!/usr/bin/env bash
# Consilium-2 (21/10/2026): prepares the VM. Trainer = our patched nnue-pytorch shipped as a tarball (no git clone).
#   1) Python packages (uv if present), Hugging Face CLI with hf_transfer for the download;
#   2) trainer unpacked in $DEST/nnue-pytorch, data loader built with cmake (Release);
#   3) checks: torch + CUDA, GPU count, CPU threads, free disk.
# Usage: bash setup_c2.sh [DEST]      (DEST default /root/c2; the tarball nnue-pytorch-c2.tar.gz next to this script)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DEST="${1:-/root/c2}"
mkdir -p "$DEST"
PY=""
for c in /venv/main/bin/python3 /usr/bin/python3 python3; do
  command -v "$c" >/dev/null 2>&1 && "$c" -c "import torch" >/dev/null 2>&1 && { PY="$c"; break; }
done
[ -n "$PY" ] || { echo "no python with torch"; exit 1; }
echo "=== python: $PY ($($PY -c 'import torch;print(torch.__version__, torch.version.cuda)'))"

echo "=== trainer"
tar -xzf "$HERE/nnue-pytorch-c2.tar.gz" -C "$DEST"
cd "$DEST/nnue-pytorch"
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$PY" -r requirements.txt "cupy-cuda12x<14" "huggingface_hub[hf_transfer]"
else
  "$PY" -m pip install -q -r requirements.txt "cupy-cuda12x<14" "huggingface_hub[hf_transfer]"
fi

echo "=== data loader (cmake, Release)"
command -v cmake >/dev/null 2>&1 || "$PY" -m pip install -q cmake
rm -rf build
cmake -S data_loader/cpp -B build -DCMAKE_BUILD_TYPE=Release > build_cmake.log 2>&1
cmake --build build -j"$(nproc)" --target training_data_loader > build_make.log 2>&1
ls build/*training_data_loader* >/dev/null || { echo "loader build failed (build_make.log)"; exit 1; }

echo "=== machine"
nvidia-smi -L
echo "CPU threads: $(nproc)"; free -g | head -2; df -h / /data 2>/dev/null | tail -n +1
"$PY" -c "import torch;print('cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())"
echo "=== ready. Next: bash download_c2.sh /data/bt4 ; bash gate_c2_vm.sh"
