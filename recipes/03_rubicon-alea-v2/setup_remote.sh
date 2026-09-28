#!/usr/bin/env bash
# setup_remote.sh — prepara la VM GCP (Debian/Ubuntu, driver NVIDIA gia' presenti
# sulle immagini Deep Learning). Uso:  bash gcp/setup_remote.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # root del pacchetto estratto

sudo apt-get update -qq
sudo apt-get install -y -qq python3.12 python3.12-venv cmake g++ || \
  sudo apt-get install -y -qq python3 python3-venv cmake g++   # fallback: 3.12 gia' default

PY=$(command -v python3.12 || command -v python3)
$PY -m venv venv
source venv/bin/activate

pip install -q -U pip
pip install -q -r nnue-pytorch/requirements.txt
pip install -q lightning tyro

# loader C++ (Release semplice: il PGO dello script ufficiale e' un +% opzionale)
cd nnue-pytorch
mkdir -p build
g++ -shared -fPIC -O3 -std=c++20 -march=native -DNDEBUG \
    data_loader/cpp/training_data_loader.cpp data_loader/cpp/training_data_loader_abi.cpp \
    -o build/libtraining_data_loader.so

# sanity: registry + loader + graft
python - <<'EOF'
import sys; sys.path.insert(0, '.')
from model.modules.features import get_feature_cls
get_feature_cls("Full_Threats+HalfKAv2_hm^+PawnPair")
import data_loader  # carica la .so
print("[OK] trainer + loader + PawnPair pronti")
EOF
cd ..
python graft/verify_graft.py   # self-check [A] del riferimento
echo "[OK] setup completo. Ora: bash gcp/train_gcp.sh l4|blackwell /path/dati"
