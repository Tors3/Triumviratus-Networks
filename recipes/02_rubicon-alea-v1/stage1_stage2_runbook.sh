#!/usr/bin/env bash
# =============================================================================
#  TRIUMVIRATUS 5 — RETE THREATS (SFNNv10), training A STADI come SF (vondele threats.yaml).
#  GPU: Blackwell RTX PRO 6000 (g4-standard-48) SPOT, da ZERO. Master-nuovo nnue-pytorch
#  (Full_Threats nativo). Disco 250GB → dati SCARICATI PER STADIO (no SF-gen+Leela insieme).
#  Eseguire BLOCCO PER BLOCCO (il driver chiede UN reboot, poi si ri-entra e si salta il blocco 2).
#
#  STADI (threats.yaml scalato):
#    Stadio 1 = dati SF-gen (5000-nodi)  | lr 1.5e-3 | lambda 1.0->0.75 | ~250 ep
#    Stadio 2 = dati Leela (test80)      | lr 1.3e-3 | lambda 0.74      | ~950 ep (resume da st1)
#    Stadio 3 = Leela, lr 0.65e-3 (opz., lungo)
# =============================================================================
set -e
WORK="$HOME/triumviratus_nnue"; NP="$WORK/nnue-pytorch"; DATA="$WORK/data"; RUN="$WORK/threats_run"

###############################################################################
# 0) CREA VM (dal LAPTOP, PowerShell) — g4 SPOT
#  gcloud compute instances create trium-nnue-train --project=<PROJECT_ID> `
#    --zone=us-central1-b --machine-type=g4-standard-48 --provisioning-model=SPOT `
#    --instance-termination-action=STOP --image-family=ubuntu-2404-lts-amd64 `
#    --image-project=ubuntu-os-cloud --boot-disk-size=250GB --boot-disk-type=hyperdisk-balanced `   # g4 vuole HYPERDISK, non pd-ssd
#    --maintenance-policy=TERMINATE --metadata enable-oslogin=FALSE
###############################################################################

###############################################################################
# 1) SETUP (da ZERO). Driver Blackwell = -open OBBLIGATORIO (sm_120).
###############################################################################
sudo apt-get update && sudo apt-get install -y build-essential cmake git wget tmux zstd
if ! nvidia-smi >/dev/null 2>&1; then
  sudo apt-get install -y nvidia-driver-580-open      # -open per Blackwell
  echo ">>> REBOOT ORA: sudo reboot — poi ri-entra e RILANCIA da blocco 1 (salta il driver)"; exit 0
fi
nvidia-smi   # deve dire "RTX PRO 6000 Blackwell"
# CUDA 12.8
if [ ! -d /usr/local/cuda-12.8 ]; then
  cd ~ && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-8
fi
export CUDA_PATH=/usr/local/cuda-12.8; export PATH="$CUDA_PATH/bin:$PATH"
grep -q CUDA_PATH ~/.bashrc || echo 'export CUDA_PATH=/usr/local/cuda-12.8; export PATH="$CUDA_PATH/bin:$PATH"' >> ~/.bashrc
# conda + Python 3.12 (PEP695 nel master) + TOS
[ -d ~/miniconda ] || { wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/mc.sh; bash ~/mc.sh -b -p ~/miniconda; }
~/miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
~/miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
~/miniconda/bin/conda create -y -n nnue python=3.12 || true
source ~/miniconda/bin/activate nnue
# master nnue-pytorch (Full_Threats nativo) + torch cu128 + data-loader
mkdir -p "$WORK" && cd "$WORK"
[ -d nnue-pytorch ] || git clone https://github.com/official-stockfish/nnue-pytorch.git
cd "$NP" && pip install --upgrade pip && pip install -r requirements.txt
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
pip install -q gdown "huggingface_hub[hf_transfer]"
bash compile_data_loader.sh
python -c "import torch,data_loader,lightning;print('OK',torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))"
# NB: NIENTE VERSION sed qui — la rete-threats va in formato SFNNv10 NATIVO (la testeremo col motore PORTATO a SF18, non l'attuale SFNNv8).

###############################################################################
# 2) STADIO-1 DATI: SF-gen (5000 nodi) da Google Drive (~135G il set pieno; qui i 2 pubblici)
###############################################################################
mkdir -p "$DATA" && cd "$DATA"
# ⭐ TUTTI e 5 i file SF-gen stadio-1 sono su KAGGLE (vondele) — niente Drive (che si blocca). ~135G, sta nei 250GB.
# Serve kaggle.json: kaggle.com Settings -> Create New Token -> scp in /tmp -> mkdir ~/.kaggle; cp; chmod 600.
for slug in nodes5000pv2-u-uho dfrc-u-n5000 multinet-u-pv-2-u-diff-100-u-nodes-5000 \
            data-u-pv-2-u-diff-100-u-nodes-5000 wrongisright-u-nodes5000pv2; do
  kaggle datasets download "joostvandevondele/$slug"
done
unzip -o '*.zip' && rm -f *.zip
ls -lah "$DATA"/*.binpack; df -h /

###############################################################################
# 3) STADIO-1 TRAINING (SF-gen, lr 1.5e-3, lambda 1.0->0.75)
###############################################################################
cd "$NP" && source ~/miniconda/bin/activate nnue
# ⚠️ GOTCHA: torch 2.11+/cu128 + torch.compile(dynamo) crasha sul self.log del master
#   ('method' object has no attribute '__dict__') → DISABILITARE compile con TORCHDYNAMO_DISABLE=1 (eager, ok).
TORCHDYNAMO_DISABLE=1 nohup python train.py "$DATA"/*.binpack \
  --gpus 0 --l1 1024 --l2 31 --l3 32 --features "Full_Threats+HalfKAv2_hm^" \
  --start-lambda 1.0 --end-lambda 0.75 --lr 1.5e-3 --one-cycle-steps 381500 \
  --batch-size 65536 --num-workers "$(nproc)" --epoch-size 100000000 --max-epochs 250 --network-save-period 25 \
  --factorized-weight-decay 0.001 --early-fen-skipping 18 --random-fen-skipping 10 \
  --default-root-dir "$RUN/stage1" > "$HOME/threats_s1.log" 2>&1 &
sleep 30 && tail -30 "$HOME/threats_s1.log"
# (OOM su 65536 -> batch 32768 + one-cycle-steps 763000)

###############################################################################
# 4) STADIO-2 (dopo st1): libera disco, scarica Leela test80, RESUME dai pesi st1
###############################################################################
# serializza i pesi st1 -> .pt (seed per st2)
#   CK=$(ls -t "$RUN/stage1"/lightning_logs/version_*/checkpoints/last.ckpt | head -1)
#   python serialize.py "$CK" "$WORK/stage1.pt" --features "Full_Threats+HalfKAv2_hm^"
# libera i dati SF-gen e scarica Leela (12 mesi test80 2023+2024):
#   rm -f "$DATA"/*.binpack
#   export HF_XET_HIGH_PERFORMANCE=1; cd "$DATA"
#   for m in 06-jun 07-jul 09-sep 10-oct 11-nov 12-dec; do hf download linrock/test80-2023 test80-2023-$m-2tb7p.min-v2.v6.binpack.zst --repo-type dataset --local-dir . && zstd -d --rm test80-2023-$m-*.zst; done
#   for m in 01-jan 02-feb 03-mar 04-apr 05-may 06-jun; do hf download linrock/test80-2024 test80-2024-$m-2tb7p.min-v2.v6.binpack.zst --repo-type dataset --local-dir . && zstd -d --rm test80-2024-$m-*.zst; done
# train st2 (resume pesi st1, lr 1.3e-3, lambda 0.74):
#   nohup python train.py "$DATA"/*.binpack --resume-from-model "$WORK/stage1.pt" \
#     --gpus 0 --l1 1024 --l2 31 --l3 32 --features "Full_Threats+HalfKAv2_hm^" \
#     --start-lambda 0.74 --end-lambda 0.74 --lr 1.3e-3 --one-cycle-steps 1449700 \
#     --batch-size 65536 --num-workers $(nproc) --epoch-size 100000000 --max-epochs 950 --network-save-period 50 \
#     --factorized-weight-decay 0.001 --early-fen-skipping 18 --random-fen-skipping 10 \
#     --default-root-dir "$RUN/stage2" > ~/threats_s2.log 2>&1 &
# STADIO-3 (opz., lungo): come st2 ma lr 0.65e-3, one-cycle-steps 4578000, max-epochs 3000.

###############################################################################
# 5) SERIALIZE finale (dopo lo stadio scelto) -> .nnue SFNNv10, poi PORT motore per testarla
#   python serialize.py <last.ckpt> "$WORK/threats_v1.nnue" --features "Full_Threats+HalfKAv2_hm^"
#   scp al laptop. NB: serve il motore PORTATO a SF18 (inference Full_Threats) per caricarla.
###############################################################################
echo "Runbook caricato. Esegui blocco-per-blocco: 1 (reboot) -> 1 (resto) -> 2 -> 3. Poi 4 (stadio2) quando st1 finisce."
