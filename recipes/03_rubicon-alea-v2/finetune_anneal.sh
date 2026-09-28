#!/usr/bin/env bash
# finetune_anneal.sh — anneal finale LR-ripido a partire da un checkpoint del run
# principale. USATO PER LA v2 (2026-07-17).
#
# PERCHE': il run principale (train_gcp.sh, gamma 0.997 tarato su 800 epoche) aveva
# la val PIATTA da ~ep80 con LR ancora vivo (0.997^459 = 25%) -> non era LR-starvation
# ma tetto-label. Ipotesi (utente): un decay RIPIDO puo' comunque assestare i pesi in
# un minimo migliore. Verdetto: la val floor e' scesa pochissimo (~1sigma) e il gate
# overlast-vs-ep459 ha dato +3.81±7.67 = parita'. L'anneal NON ha dato il salto
# sperato, ma e' costato ~8h ed e' servito a saperlo.
#
# ⚠️ LEZIONE PER I RUN FUTURI: NON serve una fase di anneal separata. Basta tarare
# gamma sulle epoche REALI del run (es. 100 ep -> gamma 0.97, 0.97^100 = 0.05):
# l'anneal e' dentro il decay. Vedi README.md.
#
# USO:  bash gcp/finetune_anneal.sh blackwell /home/USER/data /path/to/start.pt
#   $3 = modello di partenza (.pt picklato; da un .ckpt: serialize.py <ckpt> <out.pt> --features ...)
#
# --resume-from-model (NON --resume-from-checkpoint): carica solo i PESI e crea un
# optimizer/scheduler FRESCHI -> il nuovo lr/gamma partono da zero. Con
# --resume-from-checkpoint si ripristinerebbe lo scheduler vecchio (gamma 0.997)
# e il cambio di gamma sarebbe ignorato/rotto (ExponentialLR ricalcola da last_epoch).
set -euo pipefail
GPU=${1:?l4|blackwell}; DATA=${2:?data dir}; PT=${3:?start .pt}
OWN_REPEAT=${OWN_REPEAT:-10}
cd "$(dirname "$0")/.."; source venv/bin/activate; export TORCHDYNAMO_DISABLE=1
RUNDIR="$(pwd)/runs/alea_v2_anneal"
cd nnue-pytorch
case "$GPU" in l4) BATCH=32768;WORKERS=8;; blackwell) BATCH=65536;WORKERS=12;; *) echo "gpu sconosciuta: $GPU";exit 1;; esac
mapfile -t T80 < <(find "$DATA/t80" -name "*.binpack"|sort)
mapfile -t OWN < <(find "$DATA/own" -name "*.binpack"|sort)
[ "${#T80[@]}" -gt 0 ] && [ "${#OWN[@]}" -gt 0 ] || { echo "!! binpack mancanti"; exit 1; }
BINPACKS=("${T80[@]}"); for _ in $(seq 1 "$OWN_REPEAT"); do BINPACKS+=("${OWN[@]}"); done
echo "[i] ${#BINPACKS[@]} binpack | resume-from-model $PT | 100 ep | lr 2.5e-5/2.5e-4 gamma 0.95 + SWA@60"
mkdir -p "$RUNDIR"
# lr 2.5e-5 / 2.5e-4 = ESATTAMENTE l'LR che il run principale aveva raggiunto a ep459
# (1e-4 * 0.997^459 e 1e-3 * 0.997^459) -> l'anneal riparte da li' e scende ripido.
python train.py "${BINPACKS[@]}" \
  --features "Full_Threats+HalfKAv2_hm^+PawnPair" \
  --resume-from-model "$PT" \
  --max-epochs 100 --epoch-size 100000000 --batch-size $BATCH --num-workers $WORKERS \
  --lr 2.5e-5 --pawnpair-lr 2.5e-4 --gamma 0.95 --swa-start-epoch 60 \
  --start-lambda 0.75 --end-lambda 0.75 --validation-size 1000000 \
  --network-save-period 10 --save-top-k -1 --default-root-dir "$RUNDIR" \
  2>&1 | tee -a "$RUNDIR.log"
# NB: il run reale e' stato FERMATO a ~ep50 (val piatta, SWA@60 mai raggiunto) e la
# chiusura e' stata fatta con graft/overlast.py sugli ultimi 3 checkpoint.
