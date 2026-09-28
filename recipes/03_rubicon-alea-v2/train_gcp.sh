#!/usr/bin/env bash
# train_gcp.sh — run rubicon-alea-v2 (graft PawnPair) su GCP.
# USO:  bash gcp/train_gcp.sh blackwell /home/USER/data     # RTX PRO 6000 (g4)
#       bash gcp/train_gcp.sh l4 /home/USER/data            # L4 (g2)
# DATA deve contenere:  own/*.binpack  (self-play nostro)  +  t80/**/*.binpack  (T80 .min)
set -euo pipefail
GPU=${1:?l4|blackwell}
DATA=${2:?cartella dati (con own/ e t80/)}
OWN_REPEAT=${OWN_REPEAT:-10}   # quante volte ripetere il binpack own nella lista
                               # (loader campiona ~proporzionale alla size: own 2.3GB vs
                               # T80 ~327GB -> 10x ≈ 6.5% del mix, prudente anti-overfit.)
cd "$(dirname "$0")/.."
source venv/bin/activate
# torch.compile (dynamo) si rompe con torch 2.9 + questo Lightning (InternalTorch
# DynamoError sul tracing di self.log). Eager e' stabile e il loader e' CPU-bound
# comunque -> perdita di velocita' minima. Disabilitato per far girare il run.
export TORCHDYNAMO_DISABLE=1
RUNDIR="$(pwd)/runs/alea_v2_gcp"
cd nnue-pytorch

case "$GPU" in
  l4)        BATCH=32768; WORKERS=8  ;;
  blackwell) BATCH=65536; WORKERS=12 ;;
  *) echo "gpu sconosciuta: $GPU"; exit 1 ;;
esac

# --- dataset: T80 (una volta, ricorsivo) + own (ripetuto OWN_REPEAT volte) ---
mapfile -t T80 < <(find "$DATA/t80" -name "*.binpack" 2>/dev/null | sort)
mapfile -t OWN < <(find "$DATA/own" -name "*.binpack" 2>/dev/null | sort)
[ "${#T80[@]}" -gt 0 ] || { echo "!! nessun T80 binpack in $DATA/t80"; exit 1; }
[ "${#OWN[@]}" -gt 0 ] || { echo "!! nessun own binpack in $DATA/own"; exit 1; }
BINPACKS=("${T80[@]}")
for _ in $(seq 1 "$OWN_REPEAT"); do BINPACKS+=("${OWN[@]}"); done
echo "[i] T80=${#T80[@]} file, own=${#OWN[@]} file x${OWN_REPEAT} -> ${#BINPACKS[@]} voci totali"

# --- auto-resume: se esiste un last.ckpt di un run precedente (preemption spot),
#     riprendi lo stato COMPLETO da li'; altrimenti primo avvio dal graft. ---
LAST=$(find "$RUNDIR" -name 'last.ckpt' 2>/dev/null | sort | tail -1 || true)
if [ -n "$LAST" ]; then
  RESUME=(--resume-from-checkpoint "$LAST")
  echo "[resume] riprendo da $LAST (preemption-safe)"
else
  RESUME=(--resume-from-model ../graft/alea_v2_grafted.pt)
  echo "[start] primo avvio dal graft zero-init"
fi

# Run lungo (800 epoche): PawnPair parte da zero -> molte epoche a lr utile.
# gamma 0.997 (0.997^800≈0.09, lr finale vivo). pawnpair-lr 1e-3 = two-group LR
# (blocco nuovo from-scratch, resto a 1e-4; patch verificata su .pt reale).
# save-period 20 + save-top-k -1: salva ogni 20 epoche e li TIENE TUTTI (40 ckpt
# ~60GB; monitor=None NON permette top-k>1, e non c'e' validation set -> -1 e' l'unica
# opzione multi-checkpoint valida). Serve per l'over_last a fine run.
mkdir -p "$RUNDIR"   # senno' il tee del log sotto fallisce (dir inesistente)
python train.py "${BINPACKS[@]}" \
    --features "Full_Threats+HalfKAv2_hm^+PawnPair" \
    "${RESUME[@]}" \
    --max-epochs 800 \
    --epoch-size 100000000 \
    --batch-size $BATCH \
    --num-workers $WORKERS \
    --lr 1e-4 \
    --pawnpair-lr 1e-3 \
    --gamma 0.997 \
    --start-lambda 0.75 --end-lambda 0.75 \
    --validation-size 1000000 \
    --network-save-period 20 \
    --save-top-k -1 \
    --default-root-dir "$RUNDIR" \
    2>&1 | tee -a "$RUNDIR.log"

# a fine run: serializzare il checkpoint scelto (ultime ~5 epoche + media over_last)
#   python serialize.py <ckpt> ../nn-rubicon-alea-v2.nnue --features "Full_Threats+HalfKAv2_hm^+PawnPair"
