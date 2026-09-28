#!/usr/bin/env bash
# FASE 2 — rete 7.0 "legio-septima": rifinitura su dati Leela, seminata dalla rete di fase 1.
#
# La struttura a due fasi e' l'unica cosa che SF afferma esplicitamente:
#   "it is usually best to FIRST train a network with datasets generated with Stockfish
#    (depth 9, nodes 5000), and THEN retrain an already good network using various
#    Lc0-derived datasets"
# e allenare SOLO su Lc0 e' peggio (buchi di copertura posizionale). Per questo la fase 1
# non contiene dati Leela e la fase 2 non riparte da zero.
#
# USO:  ./train_phase2.sh <MODELLO_FASE1.pt> [DATA_DIR] [NUM_WORKERS]
set -euo pipefail

MODEL="${1:?serve il .pt finale della fase 1, oppure il suo over_last}"
DATA="${2:-/data/phase2}"
WORKERS="${3:-}"
FEATURES="Full_Threats+HalfKAv2_hm^+PP_3Wide+PassedPawns"
# Il trainer NON e' per forza accanto a questo script: setup_vm.sh lo installa in
# <DEST>/nnue-pytorch, mentre lo script puo' stare altrove (es. copiato in /root).
# Ordine: $TRAINER esplicito -> accanto allo script -> percorsi noti di setup_vm.sh.
TRAINER="${TRAINER:-}"
if [ -z "$TRAINER" ]; then
  for c in "$(dirname "$0")/nnue-pytorch"            "$HOME/legio-septima/nnue-pytorch"            /root/legio-septima/nnue-pytorch; do
    [ -f "$c/train.py" ] && { TRAINER="$c"; break; }
  done
fi
[ -n "$TRAINER" ] && [ -f "$TRAINER/train.py" ] || {
  echo "ERRORE: trainer non trovato. Passalo esplicito:  TRAINER=/percorso/nnue-pytorch $0 ..." >&2
  exit 1
}
echo "[trainer] $TRAINER"

# VM 48 vCPU -> 24 worker. Non (nproc-4): oltre un certo punto i worker si contendono
# memoria e cache e il throughput scende invece di salire. 24 lascia meta' macchina al
# resto (il loader e' gia' multithread al suo interno). Da rivedere solo se la prima
# epoca mostra utilizzo GPU basso: in quel caso il collo e' il loader e si puo' salire.
WORKERS="${WORKERS:-24}"


# --- RIPRESA AUTOMATICA (istanze spot) ------------------------------------------------
# last.ckpt e' scritto alla FINE DI OGNI EPOCA (save_last=True), quindi una preemption
# costa al massimo un'epoca. Ma la dir dei checkpoint e' lightning_logs/version_N con N
# che si INCREMENTA a ogni riavvio: senza --default-root-dir fisso + ricerca esplicita,
# al riavvio il run ripartirebbe da zero senza dirlo. Qui lo rendiamo esplicito.
ROOT="${ROOT:-$DATA/../run_phase2}"
mkdir -p "$ROOT"
RESUME=""
LAST=$(find "$ROOT" -name last.ckpt -type f -newermt '@0' -exec ls -1t {} + 2>/dev/null | head -1)
if [ -n "$LAST" ]; then
  RESUME="--resume-from-checkpoint $LAST"
  echo "[RIPRESA] trovato $LAST"
  echo "[RIPRESA] $(stat -c '%y' "$LAST" 2>/dev/null)"
else
  echo "[NUOVO] nessun checkpoint in $ROOT: parto da zero"
fi

cd "$TRAINER"

# Corpus fase 2: Leela rietichettato BT4 + T91 2026. Il glob tiene fuori le varianti .q
# finche' non sappiamo cosa sono.
#
# 🔴 T91: SOLO i mesi recenti. §13 di PROSPETTIVE_RETE_7.0.md: T91 e' la run che PRODUCE i
# net BT4 (non il contrario), quindi ogni mese e' generato dal net di quel mese e la qualita'
# CRESCE col mese. E i T91 non sono rietichettati: le loro label vengono da ricerche a 800
# visite contro le 10.000 dei T80 relabeled. Limitarsi ai mesi migliori tiene l'esposizione a
# quell'eterogeneita' bassa. Aggiungi/togli mesi qui.
T91_MESI=( May June )

shopt -s nullglob
DATASETS=( "$DATA"/*.relabel-BT4-tf13tune.binpack )
for m in "${T91_MESI[@]}"; do DATASETS+=( "$DATA"/T91-2026-"$m"-*.binpack ); done

# 🔴 FILTRO DI DIMENSIONE — non e' cosmesi, e' la correzione di un difetto misurato.
# Il data loader sceglie il file UNIFORMEMENTE A CASO (parallel_dataloader.h:145
# `fileId = local_dist(prng)`), NON proporzionalmente alla dimensione, e con cyclic=true un
# file esaurito riparte da capo. Quindi ogni file pesa uguale e i PICCOLI vengono ri-lettti
# molte volte: a 800x100M consegnate e skip 3 si attraversano ~320 G posizioni, cioe' ~15 G
# per file su 21 file. Un binpack da 3 GB (1,2 G posizioni) verrebbe attraversato 12,7 VOLTE
# -> le stesse posizioni ripetute a raffica, che e' esattamente la correlazione che il
# random-fen-skipping serve a evitare. Sopra i ~10 GB il peggio scende a 3,8 attraversamenti.
# Filtrare per BYTE e non per nome cosi' regge anche sui corpus futuri.
MIN_GB="${MIN_BINPACK_GB:-9}"
mapfile -t DATASETS < <(
  printf '%s\n' "${DATASETS[@]}" | grep -v '\.q\.binpack$' | while read -r f; do
    sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
    if [ "$sz" -ge $((MIN_GB * 1000000000)) ]; then
      echo "$f"
    else
      echo "  [escluso: $(( sz / 1000000000 )) GB < ${MIN_GB}] $(basename "$f")" >&2
    fi
  done
)
[ "${#DATASETS[@]}" -gt 0 ] || { echo "nessun binpack >= ${MIN_GB} GB in $DATA"; exit 1; }
echo "dataset fase 2: ${#DATASETS[@]} file"
printf '  %s\n' "${DATASETS[@]}"

EPOCHS=800
EPOCH_SIZE=100000000
BATCH=131072
# 🔴 lambda_schedule_steps e' in STEP, non in epoche, quindi dipende dal batch: calcolarlo
# qui invece di cablarlo, o cambiando il batch l'anneal cadrebbe nel punto sbagliato IN
# SILENZIO. Anneal completato in LAMBDA_EPOCHS, poi tenuto fermo (ratio satura a 1.0).
LAMBDA_EPOCHS=100
LAMBDA_STEPS=$(( LAMBDA_EPOCHS * EPOCH_SIZE / BATCH ))
echo "lambda: 0.79 -> 0.75 in $LAMBDA_EPOCHS epoche ($LAMBDA_STEPS step), poi fisso"

# 🔴 Su molte immagini (vast.ai incluse) esiste solo `python3`: un `python` nudo muore con
# "command not found" DOPO aver stampato tutta la lista dei dataset, quindi sembra che sia
# partito. Si sceglie il primo interprete che ha davvero torch, non il primo che esiste.
# 🔴 TRAPPOLA vast.ai, costata due lanci falliti il 29/07. Sull'immagine ci sono DUE python:
#   /usr/bin/python3        <- setup_vm.sh installa torch QUI
#   /venv/main/bin/python3  <- attivato AUTOMATICAMENTE nelle shell di login, e NON ha torch
# Quindi lo stesso script funziona da `ssh host cmd` (shell non-login, niente venv) e muore
# da terminale interattivo, dove `python3` risolve al venv. Sintomo ingannevole: la lista dei
# dataset viene stampata tutta e l'errore arriva DOPO, quindi sembra che il training sia
# partito. Si prova ogni candidato con un vero `import torch`: e' l'unico test che non mente.
PY=""
for c in /usr/bin/python3 "${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}" \
         /venv/main/bin/python python3 python; do
  [ -n "$c" ] || continue
  command -v "$c" >/dev/null 2>&1 && "$c" -c "import torch" >/dev/null 2>&1 && { PY="$c"; break; }
done
[ -n "$PY" ] || { echo "nessun interprete python con torch trovato (provati: /usr/bin/python3, venv, python3)"; exit 1; }
echo "interprete: $PY ($($PY --version 2>&1))"

# 🔴 --gpus DA SOLO NON FA DDP: train.py legge world_size dall'ambiente distribuito e senza
# torchrun resta 1, quindi girerebbe su UNA scheda sola (verificato in fase 1 il 28/07:
# GPU 0 al 96%, le altre tre a 0%). Su 4 GPU sono 130 ore invece di 33.
# In DDP `--gpus` non serve: ogni rank usa la sua LOCAL_RANK. ddp_launcher.py si occupa
# anche di affinity e thread.
GPUS="${GPUS:-$(nvidia-smi -L 2>/dev/null | grep -c . || echo 1)}"
NGPU="$GPUS"
if [ "$NGPU" -gt 1 ]; then
  echo "[DDP] $NGPU GPU via torchrun"
  LAUNCH=("$PY" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" ddp_launcher.py train.py)
  GPUARG=()
else
  echo "[single-GPU]"
  LAUNCH=("$PY" train.py)
  GPUARG=(--gpus 0)
fi

# NNUE_WORKER_THREAD_RATIO: quota di thread del loader dedicati alla costruzione delle
# feature invece che alla lettura. MISURATO in fase 1 (28/07) e CONTRO-INTUITIVO: piu' basso
# e' meglio — 0.05 -> 399k pos/s, 0.14 -> 318k, 0.40 -> 239k, 0.70 -> 169k. La mia previsione
# iniziale era l'opposto. Non alzarlo senza rimisurare.
export NNUE_WORKER_THREAD_RATIO="${NNUE_WORKER_THREAD_RATIO:-0.05}"
echo "loader: NNUE_WORKER_THREAD_RATIO=$NNUE_WORKER_THREAD_RATIO  num-workers=$WORKERS"

"${LAUNCH[@]}" "${DATASETS[@]}" \
  "${GPUARG[@]}" \
  --threads "${TORCH_THREADS:-1}" \
  --features "$FEATURES" \
  --resume-from-model "$MODEL" \
  --max-epochs "$EPOCHS" \
  --epoch-size "$EPOCH_SIZE" \
  --batch-size "$BATCH" \
  --lr 1.237e-3 \
  --gamma 0.995 \
  --start-lambda 0.79 \
  --end-lambda 0.75 \
  --lambda-schedule-steps "$LAMBDA_STEPS" \
  --random-fen-skipping 3 \
  --validation-size 1000000 \
  --check-val-every-n-epoch 5 \
  --network-save-period 20 \
  --save-last-network True \
  --save-top-k 5 \
  --num-workers "$WORKERS" \
  --accelerator cuda \
  --default-root-dir "$ROOT" \
  --seed 42 \
  $RESUME \
  "${@:4}"

# ------------------------------------------------------------------------------------
# PERCHE' DIVERSI DALLA FASE 1
#
# --resume-from-model  semina dalla rete di fase 1 con uno SCHEDULE FRESCO (non riprende
#                      lo stato dell'ottimizzatore). E' il "retrain an already good network".
# --batch-size 131072  🔴 CORRETTO IL 29/07, era 16384 (la ricetta SF originale, che non
#                      aveva mai visto le ottimizzazioni della fase 1). Con 4 GPU l'all-reduce
#                      del DDP e' un costo FISSO per step (~78 ms su PCIe): a batch 16384 il
#                      calcolo e' ~15 ms e lo step diventa ~85 ms => **193k pos/s invece di
#                      671k**, cioe' 115 ore invece di 33 su 800 epoche. E' esattamente il bug
#                      gia' pagato e riparato in fase 1. Vedi HANDOFF_PERFORMANCE.md.
# --lr 1.237e-3        META' della ricetta SF (4.375e-4) SCALATA di sqrt(131072/16384)=2.828,
#                      come in fase 1. Meta' = si rifinisce, non si riparte.
#                      🔴 Se cambi il batch, RICALCOLA la lr: lr = 4.375e-4 * sqrt(batch/16384).
# --gamma 0.995        piu' lento, coerente col run piu' lungo. NON si tocca cambiando il
#                      batch: gamma e' per-EPOCA e le epoche sono definite in POSIZIONI
#                      (--epoch-size), non in batch.
#                      Verifica sulle epoche reali: 0.995^800 = 0.018 => lr finale all'1,8%.
# --max-epochs 800     piu' lungo proprio perche' l'lr e' piu' basso.
# lambda 0.79 -> 0.75  🔴 CAMBIATO IL 29/07 (decisione utente), era anneal 1.0 -> 0.75.
#   in 100 epoche      MOTIVO: `--resume-from-model` riparte con schedule fresco, quindi un
#   poi FERMO          anneal che ricomincia da 1.0 tirerebbe la rete INDIETRO verso il target
#                      -eval, disfacendo lo spostamento verso il risultato di partita che la
#                      fase 1 ha impiegato 416 epoche a costruire. 0.79 e' il punto dove la
#                      fase 1 si e' fermata (1.0 - 0.25*416/500): si RIPRENDE la traiettoria
#                      invece di ricominciarla, e senza il salto secco che avrebbe uno 0.75
#                      fisso dal primo step.
#                      PERCHE' 100 E NON TUTTA LA RUN: 100 e' all'incirca il numero di epoche
#                      che alla fase 1 mancavano per chiudere il suo anneal (84), quindi si
#                      finisce quel lavoro allo stesso ritmo. E cosi' 700 epoche su 800 girano
#                      al lambda finale: annealando fino in fondo, il modello che spediamo
#                      sarebbe allenato quasi sempre a un lambda che tocca solo all'ultima
#                      epoca — che e' esattamente il difetto della fase 1.
#                      ⚠️ DEVIAZIONE consapevole dalla ricetta SF (che annealla in entrambe le
#                      fasi): SF fa due run indipendenti, noi ne concateniamo due.
#                      ⚠️ La val_loss NON e' confrontabile fra epoche finche' l'anneal corre
#                      (il bersaglio si sposta): dopo l'epoca 200 lo diventa.
#
# ⚠️ A/B DA FARE QUI, non altrove: due fase-2 identiche dalla STESSA rete di fase 1, stesso
# schedule e stesse posizioni viste, una CON T91-2026 nell'interleave e una SENZA. E' l'unico
# modo di trasformare la scommessa su T91 in un numero. Gate net-isolated a TC >= 20+0.2;
# pavimento di risoluzione +-Elo ~ 344/sqrt(N) => 5 Elo richiedono ~4.700 partite.
#
# CHIUSURA: over_last dei 3 checkpoint finali.
# ------------------------------------------------------------------------------------
