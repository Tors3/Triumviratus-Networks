#!/usr/bin/env bash
# FASE 3 — rete 7.0 "legio-septima": CODA DI ANNEAL a lr bassissimo.
#
# COSA RISPONDE. A fine fase 2 la diagnosi era: overfitting escluso (train ~ val, a volte val
# SOTTO train), val_loss ancora in discesa (medie a 50 epoche: 0.0036754 -> 0.0036690 ->
# 0.0036187 -> 0.0035618, l'ultimo calo a 3.4 sigma), ma lr scesa a 2.2e-5. Restano tre
# ipotesi per il rallentamento: (a) anneal troncato, (b) bacino, (c) capacita' di L1 1024.
# Questa fase testa SOLO la (a), che e' la piu' economica: si prolunga la coda dell'anneal
# invece di troncarla. Se il match appaiato contro fase2-ep800 e' positivo, era (a) e la 8.0
# non e' una questione di architettura. Se e' zero, (a) e' chiusa e restano (b) e (c).
#
# ⚠️ NON aggiunge dati "migliori": a lr ~4e-5 si assesta, non si riparte. Cambiare corpus in
# questo regime e' uno SPOSTAMENTO DI DISTRIBUZIONE che la rete non ha piu' il lr per
# assorbire. Si tiene la stessa miscela della fase 2, piu' i 4 BT4 mai visti.
#
# 🔑 PERCHE' NON RIVEDE "TUTTO DA CAPO" pur ripartendo dalla testa dei file. random-fen-skipping
# 3 SCARTA IL 75% di quello che legge: in fase 2 il cursore e' avanzato di 15.2 G posizioni per
# file ma solo 3.81 G sono state CONSEGNATE, le altre 11.4 G lette e buttate senza un solo passo
# di gradiente. Rileggere la stessa regione con un SEED DIVERSO pesca un 25% diverso, quindi la
# sovrapposizione attesa con l'insieme della fase 2 e' ~25%: il ~75% di cio' che la fase 3
# consegna e' materiale mai usato, senza toccare offset ne' tagliare file.
# (Il 53% di corpus mai raggiunto sta nella CODA dei file e resta fuori portata: il loader ha
# skipChunks() ma lo usa solo per sfalsare i rank DDP, non c'e' un offset esposto.)
#
# USO:  ./train_phase3.sh <MODELLO_FASE2.pt-o-.nnue> [DATA_DIR] [NUM_WORKERS]
set -euo pipefail

MODEL="${1:?serve il modello finale della fase 2}"
DATA="${2:-/workspace/phase2}"
WORKERS="${3:-24}"
FEATURES="Full_Threats+HalfKAv2_hm^+PP_3Wide+PassedPawns"

TRAINER="${TRAINER:-}"
if [ -z "$TRAINER" ]; then
  for c in "$(dirname "$0")/nnue-pytorch" "$HOME/legio-septima/nnue-pytorch" \
           /root/legio-septima/nnue-pytorch; do
    [ -f "$c/train.py" ] && { TRAINER="$c"; break; }
  done
fi
[ -n "$TRAINER" ] && [ -f "$TRAINER/train.py" ] || {
  echo "ERRORE: trainer non trovato. TRAINER=/percorso/nnue-pytorch $0 ..." >&2; exit 1; }
echo "[trainer] $TRAINER"

ROOT="${ROOT:-$DATA/../run_phase3}"
mkdir -p "$ROOT"
RESUME=""
LAST=$(find "$ROOT" -name last.ckpt -type f -exec ls -1t {} + 2>/dev/null | head -1)
if [ -n "$LAST" ]; then
  RESUME="--resume-from-checkpoint $LAST"
  echo "[RIPRESA] $LAST  ($(stat -c '%y' "$LAST" 2>/dev/null))"
else
  echo "[NUOVO] nessun checkpoint in $ROOT"
fi

cd "$TRAINER"

# =====================================================================================
# CORPUS — PESATO PER POSIZIONE, NON PER FILE
#
# 🔴 IL DIFETTO CHE QUESTA SEZIONE CORREGGE. Il loader sceglie il file UNIFORMEMENTE
# (parallel_dataloader.h:145, `fileId = local_dist(prng)`), NON in proporzione alla
# dimensione. In fase 2, con 21 file e 80 G posizioni consegnate, ogni file ne ha ricevute
# 3.81 G — cioe' le PASSATE su ciascuno sono state inversamente proporzionali alla sua
# dimensione. Misurato (passate = consegnate / posizioni del file, a 2.5 byte/posizione):
#
#     0.14x  test80-2022-10-oct        69 GB   <- visto il 14%
#     0.43x  T60T70Farseer x5          22 GB
#     0.44x  test78-jan-may            21 GB
#     0.49x  leela96 x5                19 GB
#     0.52x  test77-dec                18 GB
#     0.58x  T91-May / 0.62x T91-June  17/15 GB
#     0.69x  ... 0.90x  i test80 mensili e test78-jun-sep   10-14 GB
#
# I pacchetti da 19-22 GB hanno visto META' dei mensili da 11 GB. Non e' una scelta: e' un
# artefatto del conteggio per file. Qui si riequilibra dando a ogni file un numero di SLOT
# proporzionale alla sua dimensione (il loader tratta ogni voce della lista come un file a
# se', quindi ripetere un percorso ne aumenta il peso). Unita' ~5 GB.
#
# 🔴 UNICA ECCEZIONE VOLUTA: test80-2022-10-oct resta a 1 SOLO SLOT nonostante i suoi 69 GB.
# E' l'unico file senza `.min` nel nome, cioe' la versione NON deduplicata del dataset:
# pesarlo a dimensione rischierebbe di alimentare ripetizioni invece di varieta'. Decisione
# dell'utente il 30/07, deliberatamente sottopesato e non misurato.
#
# ⚠️ ONESTA': questo E' uno spostamento di distribuzione rispetto alla fase 2 — ma verso la
# composizione REALE del corpus, non verso una miscela arbitraria. In fase 2 una posizione di
# test80-aug era ~6x piu' probabile di una di oct a parita' di posizione. Ora sono uniformi.
# =====================================================================================

# slot = ARROTONDAMENTO AL PIU' VICINO di GB/4, minimo 1;  oct forzato a 1.
# 🔴 NON ceil, e non unita' 5 GB: con 350 epoche entrambe le scelte fanno CICLARE i file
# piccoli. ceil(6/5)=2 slot per un file da 6,5 GB e' un sovrappeso del 54% rispetto alla sua
# quota (1,3), e a 350 epoche il suo cursore farebbe 1,3 passate. Arrotondando al piu' vicino
# con unita' 4 GB l'errore di arrotondamento scende e il peggio resta ~1,1 (vedi sotto).
UNIT_GB=4
# I file sotto UNIT_GB escono: con meno di 1 unita' prenderebbero comunque 1 slot, cioe' un
# sovrappeso illimitato. Esclusi test60-nov (3,4 GB) e test60-dec (3,2 GB) - 6,6 GB di BT4 mai
# visto che si perde, il prezzo per un run da 350 epoche senza riciclo.
MIN_GB=$UNIT_GB
OCT_PAT="test80-2022-10-oct"
T91_MESI=( May June )

shopt -s nullglob
CAND=( "$DATA"/*.relabel-BT4-tf13tune.binpack )
for m in "${T91_MESI[@]}"; do CAND+=( "$DATA"/T91-2026-"$m"-*.binpack ); done

DATASETS=()
echo "corpus fase 3 (slot = peso nel campionamento):"
for f in "${CAND[@]}"; do
  case "$f" in *.q.binpack) continue;; esac
  gb=$(( $(stat -c%s "$f") / 1000000000 ))
  if [ "$gb" -lt "$MIN_GB" ]; then
    printf '   -- escluso  %3d GB  %s  (< %d GB: ciclerebbe)\n' "$gb" "$(basename "$f")" "$MIN_GB"
    continue
  fi
  if [[ "$(basename "$f")" == *"$OCT_PAT"* ]]; then
    slots=1; nota="  <- non-.min: sottopesato di proposito"
  else
    # arrotondamento al piu' vicino: (gb + UNIT/2) / UNIT
    slots=$(( (gb + UNIT_GB / 2) / UNIT_GB )); [ "$slots" -lt 1 ] && slots=1; nota=""
  fi
  for _ in $(seq "$slots"); do DATASETS+=( "$f" ); done
  printf '  %2d slot  %3d GB  %s%s\n' "$slots" "$gb" "$(basename "$f")" "$nota"
done
[ "${#DATASETS[@]}" -gt 0 ] || { echo "nessun binpack in $DATA"; exit 1; }
echo "slot totali: ${#DATASETS[@]}"

# Il filtro qui e' MIN_GB = UNIT_GB = 4, non il MIN_BINPACK_GB=9 della fase 2. Quello serviva a
# impedire che un file da 3 GB fosse riletto 12 volte su 800 epoche a peso uniforme; con la
# pesatura per posizione il criterio diventa "almeno un'unita' di slot", altrimenti il file e'
# sovrappesato senza limite. Restano dentro test79-apr (8,5 GB) e test79-may (6,5 GB), 15 GB di
# BT4 MAI VISTO in fase 2.

EPOCHS=350
EPOCH_SIZE=100000000      # 🔴 NON alzarlo: le "epoche" non sarebbero piu' confrontabili con
BATCH=131072              #    le misure della fase 2, e gamma e' per-EPOCA.

# 🔴 QUANTE VOLTE OGNI FILE VIENE ATTRAVERSATO. Il loader ha cyclic=true e un file esaurito
# RIPARTE DA CAPO (parallel_dataloader.h:257 seek_to_start). Con S slot totali, E epoche
# consegnano E*100M/S posizioni per slot, e il CURSORE avanza 4x tanto (skip 3 legge 4 posizioni
# per ognuna consegnata). Con la pesatura per dimensione gli attraversamenti sono ~uguali per
# tutti: 4 * (E*100M/S) / posizioni-per-unita'. A 350 epoche e ~96 slot vengono ~0.8-1.0, con due
# eccezioni dovute all'arrotondamento (stima a 2.5 byte/posizione):
#     test79-may   1.12   <- il piu' esposto
#     test78-jun-sep 1.03
#     tutti gli altri 0.80 - 1.00 ;  test80-oct 0.05
# ⚠️ ONESTA': quei due SFORANO l'1.0, quindi rileggono un pezzo della loro testa. Il danno e'
# minimo e quantificabile: dell'11% ri-attraversato solo il 25% viene consegnato (skip 3), quindi
# ~3% delle posizioni consegnate da quel file sono ripetute. Non e' "zero ripetizioni", e' "~3%
# sul file piu' piccolo". La densita' 2.5 byte/posizione e' una STIMA, non una misura: se fosse
# piu' alta del 10% questi numeri salgono in proporzione.

PY=""
for c in /usr/bin/python3 "${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}" \
         /venv/main/bin/python python3 python; do
  [ -n "$c" ] || continue
  command -v "$c" >/dev/null 2>&1 && "$c" -c "import torch" >/dev/null 2>&1 && { PY="$c"; break; }
done
[ -n "$PY" ] || { echo "nessun interprete python con torch (vast.ai: /venv/main NON ha torch)"; exit 1; }
echo "interprete: $PY ($($PY --version 2>&1))"

GPUS="${GPUS:-$(nvidia-smi -L 2>/dev/null | grep -c . || echo 1)}"
NGPU="$GPUS"
if [ "$NGPU" -gt 1 ]; then
  echo "[DDP] $NGPU GPU via torchrun"
  LAUNCH=("$PY" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" ddp_launcher.py train.py)
  GPUARG=()
else
  echo "[single-GPU]"
  LAUNCH=("$PY" train.py); GPUARG=(--gpus 0)
fi

export NNUE_WORKER_THREAD_RATIO="${NNUE_WORKER_THREAD_RATIO:-0.05}"

"${LAUNCH[@]}" "${DATASETS[@]}" \
  "${GPUARG[@]}" \
  --threads "${TORCH_THREADS:-1}" \
  --features "$FEATURES" \
  --resume-from-model "$MODEL" \
  --max-epochs "$EPOCHS" \
  --epoch-size "$EPOCH_SIZE" \
  --batch-size "$BATCH" \
  --lr 3.36e-5 \
  --gamma 0.99197 \
  --start-lambda 0.75 \
  --end-lambda 0.75 \
  --random-fen-skipping 3 \
  --validation-size 1000000 \
  --check-val-every-n-epoch 5 \
  --network-save-period 10 \
  --save-last-network True \
  --save-top-k 5 \
  --num-workers "$WORKERS" \
  --accelerator cuda \
  --default-root-dir "$ROOT" \
  --seed 43 \
  $RESUME \
  "${@:4}"

# ------------------------------------------------------------------------------------
# PERCHE' QUESTI PARAMETRI
#
# --lr 3.36e-5         1.5x il punto d'arrivo della fase 2 (1.237e-3 * 0.995^800 = 2.24e-5), ma
#                      ancora 37x sotto la sua PARTENZA. Non e' il valore identico, e la scelta
#                      e' deliberata per due ragioni.
#                      (1) `--resume-from-model` AZZERA lo stato dell'ottimizzatore: Adam
#                      riparte senza momenti, quindi "stesso lr nominale" NON e' una
#                      continuazione pura — i primi passi sono comunque piu' grandi. Il valore
#                      identico darebbe l'illusione della continuita', non la continuita'.
#                      (2) L'ipotesi sotto test e' che il lr fosse troppo basso per assorbire
#                      dati nuovi. Tenendolo al minimo si garantisce di confermare la
#                      staticita': la rete vedrebbe 20 G posizioni fresche senza il passo per
#                      usarle.
#                      ⚠️ NON alzare oltre il 2x: sopra ~1e-4 la rete si sposta davvero dal
#                      minimo e va trattato come warm restart, con criteri diversi. 1.5x
#                      (decisione utente, 30/07) e' il compromesso: mobilita' sufficiente per
#                      usare i dati freschi, disturbo trascurabile sul minimo raggiunto.
#                      ⚠️ ATTESO: la val SALE nelle prime ~10 epoche (Adam si ri-accende). Se
#                      entro l'epoca 40-50 non e' tornata al livello di fine fase 2, il bump era
#                      eccessivo: ripartire da 2.24e-5. Se scende sotto, e' il segnale cercato.
# --gamma 0.99197      0.99197^350 = 0.0595 => si arriva a 2.0e-6, un fattore 17 sotto la
#                      partenza e 11 sotto dove la fase 2 ha chiuso. Progressione:
#                        ep  50 -> 2.24e-5   (riattraversa il punto di fine fase 2)
#                        ep 100 -> 1.50e-5
#                        ep 200 -> 6.7e-6
#                        ep 300 -> 3.0e-6
#                        ep 350 -> 2.0e-6
#                      Solo ~50 epoche stanno SOPRA il punto d'arrivo della fase 2, 300 sotto:
#                      e' una coda di anneal vera, non un secondo run travestito. E il ri-anneal
#                      e' COMPLETO dentro la fase, quindi il confronto finale e' fra due punti
#                      entrambi a lr basso — non il test confuso che si otterrebbe confrontando
#                      due punti a lr diverso, dove il rumore del lr alto maschera la risposta.
#                      ⚠️ Non scendere sotto ~2e-6: il passo diventa troppo piccolo per cambiare
#                      i pesi e le epoche si spendono per niente.
# --resume-from-model  SOLO i pesi, schedule fresco. Con --resume-from-checkpoint riprenderebbe
#                      lo scheduler della fase 2 e il lr ripartirebbe dalla SUA tabella,
#                      ignorando --lr: la fase 3 non esisterebbe.
# --start/end-lambda   0.75 FISSO su entrambi. L'anneal e' finito da 700 epoche; farlo ripartire
#   entrambi 0.75      tirerebbe la rete indietro verso il target-eval.
# --seed 43            diverso dalla fase 2 (42): il campionamento del corpus non deve
#                      ripercorrere la stessa sequenza di file.
# --network-save-period 10   piu' fitto di 20: su 200 epoche servono piu' punti, e 🔴 in fase 2
#                      la rotazione ha CANCELLATO i checkpoint della zona di minimo (ep500-560)
#                      prima che li potessimo serializzare.
#
# 🔴 IL VERDETTO NON E' LA val_loss. Nella coda della fase 2 oscilla di ~1e-4 con la miscela di
# file campionata (sd/media: 0.49% a ep400-499, 1.08% a ep700-799 NONOSTANTE lr 4x piu' bassa
# => il rumore non e' del lr, e' dei dati), mentre il segnale atteso da 100 epoche a questo lr
# e' ~2e-5. Cinque volte troppo rumore. Il verdetto e' il match APPAIATO fase3-ep100 contro
# fase2-ep800, stesso binario, 3000 partite, sulla baseline congelata _baseline_7.0_pre_nps.
# ------------------------------------------------------------------------------------
