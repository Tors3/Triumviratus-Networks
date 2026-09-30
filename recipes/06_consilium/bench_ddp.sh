#!/usr/bin/env bash
# Throughput VERO del training MoE-1024 in DDP (trainer, loader, all-reduce): pochi minuti per configurazione.
#
# USO:  ./bench_ddp.sh BATCH_GLOBALE [WORKER_PER_RANK] [SECONDI] [SKIP]
#       ./bench_ddp.sh 262144 24 300 10
# Scrive /root/moe/bench/b<BATCH>_w<W>.log e stampa gli ultimi it/s e le pos/s (it/s x batch: il batch e' GLOBALE).
# Dati: tutti i *.binpack completi di /data/bt4 (i download in corso stanno in .cache, non vengono presi).
set -uo pipefail
BATCH="${1:?batch globale}"; W="${2:-24}"; SECS="${3:-300}"; SKIP="${4:-10}"
TRAINER="${TRAINER:-/root/moe/nnue-pytorch}"
PY="${PY:-/venv/main/bin/python3}"
NGPU="${NGPU:-$(nvidia-smi -L | grep -c .)}"
OUT=/root/moe/bench; mkdir -p "$OUT"
LOG="$OUT/b${BATCH}_w${W}.log"
mapfile -t DATA < <(ls /data/bt4/*.binpack)
# 🔴 28/09: 0.05 (l'ottimo della legio) lascia UN solo thread di feature per rank, e con threat + MoE un thread ne
# costruisce ~350k pos/s: era il tetto di tutto il run (381k contro 1.907k pos/s su una 5090 con 0.25).
export NNUE_WORKER_THREAD_RATIO="${NNUE_WORKER_THREAD_RATIO:-0.25}"
cd "$TRAINER"
echo "batch $BATCH, $W worker/rank, $NGPU GPU, skip $SKIP, ${#DATA[@]} binpack, ratio $NNUE_WORKER_THREAD_RATIO" | tee "$LOG"
# nvidia-smi ogni 10 s nello stesso log, per vedere se tutte le schede lavorano
( for _ in $(seq 1 $((SECS / 10))); do sleep 10
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv,noheader | tr '\n' ' ' | sed 's/^/[smi] /'
    echo; done >> "$LOG.smi" ) &
SMI=$!
timeout "$SECS" "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" ddp_launcher.py train.py \
  "${DATA[@]}" \
  --features "Full_Threats+HalfKAv2_hm_P4^+PP_3Wide+PassedPawns" \
  --l1 1024 --batch-size "$BATCH" --epoch-size $((BATCH * ${ITERS:-400})) --max-epochs 1 \
  --threads 1 --num-workers "$W" --random-fen-skipping "$SKIP" \
  --validation-size 0 --network-save-period 1000 --save-last-network False \
  --accelerator cuda --default-root-dir "$OUT/root_b${BATCH}_w${W}" --seed 1 >> "$LOG" 2>&1
kill $SMI 2>/dev/null
# 🔴 Gli it/s della barra di Lightning sono la MEDIA dall'inizio, compilazione compresa: a 262144 la barra finiva a
# 3,76 it/s con un regime vero di 5,3. Si usa invece (iterazioni, tempo trascorso) delle righe "N/400 [mm:ss<",
# dal primo punto oltre il 20% all'ultimo.
tr '\r' '\n' < "$LOG" | grep -oE '[0-9]+/[0-9]+ +\[[0-9:]+<' | awk -v b="$BATCH" '
  { split($1, a, "/"); n = a[1]; tot = a[2]; gsub(/[\[<]/, "", $2); k = split($2, t, ":");
    s = (k == 3) ? t[1]*3600 + t[2]*60 + t[3] : t[1]*60 + t[2];
    if (n >= 0.2*tot && n0 == "") { n0 = n; s0 = s } ; n1 = n; s1 = s }
  END { if (s1 > s0) printf "regime: %.2f it/s  ->  %.0fk pos/s  (da %d a %d it)\n", (n1-n0)/(s1-s0), (n1-n0)/(s1-s0)*b/1000, n0, n1;
        else print "regime: non misurabile (troppo pochi punti)" }'
tail -3 "$LOG.smi"
