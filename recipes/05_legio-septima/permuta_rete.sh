#!/usr/bin/env bash
# Permutazione del feature transformer — ULTIMO passo del capitolo rete.
#
# COSA FA. Riordina i neuroni di L1 in modo che le attivazioni nulle si raggruppino in blocchi
# contigui. Il layer `AffineTransformSparseInput` salta i blocchi interamente nulli, quindi piu'
# sono raggruppati e meno lavoro fa. La valutazione resta **bit-identica**: e' solo un cambio di
# ordine, applicato coerentemente ai pesi di TUTTI i blocchi di input, alla bias del FT e alle
# colonne di fc_0.
#
# 🔴 SUL GUADAGNO: **+1,62% NPS, MISURATO BENE** (31/07, VM scarica, ep799 permutata contro
# se stessa non permutata, stesso binario). B piu' veloce in **55 posizioni su 60 (91,7%)** —
# e' il test del segno, indipendente dalla scala, e con quel rapporto il caso e' escluso.
# Quartili +1,01% .. +2,26%.
# Storia della misura, perche' e' istruttiva: il README della 6.0 dichiarava "+2%"; il 25/07
# fu "smentito" con -0,34% e io ho creduto alla smentita perche' era piu' recente. Ma quella
# era presa con `nps_ab_binaries.py`, che esegue TUTTE le posizioni di A e poi tutte quelle di
# B — lo stesso strumento del falso +5,3% sulla patch 2A, dove il segno seguiva l'ordine e non
# la patch. Il +2% originale era sostanzialmente giusto.
# ⇒ Usare SEMPRE `nps_ab_interleaved.py` (ha ora `--net-b` per confrontare due reti sullo
#    stesso binario), su macchina scarica, e leggere prima la FRAZIONE DI POSIZIONI VINTE.
#
# 🔴 SU QUALE RETE. Quella che si spedisce, con la sua stringa di feature:
#   - TRANN2 (f2-ep799, la 7.0):  Full_Threats+HalfKAv2_hm^+PP_3Wide+PassedPawns
#   - TRANN3 (se un graft passasse): la stessa + FileState+Bishops+PasserKings+Hanging
# Sbagliare la stringa NON da' errore: produce una rete che il motore rifiuta sull'hash, oppure
# — peggio — una permutazione applicata a un layout diverso da quello reale.
#
# 🔴 DOVE. Serve CUDA: si fa sulla VM, non sul portatile. Se cupy da' problemi (la major deve
# seguire il driver: cupy-cuda12x su driver CUDA 13 esplode con libnvrtc.so.12 mancante) si passa
# `--cupy False`: piu' lento, stesso risultato.
#
# USO:  ./permuta_rete.sh <rete_in.nnue> <rete_out.nnue> [binpack] [n_posizioni]
set -euo pipefail

IN="${1:?serve la rete di ingresso .nnue}"
OUT="${2:?serve la rete di uscita .nnue}"
DATA="${3:-/workspace/phase2/test80-2022-08-aug-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack}"
COUNT="${4:-1000000}"
FEATURES="${FEATURES:-Full_Threats+HalfKAv2_hm^+PP_3Wide+PassedPawns}"

TRAINER="${TRAINER:-/root/legio-septima/nnue-pytorch}"
cd "$TRAINER"

PY=/usr/bin/python3
echo "rete in : $IN"
echo "feature : $FEATURES"
echo "dati    : $DATA  ($COUNT posizioni)"
echo

# Un solo comando fa tutto: raccolta attivazioni -> ricerca permutazione -> applicazione.
# La versione in tre passi (gather / find_perm / eval_perm) serve solo per ISPEZIONARE quanto
# la permutazione migliora la sparsita' prima di applicarla; `eval_perm` stampa la frazione di
# blocchi nulli con e senza permutazione, che e' la grandezza che il layer sparso sfrutta.
nice -n 19 $PY serialize.py "$IN" "$OUT" \
  --features "$FEATURES" \
  --ft-compression leb128 \
  --ft-optimize \
  --ft-optimize-data "$DATA" \
  --ft-optimize-count "$COUNT"

echo
echo "=============================================================="
echo " VERIFICA OBBLIGATORIA — la permutazione DEVE essere trasparente"
echo "=============================================================="
echo "Mettere le due reti accanto allo STESSO binario, una per volta, e confrontare il bench:"
echo "  la permutazione cambia l'ordine dei neuroni, non la valutazione, quindi l'albero"
echo "  dev'essere IDENTICO. Un bench diverso = permutazione applicata male, si butta."
echo
echo "  cp $IN  <dir>/nn-legio-septima.nnue && (cd <dir> && printf 'bench\\nquit\\n' | ./triumviratus | grep -i 'nodes searched')"
echo "  cp $OUT <dir>/nn-legio-septima.nnue && (cd <dir> && printf 'bench\\nquit\\n' | ./triumviratus | grep -i 'nodes searched')"
echo
echo "⚠️ Il bench va preso su macchina SCARICA e un solo bench per processo (non e' idempotente"
echo "   nello stesso processo: killer/node_cache/prev_pv non si azzerano con ucinewgame)."
