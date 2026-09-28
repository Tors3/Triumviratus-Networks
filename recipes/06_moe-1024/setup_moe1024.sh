#!/usr/bin/env bash
# Prepara la VM per la rete MoE-1024 (28/09/2026): HalfKA a 4 esperti per fascia di materiale, L1 = 1024.
#
#   1) trainer upstream al pin + patch completa (../patches/triumviratus_trainer.patch) via setup_vm.sh della legio;
#   2) motore Triumviratus compilato con -DTRIUMV_PSQ_PHASES=4 (serve a cross_check_eval);
#   3) rete CASUALE con delta per fase diversi da zero e cross_check_eval motore contro trainer.
#      Con i delta a zero le 4 fasi avrebbero gli stessi pesi e un indice di fase sbagliato NON si vedrebbe.
#
# USO:  ./setup_moe1024.sh [DIR_DESTINAZIONE] [BINPACK_PER_IL_CROSS_CHECK]
#       ./setup_moe1024.sh $HOME/moe /data/bt4/qualunque.binpack
# Senza binpack il punto 3 si ferma alla rete casuale: il cross-check va lanciato appena c'e' un file di dati.
#
# Gate: R^2 >= 0.9999 ed errore medio <= 3 unita' interne (misurato in locale il 28/09: R^2 0.999999, 1.67).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LEGIO="$HERE/../Training70_LegioSeptima"
DEST="${1:-$HOME/moe}"
DATA="${2:-}"
FEATURES="Full_Threats+HalfKAv2_hm_P4^+PP_3Wide+PassedPawns"
ENGINE_REPO="https://github.com/Tors3/Triumviratus.git"
ENGINE_COMMIT="${ENGINE_COMMIT:-ccddd72}"   # primo commit con il codice MoE (bench di default 273477)

bash "$LEGIO/setup_vm.sh" "$DEST"
TRAINER="$DEST/nnue-pytorch"
PY=${PY:-python3}

echo "=== motore MoE (Linux, g++)"
if [ ! -d "$DEST/Triumviratus/.git" ]; then
  git clone -q "$ENGINE_REPO" "$DEST/Triumviratus"
fi
( cd "$DEST/Triumviratus" && git fetch -q && git checkout -q "$ENGINE_COMMIT" )
ARCH=avx2; grep -q avx512bw /proc/cpuinfo && ARCH=avx512
( cd "$DEST/Triumviratus/source" && make -j"$(nproc)" "$ARCH" EXE=triumv_moe1024 \
    EXTRACXXFLAGS="-DTRIUMV_L1=1024 -DTRIUMV_PSQ_PHASES=4" >/dev/null )
ENGINE="$DEST/Triumviratus/source/triumv_moe1024"
[ -x "$ENGINE" ] || { echo "build del motore fallita"; exit 1; }
echo "  motore: $ENGINE ($ARCH)"

echo "=== rete casuale MoE con delta per fase casuali"
cd "$TRAINER"
$PY "$HERE/../Training80/make_random_net.py" --l1 1024 --features "$FEATURES" --perturb-phases "$DEST/random_moe1024p.nnue"

if [ -n "$DATA" ]; then
  echo "=== cross_check_eval motore contro trainer"
  # Su Windows il loader crasha a volte nella creazione dello stream (race, 28/09); su Linux non osservato. Si riprova.
  for i in 1 2 3; do
    if $PY cross_check_eval.py --engine "$ENGINE" --data "$DATA" --net "$DEST/random_moe1024p.nnue" \
         --count 4096 --features "$FEATURES" --l1 1024 | tee "$DEST/cross_check.log"; then break; fi
  done
  grep -E "Correlation|Avg Absolute Error" "$DEST/cross_check.log" | head -2
else
  echo "(nessun binpack: lanciare poi)"
  echo "  cd $TRAINER && $PY cross_check_eval.py --engine $ENGINE --data <file.binpack> \\"
  echo "     --net $DEST/random_moe1024p.nnue --count 4096 --features '$FEATURES' --l1 1024"
fi

echo "=== tetto della GPU a L1 = 1024 (loader escluso)"
$PY "$LEGIO/gpu_ceiling.py" --l1 1024 || echo "(gpu_ceiling fallito: non blocca, ma misuralo prima di partire)"
echo "=== pronto. Dati: ./download_bt4.sh /data/bt4 850 3"
