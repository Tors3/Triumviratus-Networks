#!/usr/bin/env bash
# Consilium-2 (21/10/2026): starts the P run by itself as soon as the data download and the setup have finished
# (author, 21/10: no speed probe, launch the run as planned).
#   1) waits for download_c2.sh / download_bt4.sh / setup_c2.sh to exit;
#   2) moves any .binpack left in a subfolder to the top of $DATA (train_c2.sh lists only $DATA/*.binpack);
#   3) download_missing.sh until no file is missing or of the wrong size (sizes checked against Hugging Face);
#   4) checks the loader build, then runs train_c2.sh with the caller's environment (P_EPOCHS, BATCH, ...).
# Usage: tmux new -d -s train 'P_EPOCHS=550 bash /root/c2pkg/autostart_c2.sh 2>&1 | tee -a /root/train_c2.log'
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DATA="${DATA:-/data/bt4}"
TRAINER="${TRAINER:-/root/c2/nnue-pytorch}"
: "${P_EPOCHS:?P epochs of 1 G positions}"
echo "[$(date -u +%H:%M:%S)] waiting for download and setup"
# anchored on "bash <script>": the tmux server keeps the command line of the session that started it
while pgrep -f "^bash [^ ]*(download_c2|download_bt4|setup_c2)\.sh" >/dev/null; do sleep 30; done
find "$DATA" -mindepth 2 -name "*.binpack" -not -path "*/.cache/*" -exec mv -f {} "$DATA"/ \;
for i in 1 2 3 4; do
  echo "[$(date -u +%H:%M:%S)] completeness check $i"
  bash "$HERE/download_missing.sh" "$DATA" 12 > /root/autostart_missing_$i.log 2>&1
  head -1 /root/autostart_missing_$i.log
  [ -s /root/missing_plan.tsv ] || break
done
[ -s /root/missing_plan.tsv ] && { echo "files still missing after 4 checks (/root/missing_plan.tsv): not starting"; exit 1; }
ls "$TRAINER"/build/*training_data_loader* >/dev/null 2>&1 || { echo "loader not built: not starting"; exit 1; }
echo "[$(date -u +%H:%M:%S)] data complete: $(ls "$DATA"/*.binpack | wc -l) binpacks, $(du -sh --exclude=.cache "$DATA" | cut -f1)"
exec bash "$HERE/train_c2.sh"
