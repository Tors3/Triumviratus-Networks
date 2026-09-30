#!/usr/bin/env bash
# Media dei pesi delle ultime 5 epoche della F4 + permutazione del feature transformer (29/09/2026, decisione
# dell'utente). La F4 salva un checkpoint solo ogni 10 epoche: questo script ne fotografa last.ckpt alla fine di
# ciascuna delle epoche 55..59, poi (a F4 finita, GPU libere):
#   1. media aritmetica dei tensori del MODELLO (chiavi "model.*") dei 5 checkpoint -> F4_avg5.ckpt;
#   2. F4_avg5.nnue (plain) e F4_avg5_perm.nnue (serialize.py --ft-optimize, come permuta_rete.sh della legio:
#      riordina i neuroni di L1 per la sparsita', eval bit-identica, +1,6% NPS misurato sulla legio).
# Verifica obbligatoria, poi, in locale: bench delle due reti sullo STESSO binario MoE = identico.
# USO: WAIT_PID=<pid di train_moe1024_F4.sh> nohup ~/tt/TrainingMoE1024/avg_perm_F4.sh >> ~/avg_perm_F4.log 2>&1 &
set -uo pipefail
ROOT="${ROOT:-/root/moe/run}"
TRAINER="${TRAINER:-/root/moe/nnue-pytorch}"
PY=/venv/main/bin/python3
FEATURES="Full_Threats+HalfKAv2_hm_P4^+PP_3Wide+PassedPawns"
LOG=/root/train_moe1024.log
FIRST="${AVG_FIRST:-55}"
LAST="${AVG_LAST:-59}"
SNAP="$ROOT/F4_snap"; mkdir -p "$SNAP"

ckpt() { find "$ROOT/F4" -name last.ckpt -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-; }
last_epoch() { tr '\r' '\n' < "$LOG" | grep -a -oE 'Epoch +[0-9]+ \(Train\): \[' | tail -1 | grep -oE '[0-9]+'; }

echo "=== $(date -u)  avg_perm_F4: fotografo last.ckpt alle epoche $FIRST..$LAST"
prev=0
while kill -0 "${WAIT_PID:?pid della F4}" 2>/dev/null || [ "$prev" = 0 ]; do
  L=$(ckpt); [ -n "$L" ] || { sleep 20; continue; }
  m=$(stat -c %Y "$L")
  if [ "$m" != "$prev" ] && [ $(( $(date +%s) - m )) -ge 20 ]; then
    prev=$m; e=$(last_epoch)
    if [ -n "$e" ] && [ "$e" -ge "$FIRST" ] && [ "$e" -le "$LAST" ] && [ ! -f "$SNAP/ep$e.ckpt" ]; then
      cp "$L" "$SNAP/ep$e.ckpt" && echo "$(date -u +%T) foto ep$e"
    fi
  fi
  kill -0 "$WAIT_PID" 2>/dev/null || break
  sleep 15
done
# l'ultima epoca: dopo la fine del processo last.ckpt e' quello dell'epoca finale
L=$(ckpt); e=$(last_epoch)
[ -n "$e" ] && [ ! -f "$SNAP/ep$e.ckpt" ] && [ "$e" -ge "$FIRST" ] && cp "$L" "$SNAP/ep$e.ckpt" && echo "$(date -u +%T) foto ep$e (finale)"
ls -la "$SNAP"

cd "$TRAINER"
CUDA_VISIBLE_DEVICES="" "$PY" - "$SNAP" "$ROOT/F4_avg5.ckpt" <<'PYEOF'
import glob, sys, torch
snap, out = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(snap + "/ep*.ckpt"))
print("media di:", files)
base = torch.load(files[-1], map_location="cpu", weights_only=False)
sd = base["state_dict"]
keys = [k for k, v in sd.items() if k.startswith("model.") and torch.is_tensor(v) and v.is_floating_point()]
acc = {k: sd[k].double().clone() for k in keys}
for f in files[:-1]:
    s = torch.load(f, map_location="cpu", weights_only=False)["state_dict"]
    for k in keys:
        acc[k] += s[k].double()
for k in keys:
    sd[k] = (acc[k] / len(files)).to(sd[k].dtype)
base["state_dict"] = sd
torch.save(base, out)
print("scritto", out, "tensori mediati:", len(keys))
PYEOF

BIN=$(ls /data/bt4/*test80*.binpack | head -1)
CUDA_VISIBLE_DEVICES="" "$PY" serialize.py "$ROOT/F4_avg5.ckpt" "$ROOT/nets/F4_avg5.nnue" --features "$FEATURES" --l1 1024 \
  --ft-compression leb128 && echo "plain ok"
"$PY" serialize.py "$ROOT/F4_avg5.ckpt" "$ROOT/nets/F4_avg5_perm.nnue" --features "$FEATURES" --l1 1024 \
  --ft-compression leb128 --ft-optimize --ft-optimize-data "$BIN" --ft-optimize-count 1000000 && echo "perm ok"
ls -la "$ROOT"/nets/F4_avg5*
echo "=== $(date -u)  avg_perm_F4 FINITO"
