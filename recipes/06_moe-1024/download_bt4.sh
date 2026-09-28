#!/usr/bin/env bash
# Dati della rete MoE-1024 (28/09/2026): UN SOLO MIX con TUTTO il corpus rietichettato BT4, come la ricetta SFNNv16
# (vedi README.md). Niente fasi separate, niente self-play nostro (decisione dell'utente, 28/09).
#
# Sorgenti (misurate il 28/09 via API HF, varianti .q. escluse), ~701 GB unici in tutto (misurati sulla VM il 28/09):
#   vondele/master-binpacks_relabel ~129 GB (self-play SF + DFRC), vondele/linrock_relabel_1 ~204 GB senza il mese duplicato in .part_N (test80 + test60 +
#   altro), vondele/linrock_relabel_2 ~142 GB, vondele/from_kaggle_2_relabel ~111 GB (T60/T70 wrongIsRight),
#   vondele/from_kaggle_1_relabel ~98 GB (leela96), xushawn/test80-bt4-relabel 17,7 GB. La legio-septima ne usava ~553.
#   Disco della VM: >= 1 TB liberi per i dati.
#   - T91 (jshriver/t91-binpacks, ~160 GB compressi / ~320 estratti) ESCLUSO di default: non e' rietichettato BT4
#     (etichette della rete T91 del momento) e la sua forza rispetto al T80 NON e' mai stata misurata (l'A/B
#     "con/senza T91" previsto in train_phase2.sh della legio non e' stato fatto). Si include solo con --with-t91,
#     dopo la verifica di README.md ("T91: prima di usarlo").
#
# USO:  ./download_bt4.sh [DEST] [BUDGET_GB] [PARALLELI] [--with-t91]
#       ./download_bt4.sh /data/bt4 850 3
set -euo pipefail
DEST="${1:-/data/bt4}"; BUDGET_GB="${2:-850}"; JOBS="${3:-3}"; WITH_T91=0
for a in "$@"; do [ "$a" = "--with-t91" ] && WITH_T91=1; done
mkdir -p "$DEST"; LOG="$DEST/download.log"
die() { echo "ERRORE: $*" >&2; exit 1; }
HF_CLI=""; for c in hf huggingface-cli; do command -v $c >/dev/null && { HF_CLI=$c; break; }; done
[ -n "$HF_CLI" ] || die "manca 'hf': pip install -U 'huggingface_hub[hf_transfer]'"

PLAN=$(python - "$BUDGET_GB" "$WITH_T91" <<'PY'
import json, re, sys, urllib.request
budget, with_t91 = float(sys.argv[1]) * 1e9, sys.argv[2] == "1"
REPOS = ["vondele/master-binpacks_relabel", "vondele/linrock_relabel_1", "vondele/linrock_relabel_2",
         "vondele/from_kaggle_2_relabel", "vondele/from_kaggle_1_relabel", "xushawn/test80-bt4-relabel"]
files = []
for repo in REPOS:
    tree = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true", timeout=60))
    for e in tree:
        p = e["path"]
        if e.get("type") == "file" and p.endswith(".binpack") and ".q." not in p:
            files.append((repo, p, (e.get("lfs") or {}).get("size") or e.get("size") or 0))
if with_t91:
    for e in json.load(urllib.request.urlopen("https://huggingface.co/api/datasets/jshriver/t91-binpacks/tree/main", timeout=60)):
        if e["path"].endswith(".binpack.zst"):
            files.append(("jshriver/t91-binpacks", e["path"], int(((e.get("lfs") or {}).get("size") or e.get("size") or 0) * 2)))
# dedup: stesso mese sia intero sia in .part_N (lezione della legio: 69 GB scaricati due volte)
base = lambda p: re.sub(r"\.part_\d+(?=\.binpack)", "", p)
whole = {base(p) for _, p, _ in files if ".part_" not in p}
files = [f for f in files if ".part_" not in f[1] or base(f[1]) not in whole]
prio = lambda r, p: 0 if "test80" in p else 1 if "T60T70" in p else 2 if "test60" in p else 3 if "leela96" in p else 5 if r.startswith("jshriver") else 4
files.sort(key=lambda f: (prio(f[0], f[1]), -f[2]))
used = 0
for r, p, s in files:
    if used + s <= budget:
        used += s; print(f"{r}\t{p}\t{s}")
print(f"# TOTALE: {used/1e9:.1f} GB su {budget/1e9:.0f}", file=sys.stderr)
PY
)
echo "$PLAN" | awk -F'\t' 'NF==3{printf "    %-30s %-64s %6.1f GB\n", $1, $2, $3/1e9}'
need=$(echo "$PLAN" | awk -F'\t' 'NF==3{s+=$3} END{printf "%d", s/1073741824}')
avail=$(df -PBG "$DEST" | awk 'NR==2{gsub("G","",$4); print $4}')
echo "[+] pianificati ${need} GB, liberi ${avail} GB"; [ "${avail:-0}" -ge "$need" ] || die "spazio insufficiente"
export HF_HUB_ENABLE_HF_TRANSFER=1; python -c "import hf_transfer" 2>/dev/null || export HF_HUB_ENABLE_HF_TRANSFER=0
dl_one() {
  local repo="$1" f="$2" dest="$3" log="$4"
  echo "[$(date +%H:%M:%S)] START  $f" | tee -a "$log"
  if "$HF_CLI" download "$repo" --repo-type dataset --include "$f" --local-dir "$dest" >>"$log" 2>&1; then
    [[ "$f" == *.zst ]] && { zstd -d --rm -q "$dest/$f" >>"$log" 2>&1 || { echo "zstd FALLITO $f" | tee -a "$log"; return 1; }; }
    echo "[$(date +%H:%M:%S)] OK     $f" | tee -a "$log"
  else echo "[$(date +%H:%M:%S)] FALLITO $f (rilancia: riprende)" | tee -a "$log"; return 1; fi
}
export -f dl_one; export HF_CLI
echo "$PLAN" | awk -F'\t' 'NF==3{print $1"\t"$2}' \
  | xargs -P "$JOBS" -d'\n' -I{} bash -c 'IFS=$'"'"'\t'"'"' read -r r f <<< "{}"; dl_one "$r" "$f" "$1" "$2"' _ "$DEST" "$LOG"
du -sh "$DEST"; echo "=== DATI PRONTI in $DEST (mix unico: tutti i .binpack della cartella)"
