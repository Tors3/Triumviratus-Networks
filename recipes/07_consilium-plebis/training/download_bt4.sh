#!/usr/bin/env bash
# Data of the MoE-1024 network (28/09/2026): ONE MIX with ALL of the BT4-relabelled corpus, as in the SFNNv16 recipe
# (see README.md). No separate phases, no self-play of ours (user's decision, 28/09). Used again for Consilium Plebis.
# (26/10/2026: comments and messages translated to English; the code is the one run on the VM.)
#
# Sources (measured on 28/09 through the HF API, .q. variants excluded), ~701 GB unique in all (measured on the VM on 28/09):
#   vondele/master-binpacks_relabel ~129 GB (SF self-play + DFRC), vondele/linrock_relabel_1 ~204 GB without the month
#   duplicated as .part_N (test80 + test60 + other), vondele/linrock_relabel_2 ~142 GB, vondele/from_kaggle_2_relabel
#   ~111 GB (T60/T70 wrongIsRight), vondele/from_kaggle_1_relabel ~98 GB (leela96), xushawn/test80-bt4-relabel 17.7 GB.
#   Legio Septima used ~553 of them.
#   VM disk: >= 1 TB free for the data.
#   - T91 (jshriver/t91-binpacks, ~160 GB compressed / ~320 extracted) EXCLUDED by default: it is not relabelled with
#     BT4 (labels of the T91 network of the moment) and its strength against T80 was NEVER measured (the A/B
#     "with/without T91" planned in legio's train_phase2.sh was not done). It is included only with --with-t91,
#     after the check of README.md ("T91: before using it").
#
# USAGE: ./download_bt4.sh [DEST] [BUDGET_GB] [PARALLEL] [--with-t91]
#        ./download_bt4.sh /data/bt4 850 3
set -euo pipefail
DEST="${1:-/data/bt4}"; BUDGET_GB="${2:-850}"; JOBS="${3:-3}"; WITH_T91=0
for a in "$@"; do [ "$a" = "--with-t91" ] && WITH_T91=1; done
mkdir -p "$DEST"; LOG="$DEST/download.log"
die() { echo "ERROR: $*" >&2; exit 1; }
HF_CLI=""; for c in hf huggingface-cli; do command -v $c >/dev/null && { HF_CLI=$c; break; }; done
[ -n "$HF_CLI" ] || die "'hf' missing: pip install -U 'huggingface_hub[hf_transfer]'"

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
# dedup: the same month both whole and as .part_N (legio's lesson: 69 GB downloaded twice)
base = lambda p: re.sub(r"\.part_\d+(?=\.binpack)", "", p)
whole = {base(p) for _, p, _ in files if ".part_" not in p}
files = [f for f in files if ".part_" not in f[1] or base(f[1]) not in whole]
prio = lambda r, p: 0 if "test80" in p else 1 if "T60T70" in p else 2 if "test60" in p else 3 if "leela96" in p else 5 if r.startswith("jshriver") else 4
files.sort(key=lambda f: (prio(f[0], f[1]), -f[2]))
used = 0
for r, p, s in files:
    if used + s <= budget:
        used += s; print(f"{r}\t{p}\t{s}")
print(f"# TOTAL: {used/1e9:.1f} GB of {budget/1e9:.0f}", file=sys.stderr)
PY
)
echo "$PLAN" | awk -F'\t' 'NF==3{printf "    %-30s %-64s %6.1f GB\n", $1, $2, $3/1e9}'
need=$(echo "$PLAN" | awk -F'\t' 'NF==3{s+=$3} END{printf "%d", s/1073741824}')
avail=$(df -PBG "$DEST" | awk 'NR==2{gsub("G","",$4); print $4}')
# Consilium-2 (21/10/2026): SKIP_SPACE_CHECK=1 when resuming (files already present are not downloaded again).
echo "[+] planned ${need} GB, free ${avail} GB"
[ "${SKIP_SPACE_CHECK:-0}" = 1 ] || [ "${avail:-0}" -ge "$need" ] || die "not enough space"
export HF_HUB_ENABLE_HF_TRANSFER=1; python -c "import hf_transfer" 2>/dev/null || export HF_HUB_ENABLE_HF_TRANSFER=0
dl_one() {
  local repo="$1" f="$2" dest="$3" log="$4"
  echo "[$(date +%H:%M:%S)] START  $f" | tee -a "$log"
  if "$HF_CLI" download "$repo" --repo-type dataset --include "$f" --local-dir "$dest" >>"$log" 2>&1; then
    [[ "$f" == *.zst ]] && { zstd -d --rm -q "$dest/$f" >>"$log" 2>&1 || { echo "zstd FAILED $f" | tee -a "$log"; return 1; }; }
    echo "[$(date +%H:%M:%S)] OK     $f" | tee -a "$log"
  else echo "[$(date +%H:%M:%S)] FAILED $f (run again: it resumes)" | tee -a "$log"; return 1; fi
}
export -f dl_one; export HF_CLI
echo "$PLAN" | awk -F'\t' 'NF==3{print $1"\t"$2}' \
  | xargs -P "$JOBS" -d'\n' -I{} bash -c 'IFS=$'"'"'\t'"'"' read -r r f <<< "{}"; dl_one "$r" "$f" "$1" "$2"' _ "$DEST" "$LOG"
du -sh "$DEST"; echo "=== DATA READY in $DEST (single mix: every .binpack of the folder)"
