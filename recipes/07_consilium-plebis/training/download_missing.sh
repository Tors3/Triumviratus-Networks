#!/usr/bin/env bash
# Consilium-2 (21/10/2026): downloads only the BT4 binpacks of the plan (download_bt4.sh) that are missing in DEST or
# have a size different from Hugging Face's (files copied from another machine have no HF metadata, so a plain
# download_bt4.sh would fetch them again). tb5dtm is checked too.
# Usage: bash download_missing.sh [DEST] [PARALLEL]
set -euo pipefail
DEST="${1:-/data/bt4}"; JOBS="${2:-8}"
export PATH=/venv/main/bin:$PATH HF_XET_HIGH_PERFORMANCE=1
mkdir -p "$DEST"
python3 - "$DEST" > /root/missing_plan.tsv <<'PY'
import json, os, re, sys, urllib.request
dest = sys.argv[1]
REPOS = ["vondele/master-binpacks_relabel", "vondele/linrock_relabel_1", "vondele/linrock_relabel_2",
         "vondele/from_kaggle_2_relabel", "vondele/from_kaggle_1_relabel", "xushawn/test80-bt4-relabel"]
files = []
for repo in REPOS:
    tree = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true", timeout=60))
    for e in tree:
        p = e["path"]
        if e.get("type") == "file" and p.endswith(".binpack") and ".q." not in p:
            files.append((repo, p, (e.get("lfs") or {}).get("size") or e.get("size") or 0))
base = lambda p: re.sub(r"\.part_\d+(?=\.binpack)", "", p)
whole = {base(p) for _, p, _ in files if ".part_" not in p}
files = [f for f in files if ".part_" not in f[1] or base(f[1]) not in whole]
tree = json.load(urllib.request.urlopen("https://huggingface.co/api/datasets/vondele/rescored/tree/main", timeout=60))
for e in tree:
    if e["path"] == "tb5dtm.binpack":
        files.append(("vondele/rescored", e["path"], (e.get("lfs") or {}).get("size") or e.get("size") or 0))
missing = 0
for repo, p, size in files:
    local = os.path.join(dest, os.path.basename(p))
    if os.path.exists(local) and os.path.getsize(local) == size:
        continue
    missing += 1
    print(f"{repo}\t{p}\t{size}")
print(f"# planned {len(files)} files, missing {missing}", file=sys.stderr)
PY
awk -F'\t' '{s+=$3} END{printf "missing: %d files, %.1f GB\n", NR, s/1e9}' /root/missing_plan.tsv
dl_one() {
  local repo="$1" f="$2" dest="$3"
  echo "[$(date +%H:%M:%S)] START  $f"
  if hf download "$repo" --repo-type dataset --include "$f" --local-dir "$dest" >> /root/download_missing.log 2>&1; then
    [ "$(dirname "$f")" != "." ] && mv -f "$dest/$f" "$dest/$(basename "$f")"
    echo "[$(date +%H:%M:%S)] OK     $f"
  else
    echo "[$(date +%H:%M:%S)] FAILED $f (run again: it resumes)"
  fi
}
export -f dl_one
cut -f1,2 /root/missing_plan.tsv | xargs -P "$JOBS" -d'\n' -I{} bash -c 'IFS=$'"'"'\t'"'"' read -r r f <<< "{}"; dl_one "$r" "$f" "$1"' _ "$DEST"
echo "=== done: $(ls "$DEST"/*.binpack | wc -l) binpacks, $(du -sh --exclude=.cache "$DEST" | cut -f1)"
