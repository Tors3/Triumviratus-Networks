"""Elo of "cand" against "base" from fastchess PGNs, pentanomial over game pairs (same Round tag) (22/10/2026).
Usage: python elo_pgn.py label:file1.pgn,file2.pgn [label:...]"""
import math
import re
import sys
from collections import defaultdict


def games(path):
    tags, out = {}, []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r'\[(\w+) "(.*)"\]', line)
        if m:
            tags[m.group(1)] = m.group(2)
            if m.group(1) == "Result":
                r = m.group(2)
                if r in ("1-0", "0-1", "1/2-1/2"):
                    s = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[r]
                    if tags.get("White") != "cand":
                        s = 1.0 - s
                    out.append((tags.get("Round"), s))
                tags = {}
    return out


def elo(score):
    score = min(max(score, 1e-6), 1 - 1e-6)
    return -400 * math.log10(1 / score - 1)


def report(label, paths):
    rows = []
    for p in paths:
        g = games(p)
        pairs = defaultdict(list)
        for rnd, s in g:
            pairs[(p, rnd)].append(s)
        rows.append((p, g, pairs))
    allg = [s for _, g, _ in rows for _, s in g]
    pair_scores = [sum(v) / 2 for _, _, pr in rows for v in pr.values() if len(v) == 2]
    n = len(allg)
    mu = sum(allg) / n
    w = sum(1 for s in allg if s == 1)
    l = sum(1 for s in allg if s == 0)
    d = n - w - l
    m = len(pair_scores)
    pm = sum(pair_scores) / m
    var = sum((x - pm) ** 2 for x in pair_scores) / (m - 1) / m
    se = math.sqrt(var)
    e, lo, hi = elo(pm), elo(pm - 1.96 * se), elo(pm + 1.96 * se)
    per = "  ".join(f"s{i}: {len(g)} g {elo(sum(s for _, s in g) / len(g)):+.1f}" for i, (_, g, _) in enumerate(rows))
    print(f"{label:<8} games {n:5d} (+{w} ={d} -{l})  pairs {m:4d}  Elo {e:+6.1f}  [{lo:+.1f}, {hi:+.1f}]  "
          f"(+-{(hi - lo) / 2:.1f})   {per}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        label, files = arg.split(":", 1)
        report(label, files.split(","))
