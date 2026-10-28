"""Summary of the learned routing table of Consilium-2 (router_em.jsonl lines in the file given as argument)."""
import json
import sys

R = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
SIG = 54
UPPER = [3, 5, 8, 9, 13, 18, 25]
NAMES = ["opp NPM <=3", "4-5", "6-8", "9", "10-13", "14-18", "19-25", ">25"]


def npm(s):
    return 3 * (s // 18) + 3 * ((s // 6) % 3) + 5 * ((s // 2) % 3) + 9 * (s % 2)


def default(c):
    o, e = npm(c % SIG), 0
    while e < 7 and o > UPPER[e]:
        e += 1
    return e


def desc(s):
    return f"N{s // 18}B{(s // 6) % 3}R{(s // 2) % 3}Q{s % 2}"


t = R[-1]["table"]
print("rounds (epoch, cells moved, crossing after, loss gain):",
      [(r["epoch"], r["cells_moved"], round(r["crossing_after"], 4), r["loss_gain"]) for r in R])
moved = [c for c in range(SIG * SIG) if t[c] != default(c)]
print("cells differing from the initial table:", len(moved), "of", SIG * SIG)
for e in range(8):
    print(f"expert {e} ({NAMES[e]:>11}): share {R[-1]['shares'][e] * 100:5.1f}%")
for c in moved:
    own, opp = c // SIG, c % SIG
    print(f"  own {desc(own)} (npm {npm(own):2d}) vs opp {desc(opp)} (npm {npm(opp):2d}): expert {default(c)} -> {t[c]}")
