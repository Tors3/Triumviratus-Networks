"""Consilium-2 against Consilium (21/10/2026): inference cost on a fixed tree.

Runs `nnspeed D <fens>` (search/16_tdperft.inc: nnperft's tree, evaluations at the leaves and at one internal node in
three, null moves, WITHOUT the checks) on two PGO builds of the same source, -DTRIUMV_C2 on and off, each with its
own network embedded. The engine is pinned to one processor (cmd start /NODE /AFFINITY, as tools/uci_workload.py) and
the order is alternated per round. The tree depends only on the move generator, so both builds evaluate the same
positions: the time per evaluation compares the two networks' inference, incremental updates through make/unmake
included. The number of evaluations must be equal between the builds (checked).
Usage: python measure_nnspeed.py [--rounds 4] [--pin 0:2] [--depth 4] [--first 0]
"""
import argparse
import os
import re
import statistics
import subprocess

# Default locations of the measurement machine (PGO roots under <work>, the xperf kit's position files).
DEFAULT_BASE = r"<work>\c2pgo_base\motore\x64\Release\base_avx512.exe"
DEFAULT_C2 = r"<work>\c2pgo_c2\motore\x64\Release\c2_avx512.exe"
DEFAULT_TOOLS = os.path.join("<project>", "tools", "misure_xperf")
RX =re.compile(r"nnspeed (\d+) positions (\d+) nodes (\d+) evals (\d+) time (\d+) ms ([\d.]+) ns/eval")
LOG = open("nnspeed_results.txt", "a", encoding="utf-8")


def say(msg):
    print(msg, flush=True)
    LOG.write(msg + "\n")
    LOG.flush()


def run(exe, fens, depth, pin, net=""):
    node, cpu = pin.split(":")
    cmd = f'cmd /c start "" /B /WAIT /NODE {node} /AFFINITY {1 << int(cpu):X} "{exe}"'
    load = f"setoption name EvalFile value {net}\n" if net else ""
    p = subprocess.run(cmd, input=f"uci\n{load}isready\nnnspeed {depth} {fens}\nquit\n", capture_output=True, text=True)
    m = RX.search(p.stdout)
    if not m:
        raise SystemExit(f"no nnspeed line from {exe}:\n{p.stdout[-800:]}")
    return int(m.group(4)), float(m.group(6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--pin", default="0:2", help="node:cpu")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--first", type=int, default=0, help="only the first N positions of each set (0 = all)")
    ap.add_argument("--base", default=DEFAULT_BASE, help="PGO build without -DTRIUMV_C2 (Consilium network)")
    ap.add_argument("--c2", default=DEFAULT_C2, help="PGO build with -DTRIUMV_C2 (Consilium-2 network)")
    ap.add_argument("--tools", default=DEFAULT_TOOLS, help="folder with fens.txt (middlegame) and eg64.txt (endgames)")
    ap.add_argument("--c2net", default="", help="network file for the C2 build (EvalFile), instead of its embedded one")
    a = ap.parse_args()
    EXES = {"base": a.base, "c2": a.c2}
    NETS = {"base": "", "c2": a.c2net}
    sets = {"mg": os.path.join(a.tools, "fens.txt"), "eg": os.path.join(a.tools, "eg64.txt")}
    if a.first:
        for k, path in list(sets.items()):
            short = path.replace(".txt", f"_first{a.first}.txt")
            with open(path) as fi, open(short, "w") as fo:
                fo.writelines([l for l in fi if l.strip()][:a.first])
            sets[k] = short
    deltas = {k: [] for k in sets}
    for r in range(1, a.rounds + 1):
        order = ["base", "c2"] if r % 2 else ["c2", "base"]
        for k, fens in sets.items():
            res = {}
            for name in order:
                res[name] = run(EXES[name], fens, a.depth, a.pin, NETS[name])
                say(f"round {r} {k} {name:4s}: {res[name][1]:8.2f} ns/eval ({res[name][0]} evals)")
            if res["base"][0] != res["c2"][0]:
                say(f"  WARNING: different number of evaluations ({res['base'][0]} vs {res['c2'][0]})")
            d = 100.0 * (res["c2"][1] / res["base"][1] - 1.0)
            deltas[k].append(d)
            say(f"  round {r} {k}: C2 {d:+.2f}% time per evaluation")
    say("\n=== C2 against Consilium, time per evaluation (positive = C2 slower)")
    for k, ds in deltas.items():
        say(f"{k}: rounds {' '.join(f'{d:+.2f}' for d in ds)} | median {statistics.median(ds):+.2f}%"
            f" | mean {statistics.mean(ds):+.2f}%")


if __name__ == "__main__":
    main()
