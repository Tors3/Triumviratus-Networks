"""Evaluation scale of Consilium-2 against Consilium on the same positions (21/10/2026).

  python scale_c2.py extract out.fen game1.pgn [game2.pgn ...]   positions from games (every 2nd ply from ply 12)
  python scale_c2.py analyze fens c2.eng consilium.eng c2.nnue      regressions per phase and per C2 expert

The .eng files are the engines' `evalbatch` output (psqt positional final per line, side to move). "raw" is
psqt + positional (the network, before the engine's eval scaling), "final" the value the search uses. For each group:
OLS slope of C2 on Consilium, the inverse of the OLS slope of Consilium on C2 (the two bracket the true scale when
both evaluations are noisy), the ratio of standard deviations, offset and correlation.
"""
import struct
import sys
from collections import defaultdict

import chess
import chess.pgn
import numpy as np


def extract(out, pgns):
    seen, fens = set(), []
    for path in pgns:
        with open(path, encoding="utf-8", errors="replace") as f:
            while True:
                g = chess.pgn.read_game(f)
                if g is None:
                    break
                b = g.board()
                for i, mv in enumerate(g.mainline_moves()):
                    b.push(mv)
                    if i >= 11 and i % 2 == 1 and not b.is_game_over():
                        key = b.board_fen() + (" w" if b.turn else " b")
                        if key not in seen:
                            seen.add(key)
                            fens.append(b.fen())
    with open(out, "w") as fo:
        fo.write("\n".join(fens) + "\n")
    print(f"{len(fens)} positions -> {out}")


def router_table(path):
    data = open(path, "rb").read()
    i = data.find(struct.pack("<III", 0x52544231, 2916, 8))
    assert i >= 0, "RTB1 section not found"
    return np.frombuffer(data[i + 12:i + 12 + 2916], dtype=np.uint8)


def sig(b, c):
    k = lambda pt: len(b.pieces(pt, c))
    return min(k(chess.KNIGHT), 2) * 18 + min(k(chess.BISHOP), 2) * 6 + min(k(chess.ROOK), 2) * 2 + min(k(chess.QUEEN), 1)


def stats(x, y):
    if len(x) < 30:
        return None
    b1 = np.polyfit(x, y, 1)               # y = s x + o
    b2 = np.polyfit(y, x, 1)               # x = s' y + o'
    return dict(n=len(x), ols=b1[0], off=b1[1], inv=1 / b2[0] if b2[0] else float("nan"),
                sd=np.std(y) / np.std(x), r=np.corrcoef(x, y)[0, 1], mabs=np.mean(np.abs(y)) / np.mean(np.abs(x)))


def show(title, groups, keyname):
    print(f"\n{title}")
    print(f"{keyname:>12} {'n':>6} {'OLS C2~C1':>10} {'1/OLS C1~C2':>12} {'sd ratio':>9} {'mean|.| ratio':>13} {'offset':>8} {'r':>6}")
    for k in sorted(groups):
        s = stats(*map(np.array, zip(*groups[k])))
        if s:
            print(f"{str(k):>12} {s['n']:6d} {s['ols']:10.3f} {s['inv']:12.3f} {s['sd']:9.3f} {s['mabs']:13.3f} {s['off']:8.1f} {s['r']:6.3f}")


def analyze(fens_path, c2_path, c1_path, net_path, cap=1500):
    fens = [l.strip() for l in open(fens_path) if l.strip()]
    c2 = np.loadtxt(c2_path, dtype=np.int64).reshape(-1, 3)
    c1 = np.loadtxt(c1_path, dtype=np.int64).reshape(-1, 3)
    assert len(fens) == len(c2) == len(c1)
    rt = router_table(net_path)
    for name, col in (("raw (psqt + positional)", None), ("final (search eval)", 2)):
        allg, phase, expert = defaultdict(list), defaultdict(list), defaultdict(list)
        for fen, a, b in zip(fens, c2, c1):
            x = int(b[0] + b[1]) if col is None else int(b[col])
            y = int(a[0] + a[1]) if col is None else int(a[col])
            if abs(x) > cap or abs(y) > cap:      # decided positions say nothing about the scale
                continue
            bd = chess.Board(fen)
            pc = len(bd.piece_map())
            us = bd.turn
            e = int(rt[sig(bd, us) * 54 + sig(bd, not us)])
            allg["all"].append((x, y))
            phase[(pc - 1) // 4].append((x, y))
            expert[e].append((x, y))
        print(f"\n===== {name}, |eval| <= {cap} on both sides")
        show("all positions", allg, "group")
        show("by layer-stack bucket ((pieces - 1) / 4, kings and pawns counted)", phase, "bucket")
        show("by C2 expert of the side to move", expert, "expert")


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(sys.argv[2], sys.argv[3:])
    else:
        analyze(*sys.argv[2:6])
