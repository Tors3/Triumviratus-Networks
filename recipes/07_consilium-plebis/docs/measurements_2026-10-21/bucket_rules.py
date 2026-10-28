"""Occupancy and change frequency of candidate input-bucket rules, measured on game lines.

For each position of a sample of games (both perspectives) the script computes the bucket of every rule, then counts:
  - occupancy per bucket (share of perspective-positions), max/min ratio;
  - changes per ply along the game line, per perspective (each change = one hybrid/refresh of that perspective);
  - boundary captures: share of legal captures in a position that would change the bucket of a perspective
    (a proxy for the extra hybrids in the quiescence search, where almost every move is a capture).
Usage: python -I bucket_rules.py <pgn> <games> [<skip_plies>]
"""
import random
import sys
from collections import Counter

import chess
import chess.pgn

VAL = {chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def total_pieces(b):
    return chess.popcount(b.occupied)


def band4(n):  # Consilium today: <=9, 10-15, 16-23, >=24
    return 0 if n <= 9 else 1 if n <= 15 else 2 if n <= 23 else 3


def npm(b, color):
    return sum(VAL[pt] * chess.popcount(b.pieces_mask(pt, color)) for pt in VAL)


def opp_npm_class(b, persp):
    v = npm(b, not persp)
    return 0 if v <= 5 else 1 if v <= 13 else 2 if v <= 22 else 3


def queen(b, color):
    return 1 if b.pieces_mask(chess.QUEEN, color) else 0


def pawns_band(b):
    n = chess.popcount(b.pawns)
    return 0 if n <= 7 else 1 if n <= 10 else 2 if n <= 13 else 3


# Each rule: name -> (function(board, perspective) -> bucket, symmetric?)
RULES = {
    "R0 band4 (today)": lambda b, p: band4(total_pieces(b)),
    "R1 band4 x oppQ (8)": lambda b, p: band4(total_pieces(b)) * 2 + queen(b, not p),
    "R2 band2 x oppQ (4)": lambda b, p: (1 if total_pieces(b) >= 16 else 0) * 2 + queen(b, not p),
    "R3 SF P4 (pc-1)//8 x oppQ (8)": lambda b, p: ((total_pieces(b) - 1) // 8) * 2 + queen(b, not p),
    "R4 band4 x anyQ (8)": lambda b, p: band4(total_pieces(b)) * 2 + (1 if b.queens else 0),
    "R5 oppNPM class (4)": lambda b, p: opp_npm_class(b, p),
    "R6 oppNPM class x oppQ (8)": lambda b, p: opp_npm_class(b, p) * 2 + queen(b, not p),
    "R7 pawn band (4)": lambda b, p: pawns_band(b),
    "R8 band4 x pawn band (16)": lambda b, p: band4(total_pieces(b)) * 4 + pawns_band(b),
    "R9 oppQ only (2)": lambda b, p: queen(b, not p),
    "R10 ownNPM class (4)": lambda b, p: opp_npm_class(b, not p),
    "R11 total NPM band (4)": lambda b, p: npm_total_band(b),
    "R12 oppNPM 6 classes": lambda b, p: opp_npm6(b, p),
    "R13 oppNPM4 x ownQ (8)": lambda b, p: opp_npm_class(b, p) * 2 + queen(b, p),
    "R14 oppNPM4 x pawn2 (8)": lambda b, p: opp_npm_class(b, p) * 2 + (1 if chess.popcount(b.pawns) >= 11 else 0),
}


def npm_total_band(b):
    v = npm(b, chess.WHITE) + npm(b, chess.BLACK)
    return 0 if v <= 12 else 1 if v <= 28 else 2 if v <= 46 else 3


def opp_npm6(b, persp):
    v = npm(b, not persp)
    return 0 if v <= 3 else 1 if v <= 8 else 2 if v <= 13 else 3 if v <= 19 else 4 if v <= 25 else 5


def main():
    path, ngames = sys.argv[1], int(sys.argv[2])
    skip = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    random.seed(1)
    offsets = []
    with open(path, encoding="utf-8", errors="replace") as f:
        while True:
            off = f.tell()
            line = f.readline()
            if not line:
                break
            if line.startswith("[Event "):
                offsets.append(off)
    sample = sorted(random.sample(offsets, min(ngames, len(offsets))))
    occ = {r: Counter() for r in RULES}
    changes = Counter()
    bcap = Counter()
    ncap = 0
    nply = 0
    npos = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for off in sample:
            f.seek(off)
            game = chess.pgn.read_game(f)
            if game is None:
                continue
            board = game.board()
            prev = None
            for ply, move in enumerate(game.mainline_moves()):
                board.push(move)
                if ply < skip or total_pieces(board) < 3:
                    prev = None
                    continue
                cur = {r: (fn(board, chess.WHITE), fn(board, chess.BLACK)) for r, fn in RULES.items()}
                npos += 1
                for r in RULES:
                    occ[r][cur[r][0]] += 1
                    occ[r][cur[r][1]] += 1
                if prev is not None:
                    nply += 1
                    for r in RULES:
                        changes[r] += (cur[r][0] != prev[r][0]) + (cur[r][1] != prev[r][1])
                prev = cur
                # boundary captures: every legal capture from this position
                if npos % 2 == 0:
                    for m in board.generate_legal_captures():
                        board.push(m)
                        ncap += 1
                        for r, fn in RULES.items():
                            bcap[r] += (fn(board, chess.WHITE) != cur[r][0]) + (fn(board, chess.BLACK) != cur[r][1])
                        board.pop()
    print(f"games {len(sample)}, positions {npos}, plies with a parent {nply}, captures examined {ncap}")
    print(f"{'rule':34s} {'buckets':>7s} {'min%':>6s} {'max%':>6s} {'max/min':>8s} {'chg/ply':>8s} {'capt->chg':>9s}")
    for r in RULES:
        tot = sum(occ[r].values())
        shares = sorted(v / tot * 100 for v in occ[r].values())
        nb = len(occ[r])
        print(f"{r:34s} {nb:7d} {shares[0]:6.2f} {shares[-1]:6.2f} {shares[-1] / shares[0]:8.1f} "
              f"{changes[r] / max(nply, 1):8.4f} {bcap[r] / max(ncap, 1):9.4f}")
    print("chg/ply = bucket changes per ply summed over both perspectives (2 = both change every ply)")
    print("capt->chg = perspective bucket changes per legal capture, summed over both perspectives")
    for r in RULES:
        tot = sum(occ[r].values())
        print(r, " ".join(f"{k}:{v / tot * 100:.1f}" for k, v in sorted(occ[r].items())))


if __name__ == "__main__":
    main()
