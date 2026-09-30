#!/usr/bin/env python3
"""Occupanza dei LayerStack: quante posizioni finiscono in ogni bucket.
Confronta 3 criteri: 8 bucket (attuale), 16 per materiale puro, 16 = 8 materiale x donne.
Campiona N partite dal PGN e le rigioca (le posizioni intermedie non sono nel PGN).
"""
import sys, chess, chess.pgn, io

PGN = sys.argv[1]
NGAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
SKIP = 4  # skip_plies del nostro filtro

h8 = [0]*8
h16 = [0]*16
hq = [0]*16
tot = 0
games = 0
with open(PGN, encoding="utf-8", errors="ignore") as f:
    while games < NGAMES:
        g = chess.pgn.read_game(f)
        if g is None:
            break
        games += 1
        b = g.board()
        ply = 0
        for mv in g.mainline_moves():
            b.push(mv)
            ply += 1
            if ply <= SKIP:
                continue
            n = chess.popcount(b.occupied)
            tot += 1
            h8[min(7, (n - 1) // 4)] += 1
            h16[min(15, (n - 1) // 2)] += 1
            # 8 fasce di materiale x {con donne, senza donne}
            q = 1 if (b.pieces_mask(chess.QUEEN, chess.WHITE) | b.pieces_mask(chess.QUEEN, chess.BLACK)) else 0
            hq[min(7, (n - 1) // 4) * 2 + q] += 1

def show(name, h, lab):
    print("\n== %s  (%d posizioni, %d partite)" % (name, tot, games))
    mx = max(h) or 1
    for i, v in enumerate(h):
        pct = 100.0 * v / tot if tot else 0
        bar = "#" * int(40 * v / mx)
        print("  b%-2d %-22s %7.3f%%  %-40s %d" % (i, lab(i), pct, bar, v))
    nz = [v for v in h if v > 0]
    print("  -> bucket vuoti/quasi (<0.5%%): %d su %d | rapporto max/min(non-zero): %.0fx"
          % (sum(1 for v in h if 100.0*v/tot < 0.5), len(h), (max(h)/min(nz)) if nz else 0))

show("ATTUALE: 8 bucket, (pezzi-1)//4", h8, lambda i: "%d-%d pezzi" % (4*i+1, 4*i+4))
show("16 bucket, (pezzi-1)//2 [materiale puro]", h16, lambda i: "%d-%d pezzi" % (2*i+1, 2*i+2))
show("16 bucket, 8 materiale x donne", hq,
     lambda i: "%d-%d pz, %s" % (4*(i//2)+1, 4*(i//2)+4, "CON donne" if i % 2 else "senza"))
