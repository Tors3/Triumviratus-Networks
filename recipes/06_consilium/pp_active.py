import sys, chess, chess.pgn
PGN=sys.argv[1]; N=int(sys.argv[2])
tot=full=w3=0; g_n=0; moved_full=moved_w3=nmoves=0
with open(PGN,encoding="utf-8",errors="ignore") as f:
    while g_n<N:
        g=chess.pgn.read_game(f)
        if g is None: break
        g_n+=1; b=g.board(); ply=0
        for mv in g.mainline_moves():
            was_pawn = b.piece_type_at(mv.from_square)==chess.PAWN
            b.push(mv); ply+=1
            if ply<=4: continue
            sq=[s for s in chess.SquareSet(b.pawns)]
            n=len(sq); tot+=1
            full += n*(n-1)//2
            k=sum(1 for i in range(n) for j in range(i+1,n)
                  if abs(chess.square_file(sq[i])-chess.square_file(sq[j]))<=1)
            w3 += k
            if was_pawn and n:
                nmoves+=1
                # coppie toccate dallo spostamento di UN pedone: n-1 nel pieno,
                # quelle entro 1 colonna nel 3-wide
                t=chess.square_file(mv.to_square)
                moved_full += n-1
                moved_w3   += sum(1 for s in sq if s!=mv.to_square and abs(chess.square_file(s)-t)<=1)
print("posizioni campionate: %d (%d partite)" % (tot,g_n))
print("feature PawnPair ATTIVE per posizione:")
print("   pieno  : %.1f" % (full/tot))
print("   3-wide : %.1f   (%.2fx meno)" % (w3/tot, full/w3))
print("aggiornamenti accumulatore per MOSSA DI PEDONE (%d mosse):" % nmoves)
print("   pieno  : %.1f coppie" % (moved_full/nmoves))
print("   3-wide : %.1f coppie  (%.2fx meno)" % (moved_w3/nmoves, moved_full/moved_w3))
