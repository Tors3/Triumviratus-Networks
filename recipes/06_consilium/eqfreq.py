import sys, chess, chess.pgn
PGN=sys.argv[1]; N=int(sys.argv[2])
cnt=[0]*33; g_n=0
with open(PGN,encoding="utf-8",errors="ignore") as f:
    while g_n<N:
        g=chess.pgn.read_game(f)
        if g is None: break
        g_n+=1; b=g.board(); ply=0
        for mv in g.mainline_moves():
            b.push(mv); ply+=1
            if ply<=4: continue
            n=chess.popcount(b.occupied)
            if n<4: continue          # min_pieces=4 del nostro filtro
            cnt[n]+=1
tot=sum(cnt)
print("posizioni (dopo skip_plies=4 e min_pieces=4):",tot)
target=tot/16.0
edges=[];acc=0;lo=4
for n in range(4,33):
    acc+=cnt[n]
    if acc>=target*(len(edges)+1) and len(edges)<15:
        edges.append(n); 
print("\n16 bucket a FREQUENZA UGUALE — taglio sul numero di pezzi:")
prev=4; occ=[]
bounds=edges+[32]
for i,e in enumerate(bounds):
    s=sum(cnt[prev:e+1]); occ.append(s)
    print("  b%-2d  %2d-%2d pezzi   %6.2f%%   %d" % (i,prev,e,100*s/tot,s))
    prev=e+1
nz=[v for v in occ if v]
print("  -> rapporto max/min: %.1fx  (materiale puro //2: 385x, materiale x donne: 781x)" % (max(nz)/min(nz)))
