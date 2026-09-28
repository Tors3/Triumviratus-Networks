#!/usr/bin/env python3
r"""check_loader_indices.py — verifica [C]: gli indici PawnPair del loader C++
COMPILATO coincidono col riferimento Python puro (verify_graft.py).

Da lanciare DALLA cartella nnue-pytorch della copia di lavoro (serve la dll):
  py -3.12 ..\graft\check_loader_indices.py
"""
import ctypes
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())  # la cartella nnue-pytorch (per il package data_loader)
from verify_graft import pawn_pair_indices, WHITE, BLACK

import data_loader.stream as stream
from data_loader._native import SparseBatch

# La stringa per il C++ e' quella SENZA '^' (INPUT_FEATURE_NAME)
FEATURES_CPP = "Full_Threats+HalfKAv2_hm+PawnPair"
# Spazio indici di TRAINING: halfka usa le dimensioni INTERNE (24576, 12 piece
# types), non quelle di export (22528). L'export (.nnue) e' invece 87808.
PP_OFFSET = 60720 + 24576  # = 85296
PP_END = PP_OFFSET + 4560  # = 89856

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",     # startpos: 36 coppie
    "7k/8/4p3/4p3/3PP3/8/8/K7 w - - 0 1",                            # phalanx + doppiati: 6
    "7k/8/8/8/1PP5/8/8/7K w - - 0 1",                                # mirror re bianco file h
    "r1bq1rk1/pp2ppbp/2np1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQ1RK1 w - - 0 8",  # mediogioco reale
    "8/2p5/1p6/pP5p/P6P/8/8/K6k w - - 0 1",                          # catene bloccate + en passant field
    "4k3/8/8/8/8/8/8/4K3 w - - 0 1",                                 # zero pedoni: zero coppie
]

stream.c_lib.dll.get_sparse_batch_from_fens.restype = ctypes.POINTER(SparseBatch)

batch_ptr = stream.get_sparse_batch_from_fens(
    FEATURES_CPP, FENS, [0] * len(FENS), [30] * len(FENS), [0] * len(FENS)
)
b = batch_ptr.contents
assert b.num_inputs == PP_END, f"num_inputs {b.num_inputs} != {PP_END}"

white = np.ctypeslib.as_array(b.white, shape=(b.size, b.max_active_features))
black = np.ctypeslib.as_array(b.black, shape=(b.size, b.max_active_features))

ok = True
for i, fen in enumerate(FENS):
    for name, arr, persp in (("white", white, WHITE), ("black", black, BLACK)):
        got = sorted(int(x) - PP_OFFSET for x in arr[i] if PP_OFFSET <= x < PP_END)
        want = pawn_pair_indices(fen, persp)
        # NB: la prospettiva del loader e' il COLORE (Color::White/Black), come il riferimento
        if got != want:
            ok = False
            print(f"[FAIL] fen={fen}  persp={name}")
            print(f"       C++    ({len(got)}): {got[:12]}...")
            print(f"       Python ({len(want)}): {want[:12]}...")
        else:
            print(f"[ok] {name} {len(got):3d} coppie  {fen[:40]}")

stream.destroy_sparse_batch(batch_ptr)
print("\n[C] " + ("OK — loader C++ = riferimento Python su tutte le FEN" if ok else "FALLITO"))
sys.exit(0 if ok else 1)
