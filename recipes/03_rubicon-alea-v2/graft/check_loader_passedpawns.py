#!/usr/bin/env python3
r"""check_loader_passedpawns.py — verifica [C]: gli indici PassedPawns del
loader C++ COMPILATO coincidono col riferimento Python puro
(verify_passedpawns.py). Verifica ANCHE che gli indici PawnPair restino
identici (il blocco nuovo non deve spostare i vecchi).

Da lanciare DALLA cartella nnue-pytorch della copia di lavoro (serve la dll
RICOMPILATA con la feature string a 4 blocchi):
  py -3.12 ..\graft\check_loader_passedpawns.py
"""
import ctypes
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())  # la cartella nnue-pytorch (per il package data_loader)
from verify_graft import pawn_pair_indices, WHITE, BLACK
from verify_passedpawns import passed_pawn_indices

import data_loader.stream as stream
from data_loader._native import SparseBatch

# La stringa per il C++ e' quella SENZA '^' (INPUT_FEATURE_NAME)
FEATURES_CPP = "Full_Threats+HalfKAv2_hm+PawnPair+PassedPawns"
# Spazio indici di TRAINING: halfka usa le dimensioni INTERNE (24576).
PP_OFFSET = 60720 + 24576      # = 85296 (PawnPair)
PP_END = PP_OFFSET + 4560      # = 89856
PS_OFFSET = PP_END             # = 89856 (PassedPawns)
PS_END = PS_OFFSET + 96        # = 89952

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",     # startpos: 0 passati
    "4k3/8/8/8/8/8/P7/4K3 w - - 0 1",                                # a2 solo: 1 passato
    "4k3/8/8/8/8/P7/P7/4K3 w - - 0 1",                               # doppiati: solo a3
    "4k3/8/8/4P3/4p3/8/8/4K3 w - - 0 1",                             # incrociati: 2 passati
    "4k3/4P3/8/8/8/8/3p4/4K3 w - - 0 1",                             # 7a traversa + d2: 2
    "8/2p5/1p6/pP5p/P6P/8/8/K6k w - - 0 1",                          # catene bloccate
    "r1bq1rk1/pp2ppbp/2np1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQ1RK1 w - - 0 8",  # mediogioco reale
    "4k3/8/8/8/8/8/8/4K3 w - - 0 1",                                 # zero pedoni
]

stream.c_lib.dll.get_sparse_batch_from_fens.restype = ctypes.POINTER(SparseBatch)

batch_ptr = stream.get_sparse_batch_from_fens(
    FEATURES_CPP, FENS, [0] * len(FENS), [30] * len(FENS), [0] * len(FENS)
)
b = batch_ptr.contents
assert b.num_inputs == PS_END, f"num_inputs {b.num_inputs} != {PS_END}"

white = np.ctypeslib.as_array(b.white, shape=(b.size, b.max_active_features))
black = np.ctypeslib.as_array(b.black, shape=(b.size, b.max_active_features))

ok = True
for i, fen in enumerate(FENS):
    for name, arr, persp in (("white", white, WHITE), ("black", black, BLACK)):
        got_ps = sorted(int(x) - PS_OFFSET for x in arr[i] if PS_OFFSET <= x < PS_END)
        want_ps = passed_pawn_indices(fen, persp)
        got_pp = sorted(int(x) - PP_OFFSET for x in arr[i] if PP_OFFSET <= x < PP_END)
        want_pp = pawn_pair_indices(fen, persp)
        if got_ps != want_ps or got_pp != want_pp:
            ok = False
            print(f"[FAIL] fen={fen}  persp={name}")
            if got_ps != want_ps:
                print(f"       passed C++    ({len(got_ps)}): {got_ps}")
                print(f"       passed Python ({len(want_ps)}): {want_ps}")
            if got_pp != want_pp:
                print(f"       pawnpair C++    ({len(got_pp)}): {got_pp[:12]}...")
                print(f"       pawnpair Python ({len(want_pp)}): {want_pp[:12]}...")
        else:
            print(f"[ok] {name} passati={len(got_ps)} coppie={len(got_pp):3d}  {fen[:40]}")

stream.destroy_sparse_batch(batch_ptr)
print("\n[C] " + ("OK — loader C++ = riferimento Python su tutte le FEN" if ok else "FALLITO"))
sys.exit(0 if ok else 1)
