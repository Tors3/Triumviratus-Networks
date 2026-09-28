#!/usr/bin/env python3
r"""verify_passedpawns.py — verifica del graft PassedPawns in 3 parti.

[A] Riferimento PYTHON PURO degli indici passed-pawn (stessa spec del loader
    C++ e del motore: 96 slot orientati; "passed" = nessun pedone nemico su
    colonne +-1 davanti E nessun pedone proprio davanti sulla stessa colonna)
    + self-check su FEN note (startpos=0 passati, doppiati, incrociati,
    mirror del re). Riusabile per il port engine.
[B] Identita' zero-init: forward del feature transformer del modello GRAFTATO
    vs modello v2 sugli STESSI indici -> output identici; e con le feature
    passed-pawn ATTIVE -> ancora identici (il blocco a zero non contribuisce).
[C] (in check_loader_passedpawns.py, dopo la compilazione del loader) indici
    C++ vs riferimento [A].

USO (dalla cartella nnue-pytorch della copia di lavoro):
  python ..\graft\verify_passedpawns.py                       # solo [A]
  python ..\graft\verify_passedpawns.py v2.nnue grafted.pt    # [A] + [B]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # per verify_graft
from verify_graft import WHITE, BLACK, parse_fen_minimal, pawn_id

# ---------------------------------------------------------------------------
# [A] Riferimento puro-Python degli indici passed-pawn
# ---------------------------------------------------------------------------

# Spazio di TRAINING della v2 (3 blocchi): threats 60720 + halfka INTERNO
# 24576 + pawnpair 4560 = 89856. Il blocco PassedPawns e' appeso in coda.
V2_TRAIN_INPUTS = 89856
PASSED_DIMS = 96


def is_passed(sq: int, color: int, pawns) -> bool:
    """pawns = [(sq, color)] RAW. Nessun pedone nemico su colonne +-1 davanti,
    nessun pedone proprio davanti sulla stessa colonna (doppiato dietro != passato)."""
    f, r = sq & 7, sq >> 3
    for psq, pc in pawns:
        if psq == sq:
            continue
        pf, pr = psq & 7, psq >> 3
        ahead = pr > r if color == WHITE else pr < r
        if not ahead:
            continue
        if pc != color and abs(pf - f) <= 1:
            return False
        if pc == color and pf == f:
            return False
    return True


def passed_pawn_indices(fen: str, perspective: int):
    """Indici passed-pawn attivi (0..95, ordinati) per una prospettiva.
    Spec = loader C++ = motore (passed_pawns.cpp)."""
    pawns, ksq = parse_fen_minimal(fen)
    k = ksq[perspective]
    out = []
    for sq, c in pawns:
        if is_passed(sq, c, pawns):
            out.append(pawn_id(perspective, k, c, sq))
    return sorted(out)


def self_check():
    # startpos: ogni pedone ha il dirimpettaio nemico nella span -> 0 passati
    START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    for persp in (WHITE, BLACK):
        assert passed_pawn_indices(START, persp) == [], "startpos deve avere 0 passati"

    # pedone bianco a2 solo -> 1 passato (own per il bianco, enemy per il nero)
    fen = "4k3/8/8/8/8/8/P7/4K3 w - - 0 1"
    iw = passed_pawn_indices(fen, WHITE)
    ib = passed_pawn_indices(fen, BLACK)
    assert len(iw) == 1 and 0 <= iw[0] < 48, f"own passer atteso in 0..47: {iw}"
    assert len(ib) == 1 and 48 <= ib[0] < 96, f"enemy passer atteso in 48..95: {ib}"

    # doppiati a2+a3 senza neri: a3 passato, a2 NO (proprio pedone davanti)
    fen = "4k3/8/8/8/8/P7/P7/4K3 w - - 0 1"
    assert len(passed_pawn_indices(fen, WHITE)) == 1, "doppiato dietro non e' passato"

    # e4 bianco vs d5 nero: nessuno dei due e' passato (nemico in span)
    fen = "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1"
    assert passed_pawn_indices(fen, WHITE) == [], "e4/d5 non sono passati"

    # e5 bianco vs e4 nero (incrociati, stessa colonna): ENTRAMBI passati
    fen = "4k3/8/8/4P3/4p3/8/8/4K3 w - - 0 1"
    assert len(passed_pawn_indices(fen, WHITE)) == 2, "pedoni incrociati = entrambi passati"

    # e7 bianco (7a traversa): sempre passato anche con pedone nemico dietro-adiacente
    fen = "4k3/4P3/8/8/8/8/3p4/4K3 w - - 0 1"
    idx = passed_pawn_indices(fen, WHITE)
    assert len(idx) == 2, f"e7 e d2 sono entrambi passati: {idx}"

    # mirror del re: passer b4, re bianco su a1 vs h1 -> indici DIVERSI per la
    # prospettiva bianca (file-flip), UGUALI per la nera (il SUO re non si muove)
    fen_a = "7k/8/8/8/1P6/8/8/K7 w - - 0 1"
    fen_h = "7k/8/8/8/1P6/8/8/7K w - - 0 1"
    assert passed_pawn_indices(fen_a, WHITE) != passed_pawn_indices(fen_h, WHITE), "file-mirror non applicato"
    assert passed_pawn_indices(fen_a, BLACK) == passed_pawn_indices(fen_h, BLACK), "prospettiva nera contaminata dal re bianco"

    # range check su tutte le FEN usate
    for f in (START, fen, fen_a, fen_h):
        for persp in (WHITE, BLACK):
            assert all(0 <= i < PASSED_DIMS for i in passed_pawn_indices(f, persp)), "indice fuori range"

    print("[A] OK — riferimento passed-pawn: startpos=0, doppiati/incrociati/7a/mirror corretti")


# ---------------------------------------------------------------------------
# [B] Identita' zero-init del modello graftato
# ---------------------------------------------------------------------------

def verify_identity(v2_nnue: str, grafted_pt: str):
    sys.path.insert(0, os.getcwd())  # la cartella nnue-pytorch

    import torch
    from model.config import ModelConfig
    from model.utils.load_model import load_model

    old = load_model(v2_nnue, "Full_Threats+HalfKAv2_hm^+PawnPair", ModelConfig())
    new = torch.load(grafted_pt, weights_only=False, map_location="cpu").model
    old.eval(); new.eval()

    # 1) testa e blocchi condivisi: parametri IDENTICI
    os_, ns = old.state_dict(), new.state_dict()
    for k in os_:
        assert torch.equal(os_[k], ns[k]), f"parametro condiviso diverso: {k}"
    # 2) blocco passed-pawn: tutto zero
    assert ns["input.features.3.weight"].abs().max().item() == 0.0, "blocco PassedPawns non a zero"

    # 3) forward FT: stessi indici (blocchi vecchi, 0..89855 nello spazio di
    #    TRAINING) -> output identico; aggiungendo indici passed-pawn attivi
    #    (89856..89951) -> ANCORA identico.
    torch.manual_seed(42)
    B, A_OLD = 16, 64
    idx_old = torch.randint(0, V2_TRAIN_INPUTS, (B, A_OLD), dtype=torch.int32)
    val = torch.ones(B, A_OLD, dtype=torch.float32)
    out_old = old.input(idx_old, val, idx_old, val)

    idx_pp = torch.randint(V2_TRAIN_INPUTS, V2_TRAIN_INPUTS + PASSED_DIMS, (B, 8), dtype=torch.int32)
    idx_new = torch.cat([idx_old, idx_pp], dim=1)
    val_new = torch.ones(B, A_OLD + 8, dtype=torch.float32)
    out_new = new.input(idx_new, val_new, idx_new, val_new)

    for a, b in zip(out_old, out_new):
        d = (a - b).abs().max().item()
        assert d == 0.0, f"output FT diverso (max diff {d})"
    print("[B] OK — zero-init: FT identico con e senza feature passed-pawn attive; testa identica")


if __name__ == "__main__":
    self_check()
    if len(sys.argv) == 3:
        verify_identity(sys.argv[1], sys.argv[2])
    else:
        print("(salta [B]: passa v2.nnue e grafted.pt per la verifica d'identita')")
