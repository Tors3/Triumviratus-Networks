#!/usr/bin/env python3
r"""verify_graft.py — verifica del graft PawnPair in 3 parti.

[A] Riferimento PYTHON PURO degli indici pawn-pair (stessa spec del loader C++:
    96 slot orientati, banda di file +-1, hi*(hi-1)/2+lo) + self-check su FEN
    note (startpos=36 coppie, phalanx, doppiati). Riusabile per il port engine.
[B] Identita' zero-init: forward del feature transformer del modello GRAFTATO
    vs modello v1 sugli STESSI indici -> output identici; e con le feature
    pawn-pair ATTIVE -> ancora identici (il blocco a zero non contribuisce).
    La testa (layer_stacks) e' verificata per uguaglianza dei parametri: testa
    identica + FT identico => rete identica, senza servire dati veri.
[C] (in smoke_train, dopo la compilazione del loader) indici C++ vs riferimento [A].

USO (dalla cartella nnue-pytorch della copia di lavoro):
  python ..\graft\verify_graft.py                          # solo [A]
  python ..\graft\verify_graft.py v1.nnue grafted.pt       # [A] + [B]
"""
import sys

# ---------------------------------------------------------------------------
# [A] Riferimento puro-Python degli indici pawn-pair
# ---------------------------------------------------------------------------

WHITE, BLACK = 0, 1


def parse_fen_minimal(fen: str):
    """Ritorna (pawns=[(sq, color)], ksq=[white_ksq, black_ksq]). sq: a1=0..h8=63."""
    board = fen.split()[0]
    pawns, ksq = [], [None, None]
    rank = 7
    file = 0
    for ch in board:
        if ch == "/":
            rank -= 1
            file = 0
        elif ch.isdigit():
            file += int(ch)
        else:
            sq = rank * 8 + file
            if ch == "P":
                pawns.append((sq, WHITE))
            elif ch == "p":
                pawns.append((sq, BLACK))
            elif ch == "K":
                ksq[WHITE] = sq
            elif ch == "k":
                ksq[BLACK] = sq
            file += 1
    return pawns, ksq


def orient_xor(perspective: int, king_sq: int) -> int:
    """Identico a OrientTBL del loader: rank-flip per prospettiva nera (xor 56),
    file-flip se il re della prospettiva sta sui file e-h (xor 7)."""
    x = 56 if perspective == BLACK else 0
    if (king_sq & 7) >= 4:
        x ^= 7
    return x


def pawn_id(perspective: int, king_sq: int, pawn_color: int, sq: int) -> int:
    o = sq ^ orient_xor(perspective, king_sq)
    return (48 if pawn_color != perspective else 0) + o - 8


def pawn_pair_indices(fen: str, perspective: int):
    """Indici pawn-pair attivi (ordinati) per una prospettiva. Spec = loader C++."""
    pawns, ksq = parse_fen_minimal(fen)
    k = ksq[perspective]
    out = []
    for i in range(len(pawns)):
        for j in range(i + 1, len(pawns)):
            (si, ci), (sj, cj) = pawns[i], pawns[j]
            if abs((si & 7) - (sj & 7)) > 1:
                continue
            a, b = pawn_id(perspective, k, ci, si), pawn_id(perspective, k, cj, sj)
            hi, lo = max(a, b), min(a, b)
            out.append(hi * (hi - 1) // 2 + lo)
    return sorted(out)


def self_check():
    START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    for persp in (WHITE, BLACK):
        idx = pawn_pair_indices(START, persp)
        # 7 coppie own-own bianche + 7 nere + 22 cross (banda file +-1) = 36
        assert len(idx) == 36, f"startpos persp={persp}: {len(idx)} != 36"
        assert len(set(idx)) == 36, "indici duplicati"
        assert all(0 <= i < 4560 for i in idx), "indice fuori range"

    # phalanx d4-e4 bianca + doppiati e5/e6 neri, re agli angoli opposti
    fen = "7k/8/4p3/4p3/3PP3/8/8/K7 w - - 0 1"
    for persp in (WHITE, BLACK):
        idx = pawn_pair_indices(fen, persp)
        # coppie in banda: (d4,e4), (e6,e5), (d4,e5), (d4,e6), (e4,e5), (e4,e6) = 6
        assert len(idx) == 6, f"phalanx persp={persp}: {len(idx)} != 6"

    # mirror del re: stessa posizione, re bianco su file h -> indici DIVERSI per
    # la prospettiva bianca (file-flip), uguali per la nera (il SUO re non si muove).
    # Pedoni b4+c4: NON simmetrici rispetto al flip (b,c -> g,f).
    fen_a = "7k/8/8/8/1PP5/8/8/K7 w - - 0 1"
    fen_h = "7k/8/8/8/1PP5/8/8/7K w - - 0 1"
    assert pawn_pair_indices(fen_a, WHITE) != pawn_pair_indices(fen_h, WHITE), "file-mirror non applicato"
    assert pawn_pair_indices(fen_a, BLACK) == pawn_pair_indices(fen_h, BLACK), "prospettiva nera contaminata dal re bianco"
    print("[A] OK — riferimento pawn-pair: startpos=36, phalanx=6, mirror re corretto")


# ---------------------------------------------------------------------------
# [B] Identita' zero-init del modello graftato
# ---------------------------------------------------------------------------

def verify_identity(v1_nnue: str, grafted_pt: str):
    import os
    sys.path.insert(0, os.getcwd())  # la cartella nnue-pytorch

    import torch
    from model.config import ModelConfig
    from model.utils.load_model import load_model

    old = load_model(v1_nnue, "Full_Threats+HalfKAv2_hm^", ModelConfig())
    new = torch.load(grafted_pt, weights_only=False, map_location="cpu").model
    old.eval(); new.eval()

    # 1) testa e blocchi condivisi: parametri IDENTICI
    os_, ns = old.state_dict(), new.state_dict()
    for k in os_:
        assert torch.equal(os_[k], ns[k]), f"parametro condiviso diverso: {k}"
    # 2) blocco pawn-pair: tutto zero
    assert ns["input.features.2.weight"].abs().max().item() == 0.0, "blocco PawnPair non a zero"

    # 3) forward FT: stessi indici (blocchi vecchi, 0..85295 nello spazio di
    #    TRAINING: threats 60720 + halfka INTERNO 24576) -> output identico;
    #    aggiungendo indici pawn-pair attivi (85296..89855) -> ANCORA identico.
    torch.manual_seed(42)
    B, A_OLD = 16, 64
    idx_old = torch.randint(0, 85296, (B, A_OLD), dtype=torch.int32)
    val = torch.ones(B, A_OLD, dtype=torch.float32)
    out_old = old.input(idx_old, val, idx_old, val)

    idx_pp = torch.randint(85296, 89856, (B, 8), dtype=torch.int32)
    idx_new = torch.cat([idx_old, idx_pp], dim=1)
    val_new = torch.ones(B, A_OLD + 8, dtype=torch.float32)
    out_new = new.input(idx_new, val_new, idx_new, val_new)

    for a, b in zip(out_old, out_new):
        d = (a - b).abs().max().item()
        assert d == 0.0, f"output FT diverso (max diff {d})"
    print("[B] OK — zero-init: FT identico con e senza feature pawn-pair attive; testa identica")


if __name__ == "__main__":
    self_check()
    if len(sys.argv) == 3:
        verify_identity(sys.argv[1], sys.argv[2])
    else:
        print("(salta [B]: passa v1.nnue e grafted.pt per la verifica d'identita')")
