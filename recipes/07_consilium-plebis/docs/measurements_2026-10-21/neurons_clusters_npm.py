"""How does the network itself divide the positions? Clustering of the first-layer outputs (09/10/2026).

Consilium picks the expert from the piece count. Here the network is asked which "regimes" it sees: the first-layer
outputs (1,024 bytes per evaluation, from the -DTRIUMV_NEURONS build) are clustered without labels (PCA + k-means), and
the clusters are compared with criteria that could pick the expert of a future network: material band (today's),
queens on the board, number of pawns, opposite castling, closed structure. The criterion that best explains the clusters
(normalised mutual information, NMI) is the candidate for Consilium-2. A suitable criterion must also change seldom
along the game, so as not to break the incremental update: queens and pawns change only with a few moves.
(26/10/2026: comments, labels and output translated to English.)

Needs the files meta_b<k>.bin (position of every record, nneurons.h from 09/10 evening) next to sample_b<k>.bin.

USAGE: python neurons_clusters.py <run_folder> [...] [--per-band 25000] [--k 4 6 8] [--out report.md]
"""
import argparse
import os

import numpy as np

W = 1024
META = 104
BAND = 'material band (today)'
BAND_Q = 'band x queens'


def load(dirs, per_band):
    X, B, MB = [], [], []
    for b in range(4):
        xs, ms = [], []
        for d in dirs:
            ps, pm = os.path.join(d, f'sample_b{b}.bin'), os.path.join(d, f'meta_b{b}.bin')
            if not (os.path.exists(ps) and os.path.exists(pm)) or not os.path.getsize(pm):
                continue
            s = np.fromfile(ps, dtype=np.uint8)
            m = np.fromfile(pm, dtype=np.uint8)
            n = min(len(s) // (8 + W), len(m) // META)
            xs.append(s[:n * (8 + W)].reshape(n, 8 + W))
            ms.append(m[:n * META].reshape(n, META))
        if not xs:
            continue
        x, m = np.concatenate(xs), np.concatenate(ms)
        rng = np.random.default_rng(b)
        idx = rng.permutation(len(x))[:per_band]
        X.append(x[idx])
        MB.append(m[idx])
        B.append(np.full(len(idx), b))
    return np.concatenate(X), np.concatenate(B), np.concatenate(MB)


def concepts(R, M):
    """Criteria from the position: bitboards P N B R Q K p n b r q k (a8 = 0), side to move."""
    bbs = M[:, :96].copy().view(np.uint64).reshape(-1, 12)
    cnt = np.zeros((len(bbs), 12), dtype=np.int32)
    for i in range(12):
        v = bbs[:, i]
        c = np.zeros(len(v), dtype=np.int32)
        for _ in range(64):
            c += (v & np.uint64(1)).astype(np.int32)
            v = v >> np.uint64(1)
        cnt[:, i] = c
    pieces_meta = cnt.sum(1)
    pieces_hdr = R[:, 0].astype(np.int32)
    ok = (pieces_meta == pieces_hdr).mean()
    queens = cnt[:, 4] + cnt[:, 10]
    pawns = cnt[:, 0] + cnt[:, 6]
    minors = cnt[:, 1] + cnt[:, 2] + cnt[:, 7] + cnt[:, 8]
    rooks = cnt[:, 3] + cnt[:, 9]

    def sqfile(bb):
        s = np.zeros(len(bb), dtype=np.int32)
        v = bb.copy()
        for q in range(64):
            hit = (v >> np.uint64(q)) & np.uint64(1)
            s = np.where(hit == 1, q % 8, s)
        return s
    kw, kb = sqfile(bbs[:, 5]), sqfile(bbs[:, 11])
    opposite = ((kw <= 2) & (kb >= 5)) | ((kw >= 5) & (kb <= 2))
    # blocked pawns: a white pawn with a black pawn right in front of it (a8 = 0: in front of White = square - 8)
    wp, bp = bbs[:, 0], bbs[:, 6]
    locked = np.zeros(len(bbs), dtype=np.int32)
    v = wp >> np.uint64(8)
    blk = v & bp
    for _ in range(64):
        locked += (blk & np.uint64(1)).astype(np.int32)
        blk = blk >> np.uint64(1)
    npm_w = 3 * (cnt[:, 1] + cnt[:, 2]) + 5 * cnt[:, 3] + 9 * cnt[:, 4]
    npm_b = 3 * (cnt[:, 7] + cnt[:, 8]) + 5 * cnt[:, 9] + 9 * cnt[:, 10]
    stm = R[:, 1].astype(np.int32)
    npm_opp = np.where(stm == 0, npm_b, npm_w)
    npm_own = np.where(stm == 0, npm_w, npm_b)
    q_opp = (np.where(stm == 0, cnt[:, 10], cnt[:, 4]) > 0).astype(int)
    q_own = (np.where(stm == 0, cnt[:, 4], cnt[:, 10]) > 0).astype(int)
    cls = lambda v: np.digitize(v, [6, 14, 23])
    C_new = {
        'N1 total NPM 4 bands (<=12/13-28/29-46/47+)': np.digitize(npm_w + npm_b, [13, 29, 47]),
        "N2 NPM of the side to move's opponent, 4 classes": cls(npm_opp),
        'N3 pair of NPM classes (side to move, other)': cls(npm_own) * 4 + cls(npm_opp),
        'N4 opponent NPM x opponent queen': cls(npm_opp) * 2 + q_opp,
        'N5 SF P4 (pieces-1)//8 x queens 0/1/2+': ((pieces_hdr - 1) // 8) * 3 + np.minimum(queens, 2),
        "N6 queen of the side to move's opponent (K32Q2)": q_opp,
        'N7 queens (side to move, other)': q_own * 2 + q_opp,
        'N8 total NPM 4 bands x queens 0/1/2+': np.digitize(npm_w + npm_b, [13, 29, 47]) * 3 + np.minimum(queens, 2),
    }
    C = {
        **C_new,
        BAND: np.digitize(pieces_hdr, [10, 16, 24]),
        'queens: 0 / 1 / 2+': np.minimum(queens, 2),
        'pawns: <=6 / 7-10 / 11-13 / 14+': np.digitize(pawns, [7, 11, 14]),
        'non-pawn pieces: <=2 / 3-6 / 7-10 / 11+': np.digitize(cnt[:, [1, 2, 3, 4, 7, 8, 9, 10]].sum(1), [3, 7, 11]),
        BAND_Q: np.digitize(pieces_hdr, [10, 16, 24]) * 3 + np.minimum(queens, 2),
        'opposite castling': opposite.astype(int),
        'closed structure (blocked pawns: 0-1 / 2-3 / 4+)': np.digitize(locked, [2, 4]),
        'rooks and minors: rooks 0/1+/3+ x minors 0-1/2-3/4+': np.digitize(rooks, [1, 3]) * 3 + np.digitize(minors, [2, 4]),
    }
    return C, ok


def nmi(a, b):
    a = np.unique(a, return_inverse=True)[1]
    b = np.unique(b, return_inverse=True)[1]
    n = len(a)
    ct = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(ct, (a, b), 1)
    p = ct / n
    pa, pb = p.sum(1), p.sum(0)
    nz = p > 0
    mi = (p[nz] * np.log(p[nz] / (pa[:, None] * pb[None, :])[nz])).sum()
    ha = -(pa[pa > 0] * np.log(pa[pa > 0])).sum()
    hb = -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    return mi / max(1e-12, np.sqrt(ha * hb))


def kmeans(Z, k, iters=40, seed=0):
    rng = np.random.default_rng(seed)
    C = Z[rng.choice(len(Z), k, replace=False)]
    for _ in range(iters):
        d = ((Z[:, None, :] - C[None, :, :]) ** 2).sum(2)
        lab = d.argmin(1)
        for j in range(k):
            if (lab == j).any():
                C[j] = Z[lab == j].mean(0)
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--per-band', type=int, default=25000)
    ap.add_argument('--k', type=int, nargs='+', default=[4, 6, 8])
    ap.add_argument('--pcs', type=int, default=32)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    R, band, M = load(a.dirs, a.per_band)
    X = R[:, 8:].astype(np.float32)
    C, ok = concepts(R, M)
    L = []
    say = lambda s='': (print(s), L.append(s))
    say(f'# Clustering of the first-layer outputs: {len(X)} evaluations '
        f'({", ".join(str((band == b).sum()) for b in range(4))} per band)')
    say(f'Check: piece count from the position = recorded piece count in {100 * ok:.2f}% of the records.')
    Xc = X - X.mean(0)
    Xc /= X.std(0) + 1e-6
    _, S, Vt = np.linalg.svd(Xc[:: max(1, len(Xc) // 40000)], full_matrices=False)
    ev = S ** 2 / (S ** 2).sum()
    Z = Xc @ Vt[:a.pcs].T
    say(f'PCA: the first {a.pcs} components explain {100 * ev[:a.pcs].sum():.1f}% of the variance '
        f'(first 5: {", ".join(f"{100 * e:.1f}%" for e in ev[:5])}).')
    say()
    say("## How much each criterion explains the network's clusters (NMI, 0 = nothing, 1 = all)")
    say('| criterion | ' + ' | '.join(f'k = {k}' for k in a.k) + ' |')
    say('|---|' + '---:|' * len(a.k))
    labs = {k: kmeans(Z, k, seed=k) for k in a.k}
    rows = []
    for name, v in C.items():
        rows.append((np.mean([nmi(labs[k], v) for k in a.k]), name, [nmi(labs[k], v) for k in a.k]))
    for _, name, vals in sorted(rows, reverse=True):
        say(f'| {name} | ' + ' | '.join(f'{x:.3f}' for x in vals) + ' |')
    # inside each band: does the network still divide the positions? by which criterion
    say()
    say('## Inside each band: clusters at k = 3 and the criterion that explains them best')
    say('| band | ' + ' | '.join(n for n in C if n != BAND and n != BAND_Q) + ' |')
    say('|---|' + '---:|' * (len(C) - 2))
    for b in range(4):
        sel = band == b
        if sel.sum() < 1000:
            continue
        lab = kmeans(Z[sel], 3, seed=b)
        say(f'| {b} | ' + ' | '.join(f'{nmi(lab, v[sel]):.3f}' for n, v in C.items()
                                     if n != BAND and n != BAND_Q) + ' |')
    # composition of the clusters at k = max
    k = max(a.k)
    say()
    say(f'## Composition of the {k} clusters')
    say('| cluster | share | mean band | mean queens | mean pawns | opposite castling | mean blocked pawns |')
    say('|---|---:|---:|---:|---:|---:|---:|')
    lab = labs[k]
    q = C['queens: 0 / 1 / 2+']
    for j in range(k):
        s = lab == j
        if not s.any():
            continue
        say(f'| {j} | {100 * s.mean():.1f}% | {band[s].mean():.2f} | {q[s].mean():.2f} | '
            f'{C["pawns: <=6 / 7-10 / 11-13 / 14+"][s].mean():.2f} | {100 * C["opposite castling"][s].mean():.0f}% | '
            f'{C["closed structure (blocked pawns: 0-1 / 2-3 / 4+)"][s].mean():.2f} |')
    if a.out:
        open(a.out, 'w', encoding='utf-8').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
