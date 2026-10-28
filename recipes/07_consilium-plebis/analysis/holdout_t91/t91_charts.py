"""Charts of the t91 loss of Consilium Plebis against Consilium by expert and by material bucket (26/10/2026).

Values from t91_expert.py and t91_bucket.py on the 300-batch runs of 26/10 (C:\\rz\\c2eng\\scale\\t91_2026-10-26_expert and
t91_2026-10-26_bucket): every network on the same 2,457,600 positions, difference in % of Consilium's loss.
  c2_t91_by_expert.svg   lambda 0.75, expert of the side to move's perspective, epochs 376 and 435 (dumbbells);
  c2_t91_by_bucket.svg   epoch 435, layer-stack bucket ((pieces - 1) / 4), lambda 0.75 and lambda 0.
Usage: python t91_charts.py <out_dir>
"""
import os
import sys

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

# expert, opponent's non-pawn material (N=B=3, R=5, Q=9), share of training positions %, ep376, ep435, s.e. (ep435)
EXPERTS = [(0, "at most a minor", 8.2, -4.92, -6.18, 0.68), (1, "a rook", 7.6, -6.08, -7.25, 0.68),
           (2, "two minors or R+minor", 12.3, -3.48, -4.34, 0.33), (3, "a queen alone", 3.1, -7.26, -8.32, 1.02),
           (4, "10-13 units", 13.1, -2.12, -3.13, 0.33), (5, "14-18 units", 11.8, -0.43, -1.62, 0.35),
           (6, "19-25 units", 22.4, -0.91, -2.08, 0.24), (7, "26-31 units", 21.5, -0.68, -2.34, 0.24)]
# bucket, pieces, share of t91 positions %, lambda 0.75, lambda 0 (ep435); b0 (up to 4 pieces, 1.4%) is left out: +-11%
BUCKETS = [(1, "5-8", 5.6, -5.90, -3.87), (2, "9-12", 10.6, -6.61, -3.80), (3, "13-16", 16.2, -3.34, -1.85),
           (4, "17-20", 18.6, -1.33, -0.24), (5, "21-24", 18.1, -2.87, -0.79), (6, "25-28", 16.6, -1.52, -0.06),
           (7, "29-32", 13.0, -3.76, -0.50)]


def svg(w, h, body, title, subtitle):
    head = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="{FONT}">',
            f'<rect width="{w}" height="{h}" fill="{SURFACE}"/>',
            f'<text x="24" y="34" font-size="20" font-weight="600" fill="{INK}">{title}</text>',
            f'<text x="24" y="56" font-size="13" fill="{INK2}">{subtitle}</text>']
    return "\n".join(head + body + ["</svg>"])


def axis(out, X, y_top, y_bot, ticks):
    for t in ticks:
        out.append(f'<line x1="{X(t):.1f}" y1="{y_top}" x2="{X(t):.1f}" y2="{y_bot}" stroke="{GRID if t else INK2}" '
                   f'stroke-width="{1 if t else 1.5}"{"" if t else " stroke-dasharray=\"6 4\""}/>')
        out.append(f'<text x="{X(t):.1f}" y="{y_bot + 18}" text-anchor="middle" font-size="12" fill="{MUTED}">'
                   f'{f"{t:+d}%" if t else "0"}</text>')


def legend(out, x, y, items):
    for i, (color, text) in enumerate(items):
        xx = x + i * 220
        out.append(f'<circle cx="{xx + 6}" cy="{y}" r="5" fill="{color}"/>')
        out.append(f'<text x="{xx + 18}" y="{y + 4}" font-size="12" fill="{INK2}">{text}</text>')


def by_expert():
    W, H, left, right = 960, 520, 300, 120
    lo, hi = -10, 1
    X = lambda v: left + (v - lo) / (hi - lo) * (W - left - right)
    top, row = 110, 44
    out = []
    legend(out, left, 84, [(MUTED, "epoch 376"), (BLUE, "epoch 435 (+- s.e.)")])
    axis(out, X, top - 10, top + row * len(EXPERTS) - 10, list(range(-10, 2, 2)))
    for i, (e, name, share, a, b, se) in enumerate(EXPERTS):
        y = top + i * row + 10
        out.append(f'<text x="{left - 14}" y="{y + 4}" text-anchor="end" font-size="12" fill="{INK}">'
                   f'e{e} · {name}</text>')
        out.append(f'<text x="{left - 14}" y="{y + 18}" text-anchor="end" font-size="11" fill="{MUTED}">'
                   f'{share:.1f}% of training positions</text>')
        out.append(f'<line x1="{X(a):.1f}" y1="{y}" x2="{X(b):.1f}" y2="{y}" stroke="{AXIS}" stroke-width="2"/>')
        out.append(f'<line x1="{X(b - se):.1f}" y1="{y}" x2="{X(b + se):.1f}" y2="{y}" stroke="{BLUE}" stroke-width="2" opacity="0.5"/>')
        out.append(f'<circle cx="{X(a):.1f}" cy="{y}" r="5" fill="{MUTED}" stroke="{SURFACE}" stroke-width="2"/>')
        out.append(f'<circle cx="{X(b):.1f}" cy="{y}" r="6" fill="{BLUE}" stroke="{SURFACE}" stroke-width="2"/>')
        out.append(f'<text x="{X(hi) + 10:.1f}" y="{y + 4}" font-size="12" fill="{INK}">{b:+.1f}%</text>')
    out.append(f'<text x="{(left + W - right) / 2:.1f}" y="{H - 14}" text-anchor="middle" font-size="13" fill="{INK2}">'
               f't91 loss at lambda 0.75, Consilium Plebis minus Consilium, in % of Consilium (left = Consilium Plebis better)</text>')
    return svg(W, H, out, "t91 loss by expert: Consilium Plebis against Consilium",
               "expert of the side to move (opponent's non-pawn material, N=B=3, R=5, Q=9); same 2.46 M positions for every network")


def by_bucket():
    W, H, left, right = 960, 470, 300, 120
    lo, hi = -8, 1
    X = lambda v: left + (v - lo) / (hi - lo) * (W - left - right)
    top, row = 110, 44
    out = []
    legend(out, left, 84, [(ORANGE, "lambda 0.75 (training target)"), (AQUA, "lambda 0 (game result)")])
    axis(out, X, top - 10, top + row * len(BUCKETS) - 10, list(range(-8, 2, 2)))
    for i, (b, pieces, share, l75, l0) in enumerate(BUCKETS):
        y = top + i * row + 10
        out.append(f'<text x="{left - 14}" y="{y + 4}" text-anchor="end" font-size="12" fill="{INK}">b{b} · {pieces} pieces</text>')
        out.append(f'<text x="{left - 14}" y="{y + 18}" text-anchor="end" font-size="11" fill="{MUTED}">{share:.1f}% of t91 positions</text>')
        out.append(f'<line x1="{X(0):.1f}" y1="{y}" x2="{X(min(l75, l0)):.1f}" y2="{y}" stroke="{GRID}" stroke-width="2"/>')
        out.append(f'<circle cx="{X(l0):.1f}" cy="{y}" r="6" fill="{AQUA}" stroke="{SURFACE}" stroke-width="2"/>')
        out.append(f'<circle cx="{X(l75):.1f}" cy="{y}" r="6" fill="{ORANGE}" stroke="{SURFACE}" stroke-width="2"/>')
        out.append(f'<text x="{X(hi) + 10:.1f}" y="{y + 4}" font-size="12" fill="{INK}">{l75:+.1f}% / {l0:+.1f}%</text>')
    out.append(f'<text x="{(left + W - right) / 2:.1f}" y="{H - 14}" text-anchor="middle" font-size="13" fill="{INK2}">'
               f't91 loss of epoch 435 minus Consilium, in % of Consilium (left = Consilium Plebis better); s.e. about 0.3%</text>')
    return svg(W, H, out, "t91 loss by material: Consilium Plebis (epoch 435) against Consilium",
               "layer-stack bucket = (pieces - 1) / 4, kings and pawns counted; bucket 0 (up to 4 pieces) too noisy to show")


def main():
    out_dir = sys.argv[1]
    with open(os.path.join(out_dir, "c2_t91_by_expert.svg"), "w", encoding="utf-8") as f:
        f.write(by_expert())
    with open(os.path.join(out_dir, "c2_t91_by_bucket.svg"), "w", encoding="utf-8") as f:
        f.write(by_bucket())


if __name__ == "__main__":
    main()
