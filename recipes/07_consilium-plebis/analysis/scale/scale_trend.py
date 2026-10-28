"""Trend of the evaluation scale of Consilium Plebis against Consilium over the P exports (25/10/2026).

Reads every scale_epNN.txt written by scale_c2.py (search evaluation, "final" section) and writes
  scale_trend.csv   epoch, scale (sd ratio), OLS C2~C1, 1/OLS C1~C2, offset, r, per-expert and per-bucket sd ratios;
  scale_trend.svg   two panels on the same epoch axis: the scale with the OLS bracket as a band, and the offset.
Labels in English by default; --it writes the Italian labels of the author's notes (26/10/2026).
Usage: python scale_trend.py <scale_dir> <out_dir> [--it]
"""
import glob
import os
import re
import sys

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, BLUE_BAND, ORANGE = "#2a78d6", "#cde2fb", "#eb6834"
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
IT = "--it" in sys.argv


def T(it, en):
    """The label in the chosen language."""
    return it if IT else en


def parse(path):
    rows, sec, part = {}, None, None
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("===== final"):
            sec = "final"
        elif line.startswith("===== raw"):
            sec = "raw"
        if sec != "final":
            continue
        if line.startswith("by layer-stack"):
            part = "b"
        elif line.startswith("by C2 expert"):
            part = "e"
        f = line.split()
        if len(f) == 8 and re.fullmatch(r"[0-9.]+", f[2] or ""):
            key = "all" if f[0] == "all" else f"{part}{f[0]}"
            rows[key] = dict(ols=float(f[2]), inv=float(f[3]), sd=float(f[4]), off=float(f[6]), r=float(f[7]))
    return rows


def main():
    src, out_dir = [a for a in sys.argv[1:] if a != "--it"][:2]
    data = []
    for p in glob.glob(os.path.join(src, "scale_ep*.txt")):
        m = re.search(r"scale_ep(\d+)\.txt$", p)
        if not m:
            continue
        rows = parse(p)
        if "all" in rows:
            data.append((int(m.group(1)), rows))
    data.sort()
    cols = ["epoch", "scale", "ols", "inv", "offset", "r"] + [f"e{i}" for i in range(8)] + [f"b{i}" for i in range(8)]
    with open(os.path.join(out_dir, "scale_trend.csv"), "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for ep, r in data:
            a = r["all"]
            vals = [ep, a["sd"], a["ols"], a["inv"], a["off"], a["r"]]
            vals += [r.get(f"e{i}", {}).get("sd", "") for i in range(8)] + [r.get(f"b{i}", {}).get("sd", "") for i in range(8)]
            f.write(",".join(str(v) for v in vals) + "\n")

    W, H, left, width = 960, 700, 84, 960 - 84 - 40
    x0, x1 = 0, max(ep for ep, _ in data) + 10
    X = lambda e: left + (e - x0) / (x1 - x0) * width
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">',
           f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
           f'<text x="24" y="34" font-size="20" font-weight="600" fill="{INK}">'
           f'{T("Consilium Plebis: scala della valutazione rispetto a Consilium", "Consilium Plebis: evaluation scale against Consilium")}</text>',
           f'<text x="24" y="56" font-size="13" fill="{INK2}">'
           + T("sd(C2)/sd(C1) sulla valutazione della ricerca, 49.594 posizioni; "
               "banda = forbice OLS C2~C1 / inversa di C1~C2; 1,00 = stessa scala",
               "sd(C2)/sd(C1) of the search evaluation, 49,594 positions; "
               "band = OLS C2~C1 to inverse OLS C1~C2; 1.00 = same scale") + '</text>']

    def panel(y0, h, lo, hi, ticks, fmt, title):
        Y = lambda v: y0 + h - (v - lo) / (hi - lo) * h
        out.append(f'<text x="{left}" y="{y0 - 10}" font-size="13" font-weight="600" fill="{INK2}">{title}</text>')
        for t in ticks:
            out.append(f'<line x1="{left}" y1="{Y(t):.1f}" x2="{left + width}" y2="{Y(t):.1f}" stroke="{GRID}"/>')
            out.append(f'<text x="{left - 8}" y="{Y(t) + 4:.1f}" text-anchor="end" font-size="12" fill="{MUTED}">{fmt(t)}</text>')
        out.append(f'<line x1="{left}" y1="{y0 + h}" x2="{left + width}" y2="{y0 + h}" stroke="{AXIS}"/>')
        return Y

    Ys = panel(100, 330, 0.68, 1.04, [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00], lambda t: f"{t:.2f}",
               T("scala (sd C2 / sd C1)", "scale (sd C2 / sd C1)"))
    band_top = " ".join(f"{X(e):.1f},{Ys(r['all']['inv']):.1f}" for e, r in data)
    band_bot = " ".join(f"{X(e):.1f},{Ys(r['all']['ols']):.1f}" for e, r in reversed(data))
    out.append(f'<polygon points="{band_top} {band_bot}" fill="{BLUE_BAND}" opacity="0.8"/>')
    out.append(f'<line x1="{left}" y1="{Ys(1.0):.1f}" x2="{left + width}" y2="{Ys(1.0):.1f}" stroke="{MUTED}" stroke-dasharray="4 4"/>')
    d = " ".join(f"{'M' if i == 0 else 'L'}{X(e):.1f},{Ys(r['all']['sd']):.1f}" for i, (e, r) in enumerate(data))
    out.append(f'<path d="{d}" fill="none" stroke="{BLUE}" stroke-width="2" stroke-linejoin="round"/>')
    for e, r in data:
        out.append(f'<circle cx="{X(e):.1f}" cy="{Ys(r["all"]["sd"]):.1f}" r="4" fill="{BLUE}" stroke="{SURFACE}" stroke-width="2"/>')
    e, r = data[-1]
    out.append(f'<text x="{X(e) - 8:.1f}" y="{Ys(r["all"]["sd"]) + 20:.1f}" text-anchor="end" font-size="12" fill="{INK}">ep{e} {r["all"]["sd"]:.3f}</text>')

    lo_off = min(-5, min(r["all"]["off"] for _, r in data) - 3)
    hi_off = max(r["all"]["off"] for _, r in data) + 3
    step = 5
    oticks = [t for t in range(int(lo_off // step * step), int(hi_off) + step, step) if lo_off <= t <= hi_off]
    Yo = panel(490, 150, lo_off, hi_off, oticks, lambda t: f"{t:+d}" if t else "0",
               T("offset per chi muove (C2 meno C1, unità del motore)", "side-to-move offset (C2 minus C1, engine units)"))
    d = " ".join(f"{'M' if i == 0 else 'L'}{X(e):.1f},{Yo(r['all']['off']):.1f}" for i, (e, r) in enumerate(data))
    out.append(f'<path d="{d}" fill="none" stroke="{ORANGE}" stroke-width="2" stroke-linejoin="round"/>')
    for e, r in data:
        out.append(f'<circle cx="{X(e):.1f}" cy="{Yo(r["all"]["off"]):.1f}" r="3.5" fill="{ORANGE}" stroke="{SURFACE}" stroke-width="1.5"/>')
    for t in range(0, x1 + 1, 50):
        out.append(f'<text x="{X(t):.1f}" y="{490 + 150 + 18}" text-anchor="middle" font-size="12" fill="{MUTED}">{t}</text>')
    out.append(f'<text x="{left + width / 2:.1f}" y="{H - 14}" text-anchor="middle" font-size="13" fill="{INK2}">{T("epoca", "epoch")}</text>')
    out.append("</svg>")
    with open(os.path.join(out_dir, "scale_trend.svg"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    for ep, r in data:
        a = r["all"]
        print(f"ep{ep:<4} scale {a['sd']:.3f}  [{a['ols']:.3f}-{a['inv']:.3f}]  off {a['off']:+6.1f}  r {a['r']:.3f}")


if __name__ == "__main__":
    main()
