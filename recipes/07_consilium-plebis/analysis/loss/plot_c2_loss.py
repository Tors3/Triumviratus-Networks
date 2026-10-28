"""Three SVG charts of the Consilium Plebis P run (23/10/2026), standard library only.

  1. train loss per epoch, whole run, log scale, with the warmup end and the lambda stop marked;
  2. the same loss from epoch 25 on a linear scale, with the LR (% of peak) and lambda below it, on the same epoch axis
     (three panels instead of a dual axis);
  3. the loss against the LR after the warmup.
LR and lambda come from the logged global step of each epoch (OneCycleLR as in the run: 2,161,025 steps, warmup 5%,
start lr/25, final lr/25000; lambda 1.0 -> 0.75 linear over 648,307 steps, then fixed).
Labels in English by default; --it writes the Italian labels and file names of the author's notes (26/10/2026).
Usage: python plot_c2_loss.py <epoch_step_loss.txt> <out_dir> [--it]"""
import math
import os
import sys

TOTAL, WARM_PCT, LAMBDA_STEPS = 2161025, 0.05, 648307
P1_END = WARM_PCT * TOTAL - 1
P2_END = TOTAL - 1

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
IT = "--it" in sys.argv


def T(it, en):
    """The label in the chosen language."""
    return it if IT else en


def lr_frac(s):
    if s <= P1_END:
        p = s / P1_END
        return 1.0 + (0.04 - 1.0) / 2 * (math.cos(math.pi * p) + 1)
    p = (s - P1_END) / (P2_END - P1_END)
    return 0.00004 + (1 - 0.00004) / 2 * (math.cos(math.pi * p) + 1)


def lam(s):
    return 1.0 - 0.25 * min(1.0, s / LAMBDA_STEPS)


def load(path):
    rows = {}
    for line in open(path):
        f = line.split()
        if len(f) == 3:
            rows[int(f[0])] = (int(f[1]), float(f[2]))
    return [(e, *rows[e]) for e in sorted(rows)]


def first_epoch_at(rows, step):
    for e, s, _ in rows:
        if s >= step:
            return e
    return rows[-1][0]


class Panel:
    """One plot area: maps data to pixels and draws axes, grid, ticks."""

    def __init__(self, x0, y0, w, h, xr, yr, log=False):
        self.x0, self.y0, self.w, self.h, self.xr, self.yr, self.log = x0, y0, w, h, xr, yr, log

    def X(self, v):
        return self.x0 + (v - self.xr[0]) / (self.xr[1] - self.xr[0]) * self.w

    def Y(self, v):
        lo, hi = self.yr
        if self.log:
            v, lo, hi = math.log10(v), math.log10(lo), math.log10(hi)
        return self.y0 + self.h - (v - lo) / (hi - lo) * self.h

    def frame(self, out, yticks, ylabel_fmt, xticks, show_x_labels=True):
        for t in yticks:
            y = self.Y(t)
            out.append(f'<line x1="{self.x0}" y1="{y:.1f}" x2="{self.x0 + self.w}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
            out.append(f'<text x="{self.x0 - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="{MUTED}" '
                       f'style="font-variant-numeric: tabular-nums">{ylabel_fmt(t)}</text>')
        yb = self.y0 + self.h
        out.append(f'<line x1="{self.x0}" y1="{yb}" x2="{self.x0 + self.w}" y2="{yb}" stroke="{AXIS}" stroke-width="1"/>')
        for t in xticks:
            x = self.X(t)
            out.append(f'<line x1="{x:.1f}" y1="{yb}" x2="{x:.1f}" y2="{yb + 4}" stroke="{AXIS}" stroke-width="1"/>')
            if show_x_labels:
                out.append(f'<text x="{x:.1f}" y="{yb + 18}" text-anchor="middle" font-size="12" fill="{MUTED}">{t}</text>')

    def line(self, out, pts, color, width=2):
        d = " ".join(f"{'M' if i == 0 else 'L'}{self.X(x):.1f},{self.Y(y):.1f}" for i, (x, y) in enumerate(pts))
        out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>')

    def vmark(self, out, x, label, anchor="start", dy=0):
        px = self.X(x)
        out.append(f'<line x1="{px:.1f}" y1="{self.y0}" x2="{px:.1f}" y2="{self.y0 + self.h}" stroke="{MUTED}" '
                   f'stroke-width="1" stroke-dasharray="4 4"/>')
        tx = px + (6 if anchor == "start" else -6)
        out.append(f'<text x="{tx:.1f}" y="{self.y0 + 14 + dy}" text-anchor="{anchor}" font-size="12" fill="{INK2}">{label}</text>')

    def shade(self, out, xa, xb):
        out.append(f'<rect x="{self.X(xa):.1f}" y="{self.y0}" width="{self.X(xb) - self.X(xa):.1f}" height="{self.h}" '
                   f'fill="#f0efec"/>')


def svg(w, h, body, title, subtitle):
    head = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="{FONT}">',
            f'<rect width="{w}" height="{h}" fill="{SURFACE}"/>',
            f'<text x="24" y="34" font-size="20" font-weight="600" fill="{INK}">{title}</text>',
            f'<text x="24" y="56" font-size="13" fill="{INK2}">{subtitle}</text>']
    return "\n".join(head + body + ["</svg>"])


def chart_epoch(rows, warm_ep, lam_ep):
    W, H = 960, 560
    last = rows[-1][0]
    p = Panel(84, 80, W - 84 - 40, H - 80 - 70, (0, last), (0.003, 0.016), log=True)
    out = []
    p.shade(out, lam_ep, last)
    yt = [0.003, 0.004, 0.005, 0.006, 0.008, 0.010, 0.012, 0.016]
    p.frame(out, yt, lambda t: f"{t:.3f}", list(range(0, last + 1, 20)))
    p.vmark(out, warm_ep, T(f"fine warmup (ep{warm_ep})", f"end of warmup (ep{warm_ep})"))
    p.vmark(out, lam_ep, T(f"lambda fisso 0,75 da ep{lam_ep + 1}", f"lambda fixed at 0.75 from ep{lam_ep + 1}"), dy=18)
    p.line(out, [(e, l) for e, _, l in rows], BLUE)
    e, _, l = rows[-1]
    out.append(f'<circle cx="{p.X(e):.1f}" cy="{p.Y(l):.1f}" r="4" fill="{BLUE}" stroke="{SURFACE}" stroke-width="2"/>')
    out.append(f'<text x="{p.X(e) - 8:.1f}" y="{p.Y(l) - 12:.1f}" text-anchor="end" font-size="12" fill="{INK}">ep{e} {l:.5f}</text>')
    out.append(f'<text x="{p.x0 + p.w / 2:.1f}" y="{H - 22}" text-anchor="middle" font-size="13" fill="{INK2}">{T("epoca", "epoch")}</text>')
    out.append(f'<text x="22" y="{p.y0 + p.h / 2:.1f}" transform="rotate(-90 22 {p.y0 + p.h / 2:.1f})" text-anchor="middle" '
               f'font-size="13" fill="{INK2}">{T("loss di training (scala log)", "training loss (log scale)")}</text>')
    return svg(W, H, out, T("Consilium Plebis P: loss di training per epoca", "Consilium Plebis P: training loss per epoch"),
               T(f"epoche 0-{last}, un punto per epoca; zona grigia = lambda fermo a 0,75",
                 f"epochs 0-{last}, one point per epoch; grey area = lambda fixed at 0.75"))


def chart_panels(rows, warm_ep, lam_ep, start=25):
    W, H = 960, 760
    last = rows[-1][0]
    sel = [r for r in rows if r[0] >= start]
    xr = (start, last)
    lo = min(l for _, _, l in sel)
    hi = max(l for _, _, l in sel)
    pad = (hi - lo) * 0.08
    left, width = 84, W - 84 - 40
    pl = Panel(left, 80, width, 300, xr, (lo - pad, hi + pad))
    lr_lo = math.floor((min(lr_frac(s) for _, s, _ in sel) - 0.03) * 20) / 20
    pr = Panel(left, 430, width, 120, xr, (lr_lo, 1.02))
    pm = Panel(left, 600, width, 100, xr, (0.74, 1.0))
    out = []
    for pan in (pl, pr, pm):
        pan.shade(out, lam_ep, last)
    xt = [t for t in range(0, last + 1, 20) if t >= start]
    step = 0.00005
    yt = [round(math.ceil((lo - pad) / step) * step + i * step, 6) for i in range(30)]
    yt = [t for t in yt if t <= hi + pad]
    pl.frame(out, yt, lambda t: f"{t:.5f}", xt, show_x_labels=False)
    lr_ticks = [t / 10 for t in range(10, 0, -1) if t / 10 >= lr_lo]
    pr.frame(out, lr_ticks, lambda t: f"{t * 100:.0f}%", xt, show_x_labels=False)
    pm.frame(out, [0.75, 0.80, 0.90, 1.00], lambda t: f"{t:.2f}", xt)
    pl.vmark(out, lam_ep, T(f"lambda fisso da ep{lam_ep + 1}", f"lambda fixed from ep{lam_ep + 1}"))
    pl.line(out, [(e, l) for e, _, l in sel], BLUE)
    pr.line(out, [(e, lr_frac(s)) for e, s, _ in sel], ORANGE)
    pm.line(out, [(e, lam(s)) for e, s, _ in sel], AQUA)
    e, s, l = sel[-1]
    out.append(f'<circle cx="{pl.X(e):.1f}" cy="{pl.Y(l):.1f}" r="4" fill="{BLUE}" stroke="{SURFACE}" stroke-width="2"/>')
    out.append(f'<text x="{pl.X(e) - 8:.1f}" y="{pl.Y(l) + 18:.1f}" text-anchor="end" font-size="12" fill="{INK}">ep{e} {l:.5f}</text>')
    out.append(f'<text x="{pr.X(e) - 8:.1f}" y="{pr.Y(lr_frac(s)) - 24:.1f}" text-anchor="end" font-size="12" fill="{INK}">'
               f'{lr_frac(s) * 100:.0f}{T("% del picco", "% of peak")}</text>')
    for pan, name in ((pl, T("loss di training", "training loss")), (pr, T("LR (% del picco)", "LR (% of peak)")),
                      (pm, "lambda")):
        out.append(f'<text x="{pan.x0}" y="{pan.y0 - 8}" font-size="13" font-weight="600" fill="{INK2}">{name}</text>')
    out.append(f'<text x="{pm.x0 + pm.w / 2:.1f}" y="{H - 22}" text-anchor="middle" font-size="13" fill="{INK2}">{T("epoca", "epoch")}</text>')
    return svg(W, H, out, T(f"Consilium Plebis P: loss, LR e lambda dall'epoca {start}",
                            f"Consilium Plebis P: loss, LR and lambda from epoch {start}"),
               T("stessa scala delle epoche nei tre pannelli; zona grigia = lambda fermo a 0,75",
                 "same epoch axis in the three panels; grey area = lambda fixed at 0.75"))


def chart_lr(rows, warm_ep, lam_ep):
    """Loss against the LR after the warmup (LR only falls there), split by lambda phase."""
    W, H = 960, 560
    sel = [(lr_frac(s) * 100, l, e) for e, s, l in rows if e >= warm_ep]
    lo = min(l for _, l, _ in sel)
    hi = max(l for _, l, _ in sel)
    pad = (hi - lo) * 0.08
    p = Panel(84, 96, W - 84 - 40, H - 96 - 70, (100.5, min(x for x, _, _ in sel) - 1.5), (lo - pad, hi + pad))
    out = []
    step = 0.00005
    yt = [round(math.ceil((lo - pad) / step) * step + i * step, 6) for i in range(30)]
    yt = [t for t in yt if t <= hi + pad]
    xt = [t for t in range(100, 0, -5) if t >= p.xr[1]]
    p.frame(out, yt, lambda t: f"{t:.5f}", xt)
    moving = [(x, l) for x, l, e in sel if e <= lam_ep]
    fixed = [(x, l) for x, l, e in sel if e >= lam_ep]
    p.line(out, moving, BLUE)
    p.line(out, fixed, ORANGE)
    # legend (two series) plus direct labels at the ends
    out.append(f'<line x1="{p.x0}" y1="76" x2="{p.x0 + 22}" y2="76" stroke="{BLUE}" stroke-width="2"/>')
    out.append(f'<text x="{p.x0 + 28}" y="80" font-size="12" fill="{INK2}">'
               f'{T("lambda in discesa", "lambda falling")} (ep{warm_ep}-{lam_ep})</text>')
    out.append(f'<line x1="{p.x0 + 250}" y1="76" x2="{p.x0 + 272}" y2="76" stroke="{ORANGE}" stroke-width="2"/>')
    out.append(f'<text x="{p.x0 + 278}" y="80" font-size="12" fill="{INK2}">'
               f'{T("lambda fisso 0,75", "lambda fixed at 0.75")} (ep{lam_ep + 1}-{sel[-1][2]})</text>')
    x, l, e = sel[-1]
    out.append(f'<circle cx="{p.X(x):.1f}" cy="{p.Y(l):.1f}" r="4" fill="{ORANGE}" stroke="{SURFACE}" stroke-width="2"/>')
    out.append(f'<text x="{p.X(x) - 8:.1f}" y="{p.Y(l) + 18:.1f}" text-anchor="end" font-size="12" fill="{INK}">ep{e}, LR {x:.0f}%</text>')
    x0, l0, e0 = sel[0]
    out.append(f'<text x="{p.X(x0) + 8:.1f}" y="{p.Y(l0) + 4:.1f}" font-size="12" fill="{INK}">ep{e0}</text>')
    out.append(f'<text x="{p.x0 + p.w / 2:.1f}" y="{H - 22}" text-anchor="middle" font-size="13" fill="{INK2}">'
               f'{T("LR (% del picco), scende verso destra", "LR (% of peak), falling to the right")}</text>')
    out.append(f'<text x="22" y="{p.y0 + p.h / 2:.1f}" transform="rotate(-90 22 {p.y0 + p.h / 2:.1f})" text-anchor="middle" '
               f'font-size="13" fill="{INK2}">{T("loss di training", "training loss")}</text>')
    return svg(W, H, out, T("Consilium Plebis P: loss di training contro LR", "Consilium Plebis P: training loss against LR"),
               T(f"dalla fine del warmup (ep{warm_ep}) a oggi; l'LR scende sempre, quindi ogni punto è un momento diverso",
                 f"from the end of the warmup (ep{warm_ep}) on; the LR only falls, so every point is a later moment"))


def main():
    args = [a for a in sys.argv[1:] if a != "--it"]
    rows = load(args[0])
    out_dir = args[1]
    warm_ep = first_epoch_at(rows, P1_END)
    lam_ep = first_epoch_at(rows, LAMBDA_STEPS)
    with open(os.path.join(out_dir, T("c2_loss_epoca.svg", "c2_loss_per_epoch.svg")), "w", encoding="utf-8") as f:
        f.write(chart_epoch(rows, warm_ep, lam_ep))
    with open(os.path.join(out_dir, "c2_loss_lr_lambda.svg"), "w", encoding="utf-8") as f:
        f.write(chart_panels(rows, warm_ep, lam_ep))
    with open(os.path.join(out_dir, "c2_loss_vs_lr.svg"), "w", encoding="utf-8") as f:
        f.write(chart_lr(rows, warm_ep, lam_ep))
    e, s, l = rows[-1]
    print(f"epochs {rows[0][0]}-{e}, warmup ends ep{warm_ep}, lambda fixed from ep{lam_ep + 1}, "
          f"last LR {lr_frac(s) * 100:.1f}%, lambda {lam(s):.3f}")


if __name__ == "__main__":
    main()
