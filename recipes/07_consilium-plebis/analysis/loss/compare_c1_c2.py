"""Consilium (C1) against Consilium Plebis (C2): training curves and the fixed-lambda holdout (25/10/2026).

  1. c1_c2_training.svg: train loss, lambda and LR (% of each run's peak) of both P runs on the same epoch axis
     (1 G positions per epoch in both); three panels, no dual axis.
  2. c1_c2_holdout.svg: loss on the t91 holdout of Consilium Plebis exports relative to the Consilium release, on the
     same positions, at lambda 1.0, 0.75 and 0 (26/10/2026: paired data replace the VM sample of 21/10).
C1 P: batch 524288, 858,306 steps, warmup 5%, final div 1000, lambda 1.0 with a cycle of -0.3 (cosine down to 0.70 over
the first 25% of the steps, then cosine back up); logged to epoch 291. C2: see plot_c2_loss.py.
Usage: python compare_c1_c2.py <c1_epoch_step_loss.txt> <c2_epoch_step_loss.txt> <out_dir>
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_c2_loss as p  # noqa: E402

C1_TOTAL, C1_WARM = 858306, 0.05
C1_P1 = C1_WARM * C1_TOTAL - 1
C1_P2 = C1_TOTAL - 1


def c1_lr(s):
    if s <= C1_P1:
        q = s / C1_P1
        return 1.0 + (0.04 - 1.0) / 2 * (math.cos(math.pi * q) + 1)
    q = (s - C1_P1) / (C1_P2 - C1_P1)
    return 0.00004 + (1 - 0.00004) / 2 * (math.cos(math.pi * q) + 1)


def c1_lam(s):
    warm = 0.25 * C1_TOTAL
    if s < warm:
        c = 0.5 * (1 - math.cos(math.pi * s / warm))
    elif s < C1_TOTAL:
        c = 0.5 * (1 + math.cos(math.pi * (s - warm) / (C1_TOTAL - warm)))
    else:
        c = 0.0
    return 1.0 - 0.3 * c


# Holdout t91 (x1e-3) at lambda 1.0 / 0.75 / 0.0, local runs of c2_val_loss_fixed2.py: every network reads the same
# positions (checked 25/10: Consilium and Consilium Plebis streams equal; ep246 repeats the 24/10 values exactly), so
# the first 120 batches of the 25/10 run pair with the 120-batch run of 24/10 (docs/rete/c2_holdout_loss_2026-10-24.txt).
# The VM values of 21/10 (Consilium 5.240, ep9-22) came from another sample and are left out.
C1_T91 = (4.3494, 5.2455, 18.0830)
C2_T91 = [(166, 4.4476, 5.3896, 18.4485), (181, 4.3996, 5.3234, 18.2928), (194, 4.3618, 5.2958, 18.3200),
          (201, 4.3767, 5.3099, 18.3340), (210, 4.3355, 5.2625, 18.2548), (246, 4.3050, 5.2405, 18.2714),
          (376, 4.2522, 5.1536, 18.0276), (435, 4.1942, 5.0901, 17.9331)]


def legend(out, x, y, items):
    for i, (color, text) in enumerate(items):
        xx = x + i * 300
        out.append(f'<line x1="{xx}" y1="{y}" x2="{xx + 22}" y2="{y}" stroke="{color}" stroke-width="2"/>')
        out.append(f'<text x="{xx + 28}" y="{y + 4}" font-size="12" fill="{p.INK2}">{text}</text>')


def training(c1, c2):
    W, H, left = 960, 860, 84
    width = W - left - 40
    start, last = 25, max(c2[-1][0], c1[-1][0])
    xr = (start, last)
    s1 = [r for r in c1 if r[0] >= start]
    s2 = [r for r in c2 if r[0] >= start]
    lo = min(l for _, _, l in s1 + s2)
    hi = max(l for _, _, l in s1 + s2)
    pad = (hi - lo) * 0.06
    pl = p.Panel(left, 110, width, 330, xr, (lo - pad, hi + pad))
    pm = p.Panel(left, 500, width, 120, xr, (0.66, 1.02))
    pr = p.Panel(left, 680, width, 110, xr, (0.0, 1.05))
    out = []
    legend(out, left, 84, [(p.BLUE, "Consilium Plebis (C2)"),
                           (p.ORANGE, p.T("Consilium (C1), log fino a ep291", "Consilium (C1), log up to ep291"))])
    xt = [t for t in range(0, last + 1, 25) if t >= start]
    step = 0.0002
    yt = [round(math.ceil((lo - pad) / step) * step + i * step, 6) for i in range(30)]
    yt = [t for t in yt if t <= hi + pad]
    pl.frame(out, yt, lambda t: f"{t:.4f}", xt, show_x_labels=False)
    pm.frame(out, [0.70, 0.75, 0.80, 0.90, 1.00], lambda t: f"{t:.2f}", xt, show_x_labels=False)
    pr.frame(out, [0.0, 0.25, 0.5, 0.75, 1.0], lambda t: f"{t * 100:.0f}%", xt)
    pl.line(out, [(e, l) for e, _, l in s1], p.ORANGE)
    pl.line(out, [(e, l) for e, _, l in s2], p.BLUE)
    pm.line(out, [(e, c1_lam(s)) for e, s, _ in s1], p.ORANGE)
    pm.line(out, [(e, p.lam(s)) for e, s, _ in s2], p.BLUE)
    pr.line(out, [(e, c1_lr(s)) for e, s, _ in s1], p.ORANGE)
    pr.line(out, [(e, p.lr_frac(s)) for e, s, _ in s2], p.BLUE)
    for rows, color, dy in ((s2, p.BLUE, -10), (s1, p.ORANGE, 18)):
        e, _, l = rows[-1]
        out.append(f'<circle cx="{pl.X(e):.1f}" cy="{pl.Y(l):.1f}" r="4" fill="{color}" stroke="{p.SURFACE}" stroke-width="2"/>')
        out.append(f'<text x="{pl.X(e) - 8:.1f}" y="{pl.Y(l) + dy:.1f}" text-anchor="end" font-size="12" fill="{p.INK}">ep{e} {l:.5f}</text>')
    e, s, _ = s1[-1]
    out.append(f'<text x="{pm.X(e) + 6:.1f}" y="{pm.Y(c1_lam(s)) + 4:.1f}" font-size="12" fill="{p.INK}">{c1_lam(s):.2f}</text>')
    for pan, name in ((pl, p.T("loss di training (dipende da lambda: non è lo stesso bersaglio)",
                               "training loss (depends on lambda: not the same target)")), (pm, "lambda"),
                      (pr, p.T("LR (% del picco di ciascun run)", "LR (% of each run's peak)"))):
        out.append(f'<text x="{pan.x0}" y="{pan.y0 - 8}" font-size="13" font-weight="600" fill="{p.INK2}">{name}</text>')
    out.append(f'<text x="{left + width / 2:.1f}" y="{H - 22}" text-anchor="middle" font-size="13" fill="{p.INK2}">'
               f'{p.T("epoca (1 miliardo di posizioni per epoca in entrambi i run)", "epoch (1 billion positions per epoch in both runs)")}</text>')
    return p.svg(W, H, out, p.T("Consilium e Consilium Plebis: fase P a confronto",
                                "Consilium and Consilium Plebis: the P phases compared"),
                 p.T("dall'epoca 25; lambda diversi rendono le loss di training non confrontabili in valore assoluto",
                     "from epoch 25; different lambdas make the training losses not comparable in absolute value"))


def holdout():
    """Loss on t91 relative to Consilium (same positions), one line per lambda; 0 = Consilium."""
    W, H, left = 960, 560, 84
    width = W - left - 170
    series = [(0, p.BLUE, p.T("lambda 1,0 (valutazione)", "lambda 1.0 (evaluation)")),
              (1, p.ORANGE, p.T("lambda 0,75 (bersaglio del training)", "lambda 0.75 (training target)")),
              (2, p.AQUA, p.T("lambda 0 (risultato)", "lambda 0 (game result)"))]
    pct = {k: [(e, (v[k] / C1_T91[k] - 1) * 100) for e, *v in C2_T91] for k, _, _ in series}
    pan = p.Panel(left, 110, width, H - 110 - 70, (150, 450), (-4.2, 3.0))
    out = []
    for i, (_, color, text) in enumerate(series):
        xx = left + i * 260
        out.append(f'<line x1="{xx}" y1="84" x2="{xx + 22}" y2="84" stroke="{color}" stroke-width="2"/>')
        out.append(f'<text x="{xx + 28}" y="88" font-size="12" fill="{p.INK2}">{text}</text>')
    pan.frame(out, [-4, -3, -2, -1, 0, 1, 2, 3], lambda t: f"{t:+.0f}%" if t else "0", list(range(150, 451, 50)))
    y0 = pan.Y(0)
    out.append(f'<line x1="{left}" y1="{y0:.1f}" x2="{left + width}" y2="{y0:.1f}" stroke="{p.INK2}" stroke-width="1.5" stroke-dasharray="6 4"/>')
    # label at the right end, above the line, where the curves are already below zero
    out.append(f'<text x="{left + width - 4}" y="{y0 - 8:.1f}" text-anchor="end" font-size="12" fill="{p.INK}">'
               f'{p.T("Consilium (rete rilasciata)", "Consilium (released network)")}</text>')
    for k, color, _ in series:
        pan.line(out, pct[k], color)
        for e, v in pct[k]:
            out.append(f'<circle cx="{pan.X(e):.1f}" cy="{pan.Y(v):.1f}" r="4" fill="{color}" stroke="{p.SURFACE}" stroke-width="2"/>')
    # direct labels at the right end, spread so they do not overlap
    ends = sorted(((pct[k][-1][1], text) for k, _, text in series), reverse=True)
    last_y = None
    for v, text in ends:
        y = pan.Y(v) + 4
        if last_y is not None and y - last_y < 16:
            y = last_y + 16
        last_y = y
        out.append(f'<text x="{pan.X(C2_T91[-1][0]) + 10:.1f}" y="{y:.1f}" font-size="12" fill="{p.INK}">'
                   f'{v:+.1f}% {text.split(" (")[1][:-1]}</text>')
    out.append(f'<text x="{left + width / 2:.1f}" y="{H - 22}" text-anchor="middle" font-size="13" fill="{p.INK2}">'
               f'{p.T("epoca di P", "P epoch")}</text>')
    out.append(f'<text x="22" y="{pan.y0 + pan.h / 2:.1f}" transform="rotate(-90 22 {pan.y0 + pan.h / 2:.1f})" '
               f'text-anchor="middle" font-size="13" fill="{p.INK2}">{p.T("loss su t91 rispetto a Consilium", "t91 loss relative to Consilium")}</text>')
    return p.svg(W, H, out, p.T("Loss su t91 rispetto a Consilium, sulle stesse posizioni",
                                "Loss on t91 relative to Consilium, on the same positions"),
                 p.T("sotto lo zero Consilium Plebis sbaglia meno; 120 lotti da 8192 posizioni, errore appaiato circa +-0,2%",
                     "below zero Consilium Plebis errs less; 120 batches of 8192 positions, paired error about +-0.2%"))


def main():
    args = [a for a in sys.argv[1:] if a != "--it"]
    c1 = p.load(args[0])
    c2 = p.load(args[1])
    out_dir = args[2]
    with open(os.path.join(out_dir, "c1_c2_training.svg"), "w", encoding="utf-8") as f:
        f.write(training(c1, c2))
    with open(os.path.join(out_dir, "c1_c2_holdout.svg"), "w", encoding="utf-8") as f:
        f.write(holdout())
    for e in (50, 100, 150, 200, 250, 291):
        r1 = next(r for r in c1 if r[0] == e)
        r2 = next(r for r in c2 if r[0] == e)
        print(f"ep{e}: C1 loss {r1[2]:.5f} lambda {c1_lam(r1[1]):.3f} LR {c1_lr(r1[1]) * 100:.0f}% | "
              f"C2 loss {r2[2]:.5f} lambda {p.lam(r2[1]):.3f} LR {p.lr_frac(r2[1]) * 100:.0f}%")


if __name__ == "__main__":
    main()
