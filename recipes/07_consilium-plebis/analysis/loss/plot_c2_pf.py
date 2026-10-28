"""Consilium Plebis P + F on one epoch axis (27-28/10/2026), standard library only; English labels, --it for Italian.

  1. c2_pf_overall.svg: training loss of the whole run (log scale), the zoom from epoch 350 on a linear scale, and the
     LR as % of P's peak below it, same epoch axis (F epoch k = global epoch 497 + k);
  2. c2_pf_loss_vs_lr.svg: loss against the LR for P (after the warmup) and F (after its 2-epoch rise);
  3. c2_pf_zoom.svg: the end of F and its continuation to a zero LR, with P's and F's final losses as references.
LR of P: OneCycleLR as in plot_c2_loss.py; LR of F: NNUE_LR_PIECES of start_F.sh and start_fr3.sh times F_LR_FRAC 0.25.
Every epoch's LR is taken at the middle of the epoch (the loss is the epoch mean).
Usage: python plot_c2_pf.py <c2_epoch_step_loss.txt> <F metrics.csv>... <out_dir> [--it]
  (the F metrics: run_logs/metrics/F_version_0.csv F_version_2.csv FR3_version_2.csv)"""
import csv
import math
import os
import sys

IT = "--it" in sys.argv
from plot_c2_loss import Panel, svg, lr_frac, load, SURFACE, INK, INK2, MUTED, BLUE, ORANGE, AQUA  # noqa: E402


def T(it, en):
    return it if IT else en


EPOCH_STEPS = 3814
# 27/10 ~08:25 UTC: the flat 15% stretch was cut from 20 to 10 epochs (F resumed from the step-epoch 32 checkpoint);
# 27/10 15:20 UTC: a second cycle from F's final step and a restart at 0.5% of P's peak, both discarded (not drawn);
# 27/10 22:39 UTC: continuation from F's final step without rise, 3 epochs of half-cosine from F's final LR
# (0.004% of P's peak) to zero; the final network is the average of these 3 epochs (5 more epochs at 0.0003% were
# stopped in their first epoch and are not drawn).
F_PIECES = [(7628, 0.1494, 1, "lin"), (87722, 1, 0.6, "cos"), (38140, 0.6, 0.6, "lin"), (95350, 0.6, 0.00016, "cos"),
            (11442, 0.00016, 0, "cos")]
F_FRAC = 0.25
C2_START = 60          # first F epoch of the continuation
F_EPOCHS = 63
PART2 = T("continuazione", "continuation")


def f_lr(step):
    """F's LR as a fraction of P's peak."""
    for n, a, b, shape in F_PIECES:
        if step < n:
            t = step / n
            if shape == "cos":
                t = (1 - math.cos(math.pi * t)) / 2
            return F_FRAC * (a + (b - a) * t)
        step -= n
    return F_FRAC * F_PIECES[-1][2]


def load_f(*paths):
    """F epochs numbered by optimizer step: a resume repeats the epoch label of its checkpoint while the steps go on."""
    rows = {}
    for path in paths:
        for r in csv.DictReader(open(path)):
            if r["epoch"] and r["train_loss_epoch"]:
                s = int(r["step"])
                rows[s // EPOCH_STEPS - 1] = (s // EPOCH_STEPS - 1, s, float(r["train_loss_epoch"]))
    return [rows[k] for k in sorted(rows)]


def dot(out, p, x, y, color, r=4):
    out.append(f'<circle cx="{p.X(x):.1f}" cy="{p.Y(y):.1f}" r="{r}" fill="{color}" stroke="{SURFACE}" stroke-width="2"/>')


def text(out, x, y, s, anchor="start", color=INK, size=12, weight=None):
    w = f' font-weight="{weight}"' if weight else ""
    out.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}" fill="{color}"{w}>{s}</text>')


def tag(out, px, py, right, base, s):
    """A label right-aligned at `right` on the baseline `base`, with a leader line down to the point (px, py)."""
    lx = min(max(px, right - len(s) * 5.2 + 10), right - 10)   # about 5.2 px per character at 12 px
    out.append(f'<line x1="{lx:.1f}" y1="{base + 4:.1f}" x2="{px:.1f}" y2="{py - 6:.1f}" stroke="{MUTED}" '
               f'stroke-width="1"/>')
    text(out, right, base, s, anchor="end")


def fmt(v, d=5):
    s = f"{v:.{d}f}"
    return s.replace(".", ",") if IT else s


def legend(out, x, y):
    out.append(f'<line x1="{x}" y1="{y}" x2="{x + 22}" y2="{y}" stroke="{BLUE}" stroke-width="2"/>')
    text(out, x + 28, y + 4, T("P (ep0-496)", "P (epochs 0-496)"), color=INK2)
    out.append(f'<line x1="{x + 150}" y1="{y}" x2="{x + 172}" y2="{y}" stroke="{ORANGE}" stroke-width="2"/>')
    text(out, x + 178, y + 4, T("F (dalla rete P di ep496)", "F (from P's epoch-496 weights)"), color=INK2)
    out.append(f'<line x1="{x + 400}" y1="{y}" x2="{x + 422}" y2="{y}" stroke="{AQUA}" stroke-width="2"/>')
    text(out, x + 428, y + 4, T(f"F {PART2} (dalla fine di F, ep{C2_START}-{F_EPOCHS - 1})",
                                f"F {PART2} (from F's end, epochs {C2_START}-{F_EPOCHS - 1})"), color=INK2)


def chart_all(prow, frow, start=350):
    W, H = 960, 980
    p_last = prow[-1][0]
    g0 = p_last + 1                      # global epoch of F epoch 0
    end = g0 + F_EPOCHS - 1
    pts_p = [(e, l) for e, _, l in prow]
    pts_f1 = [(g0 + e, l) for e, _, l in frow if e < C2_START]
    pts_f2 = [(g0 + e, l) for e, _, l in frow if e >= C2_START]
    pts_f = pts_f1 + pts_f2
    left, width = 84, W - 84 - 40
    out = []

    # panel A: whole run, log scale
    pa = Panel(left, 110, width, 220, (0, end), (0.003, 0.016), log=True)
    pa.shade(out, g0 - 0.5, end)
    pa.frame(out, [0.003, 0.004, 0.006, 0.008, 0.012, 0.016], lambda t: fmt(t, 3), list(range(0, end + 1, 50)))
    pa.line(out, pts_p, BLUE)
    pa.line(out, [pts_p[-1]] + pts_f1, ORANGE)
    if pts_f2:
        pa.line(out, [pts_f1[-1]] + pts_f2, AQUA)
    text(out, pa.x0, pa.y0 - 10, T("loss di training, tutta la corsa (scala log)", "training loss, the whole run (log scale)"),
         color=INK2, size=13, weight=600)

    # panel B: zoom from `start`, linear
    sel = [(e, l) for e, l in pts_p if e >= start] + pts_f
    lo = min(l for _, l in sel)
    hi = max(l for _, l in sel)
    pad = (hi - lo) * 0.08
    pb = Panel(left, 420, width, 330, (start, end), (lo - pad, hi + pad))
    pc = Panel(left, 800, width, 110, (start, end), (0, 0.55))
    for pan in (pb, pc):
        pan.shade(out, g0 - 0.5, end)
    xt = list(range(start, end + 1, 25))
    step = 0.00002
    yt = [round(math.ceil((lo - pad) / step) * step + i * step, 6) for i in range(40)]
    yt = [t for t in yt if t <= hi + pad]
    pb.frame(out, yt, lambda t: fmt(t), xt, show_x_labels=False)
    pc.frame(out, [0, 0.1, 0.2, 0.3, 0.4, 0.5], lambda t: f"{t * 100:.0f}%", xt)

    # P final loss as the reference F starts from
    pe, pl = pts_p[-1]
    yb = pb.Y(pl)
    out.append(f'<line x1="{pb.x0}" y1="{yb:.1f}" x2="{pb.x0 + pb.w}" y2="{yb:.1f}" stroke="{MUTED}" stroke-width="1" '
               f'stroke-dasharray="4 4"/>')
    text(out, pb.x0 + 6, yb - 6, T(f"P finale (ep{pe}, LR 3,7%): {fmt(pl, 6)}", f"P's end (epoch {pe}, LR 3.7%): {fmt(pl, 6)}"),
         color=INK2)
    pb.line(out, [(e, l) for e, l in pts_p if e >= start], BLUE)
    pb.line(out, [pts_p[-1]] + pts_f1, ORANGE)
    dot(out, pb, pe, pl, BLUE)
    fe, fl = pts_f1[-1]
    if pts_f2:
        yf = pb.Y(fl)
        out.append(f'<line x1="{pb.X(fe):.1f}" y1="{yf:.1f}" x2="{pb.x0 + pb.w}" y2="{yf:.1f}" stroke="{MUTED}" '
                   f'stroke-width="1" stroke-dasharray="4 4"/>')
        pb.line(out, [pts_f1[-1]] + pts_f2, AQUA)
        ce, cl = pts_f2[-1]
        dot(out, pb, ce, cl, AQUA)
        top = max(l for _, l in pts_f2)
        tag(out, pb.X(ce), pb.Y(cl), pb.x0 + pb.w, min(pb.Y(0.00315), pb.Y(top) - 30),
            T(f"{PART2} ep{ce - g0}: {fmt(cl, 6)}", f"{PART2}, epoch {ce - g0}: {fmt(cl, 6)}"))
    dot(out, pb, fe, fl, ORANGE)
    text(out, pb.X(fe) - 10, pb.Y(fl) + 20, T(f"fine di F, ep{fe - g0}: {fmt(fl, 6)}", f"F's end, epoch {fe - g0}: {fmt(fl, 6)}"),
         anchor="end")
    fmax = max(pts_f1, key=lambda t: t[1])
    text(out, pb.X(fmax[0]) + 8, pb.Y(fmax[1]) - 8,
         T(f"picco F ep{fmax[0] - g0} (LR 25%): {fmt(fmax[1], 6)}", f"F's peak, epoch {fmax[0] - g0} (LR 25%): {fmt(fmax[1], 6)}"))
    text(out, pb.x0, pb.y0 - 10, T(f"loss di training dall'epoca {start} (scala lineare)",
                                   f"training loss from epoch {start} (linear scale)"), color=INK2, size=13, weight=600)

    # panel C: LR, a planned part dashed while the run goes on
    pc.line(out, [(e, lr_frac(s - EPOCH_STEPS // 2)) for e, s, _ in prow if e >= start], BLUE)
    done = len(frow) * EPOCH_STEPS
    c2 = C2_START * EPOCH_STEPS
    planned = False
    for a, b, color in ((0, c2, ORANGE), (c2, F_EPOCHS * EPOCH_STEPS, AQUA)):
        seg = [(g0 - 0.5 + s / EPOCH_STEPS, f_lr(s)) for s in range(a, b + 1, 200)]
        pc.line(out, [(x, y) for x, y in seg if x <= g0 - 0.5 + done / EPOCH_STEPS], color)
        plan = [(x, y) for x, y in seg if x >= g0 - 0.5 + done / EPOCH_STEPS]
        if len(plan) > 1:
            planned = True
            d = " ".join(f"{'M' if i == 0 else 'L'}{pc.X(x):.1f},{pc.Y(y):.1f}" for i, (x, y) in enumerate(plan))
            out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="5 4"/>')
    for k, lab in ((1, "25%"), (29, T("15% fisso", "15% flat"))):
        x = g0 + k
        text(out, pc.X(x), pc.Y(f_lr((k + 1) * EPOCH_STEPS)) - 8, lab, anchor="middle", color=INK2)
    if planned:
        text(out, pc.X(end), pc.Y(0.12), T("previsto", "planned"), anchor="end", color=INK2)
    text(out, pc.x0, pc.y0 - 10, T("LR (% del picco di P)", "LR (% of P's peak)"), color=INK2, size=13, weight=600)
    text(out, pc.x0 + pc.w / 2, H - 28, T("epoca (F ep k = epoca 497 + k)", "epoch (F's epoch k = epoch 497 + k)"),
         anchor="middle", color=INK2, size=13)

    legend(out, left, 76)
    return svg(W, H, out, T("Consilium Plebis: P e F, loss e LR", "Consilium Plebis: P and F, loss and LR"),
               T(f"P epoche 0-{p_last}, F epoche 0-{frow[-1][0]}; zona grigia = F; stessa scala delle epoche nei pannelli "
                 f"in basso", f"P epochs 0-{p_last}, F epochs 0-{frow[-1][0]}; grey area = F; the lower panels share the "
                 f"epoch axis"))


def chart_zoom(prow, frow, start=50):
    """F from epoch `start` and its continuation on F's epoch axis: loss with P's and F's final values, LR below."""
    W, H = 960, 700
    end = F_EPOCHS - 1
    pts1 = [(e, l) for e, _, l in frow if start <= e < C2_START]
    pts2 = [(e, l) for e, _, l in frow if e >= C2_START]
    pl, fl = prow[-1][2], pts1[-1][1]
    allv = [l for _, l in pts1 + pts2] + [pl]
    lo, hi = min(allv), max(allv)
    pad = (hi - lo) * 0.12
    left, width = 84, W - 84 - 40
    pb = Panel(left, 125, width, 370, (start - 0.5, end + 0.5), (lo - pad, hi + pad))
    pc = Panel(left, 560, width, 80, (start - 0.5, end + 0.5), (0, 0.06))
    out = []
    for pan in (pb, pc):
        pan.shade(out, C2_START - 0.5, end + 0.5)
    xt = list(range(start, end + 1, 2))
    step = 0.00001
    yt = [round(math.ceil((lo - pad) / step) * step + i * step, 6) for i in range(40)]
    yt = [t for t in yt if t <= hi + pad]
    pb.frame(out, yt, lambda t: fmt(t), xt, show_x_labels=False)
    pc.frame(out, [0, 0.02, 0.04, 0.06], lambda t: f"{t * 100:.0f}%", xt)
    for v, lab in ((pl, T(f"P finale (ep496, LR 3,7%): {fmt(pl, 6)}", f"P's end (epoch 496, LR 3.7%): {fmt(pl, 6)}")),
                   (fl, T(f"fine di F (ep59): {fmt(fl, 6)}", f"F's end (epoch 59): {fmt(fl, 6)}"))):
        y = pb.Y(v)
        out.append(f'<line x1="{pb.x0}" y1="{y:.1f}" x2="{pb.x0 + pb.w}" y2="{y:.1f}" stroke="{MUTED}" stroke-width="1" '
                   f'stroke-dasharray="4 4"/>')
        text(out, pb.x0 + 6, y - 6, lab, color=INK2)
    pb.line(out, pts1, ORANGE)
    for x, l in pts1:
        dot(out, pb, x, l, ORANGE, r=3)
    if pts2:
        pb.line(out, [pts1[-1]] + pts2, AQUA)
        for x, l in pts2:
            dot(out, pb, x, l, AQUA, r=3)
        x, l = pts2[-1]
        dot(out, pb, x, l, AQUA)
        lab = T(f"{PART2} ep{x}: {fmt(l, 6)}", f"{PART2}, epoch {x}: {fmt(l, 6)}")
        if pb.X(x) + 8 + len(lab) * 5.6 > pb.x0 + pb.w:   # no room on the right: below-left of the point
            text(out, pb.X(x) - 10, pb.Y(l) + 22, lab, anchor="end")
        else:
            text(out, pb.X(x) + 8, pb.Y(l) - 8, lab)
    text(out, pb.x0, pb.y0 - 10, T(f"loss di training da F ep{start} (scala lineare)",
                                   f"training loss from F's epoch {start} (linear scale)"), color=INK2, size=13, weight=600)
    done = len(frow) * EPOCH_STEPS
    c2 = C2_START * EPOCH_STEPS
    for a, b, color in ((start * EPOCH_STEPS, c2, ORANGE), (c2, F_EPOCHS * EPOCH_STEPS, AQUA)):
        seg = [(-0.5 + s / EPOCH_STEPS, f_lr(s)) for s in range(a, b + 1, 100)]
        lim = -0.5 + done / EPOCH_STEPS
        pc.line(out, [(x, y) for x, y in seg if x <= lim], color)
        plan = [(x, y) for x, y in seg if x >= lim]
        if len(plan) > 1:
            d = " ".join(f"{'M' if i == 0 else 'L'}{pc.X(x):.1f},{pc.Y(y):.1f}" for i, (x, y) in enumerate(plan))
            out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="5 4"/>')
    text(out, pc.X(C2_START + 1), pc.Y(0) - 8, T("0,004% fino a 0", "0.004% down to 0"), anchor="middle", color=INK2)
    text(out, pc.x0, pc.y0 - 10, T("LR (% del picco di P)", "LR (% of P's peak)"), color=INK2, size=13, weight=600)
    text(out, pc.x0 + pc.w / 2, H - 22, T(f"epoca di F (la {PART2} parte da ep{C2_START}, dalla fine di F)",
                                          f"F's epoch (the {PART2} starts at epoch {C2_START}, from F's end)"),
         anchor="middle", color=INK2, size=13)
    out.append(f'<line x1="{left}" y1="76" x2="{left + 22}" y2="76" stroke="{ORANGE}" stroke-width="2"/>')
    text(out, left + 28, 80, "F", color=INK2)
    out.append(f'<line x1="{left + 70}" y1="76" x2="{left + 92}" y2="76" stroke="{AQUA}" stroke-width="2"/>')
    text(out, left + 98, 80, T(f"F {PART2} (da 0,004% a 0 in 3 epoche; la rete finale è la media delle 3)",
                               f"F {PART2} (0.004% to 0 in 3 epochs; the final network is the average of the 3)"),
         color=INK2)
    return svg(W, H, out, T(f"Consilium Plebis: fine di F e {PART2}", f"Consilium Plebis: F's end and its {PART2}"),
               T(f"F ep{start}-59 e {PART2} ep{C2_START}-{frow[-1][0]}; zona grigia = {PART2}",
                 f"F epochs {start}-59 and {PART2} epochs {C2_START}-{frow[-1][0]}; grey area = {PART2}"))


def chart_lr(prow, frow):
    W, H = 960, 600
    warm_ep = math.ceil(0.05 * 2161025 / EPOCH_STEPS)   # the LR only falls after the warmup
    pts_p = [(lr_frac(s - EPOCH_STEPS // 2) * 100, l, e) for e, s, l in prow
             if e > warm_ep and lr_frac(s - EPOCH_STEPS // 2) <= 0.30]
    pts_f = [(f_lr(s - EPOCH_STEPS // 2) * 100, l, e) for e, s, l in frow if e >= 2]
    allv = [l for _, l, _ in pts_p + pts_f]
    lo, hi = min(allv), max(allv)
    pad = (hi - lo) * 0.08
    p = Panel(84, 100, W - 84 - 40, H - 100 - 76, (30, 0), (lo - pad, hi + pad))
    out = []
    step = 0.00002
    yt = [round(math.ceil((lo - pad) / step) * step + i * step, 6) for i in range(40)]
    yt = [t for t in yt if t <= hi + pad]
    p.frame(out, yt, lambda t: fmt(t), list(range(30, -1, -5)))
    p.line(out, [(x, l) for x, l, _ in pts_p], BLUE)
    p.line(out, [(x, l) for x, l, e in pts_f if e <= 24], ORANGE)
    flat = [(x, l, e) for x, l, e in pts_f if 25 <= e <= 34]
    for x, l, _ in flat:
        dot(out, p, x, l, ORANGE, r=3)
    down = [(x, l, e) for x, l, e in pts_f if 34 <= e < C2_START]
    if len(down) > 1:
        p.line(out, [(x, l) for x, l, _ in down], ORANGE)
        x, l, e = down[-1]
        dot(out, p, x, l, ORANGE)
        text(out, p.X(x) - 10, p.Y(l) + 24, T(f"F ep{e}: {fmt(l, 6)}", f"F, epoch {e}: {fmt(l, 6)}"), anchor="end")
    cyc2 = [(x, l, e) for x, l, e in pts_f if e >= C2_START]
    if cyc2:
        p.line(out, [(down[-1][0], down[-1][1])] + [(x, l) for x, l, _ in cyc2], AQUA)
        for x, l, _ in cyc2:
            dot(out, p, x, l, AQUA, r=3)
        x, l, e = cyc2[-1]
        dot(out, p, x, l, AQUA)
        top = max(l for _, l, _ in cyc2)
        tag(out, p.X(x), p.Y(l), p.X(0), min(p.Y(0.00313), p.Y(top) - 24),
            T(f"{PART2} ep{e}, LR {fmt(x, 2)}%: {fmt(l, 6)}", f"{PART2}, epoch {e}, LR {fmt(x, 2)}%: {fmt(l, 6)}"))
    if flat:
        m = sum(l for _, l, _ in flat) / len(flat)
        ly = lo + (hi - lo) * 0.12
        out.append(f'<line x1="{p.X(21):.1f}" y1="{p.Y(ly) - 14:.1f}" x2="{p.X(15) - 4:.1f}" y2="{p.Y(m):.1f}" '
                   f'stroke="{MUTED}" stroke-width="1"/>')
        text(out, p.X(21), p.Y(ly), T(f"F al 15% fisso, ep{flat[0][2]}-{flat[-1][2]}: media {fmt(m, 6)}",
                                      f"F at a flat 15%, epochs {flat[0][2]}-{flat[-1][2]}: mean {fmt(m, 6)}"),
             anchor="middle")
    x, l, e = pts_p[-1]
    dot(out, p, x, l, BLUE)
    plab = T(f"P ep{e}, LR {fmt(x, 1)}%", f"P, epoch {e}, LR {fmt(x, 1)}%")
    if cyc2:   # the continuation's dots sit right of P's last one: its label goes below the F line, with a leader
        by = p.Y(lo + (hi - lo) * 0.06)
        out.append(f'<line x1="{p.X(x) - 4:.1f}" y1="{by - 13:.1f}" x2="{p.X(x):.1f}" y2="{p.Y(l) + 6:.1f}" '
                   f'stroke="{MUTED}" stroke-width="1"/>')
        text(out, p.X(x), by, plab, anchor="end")
    else:
        text(out, p.X(x) + 8, p.Y(l) + 4, plab)
    x, l, e = pts_p[0]
    text(out, p.X(x) + 8, p.Y(l) - 6, T(f"P ep{e}", f"P, epoch {e}"), color=INK2)
    x, l, e = pts_f[0]
    text(out, p.X(x) - 6, p.Y(l) + 18, T(f"F ep{e}", f"F, epoch {e}"), anchor="end", color=INK2)
    legend(out, p.x0, 76)
    text(out, p.x0 + p.w / 2, H - 28, T("LR (% del picco di P), scende verso destra", "LR (% of P's peak), falling to the right"),
         anchor="middle", color=INK2, size=13)
    out.append(f'<text x="22" y="{p.y0 + p.h / 2:.1f}" transform="rotate(-90 22 {p.y0 + p.h / 2:.1f})" '
               f'text-anchor="middle" font-size="13" fill="{INK2}">{T("loss di training", "training loss")}</text>')
    return svg(W, H, out, T("Consilium Plebis: loss contro LR, P e F", "Consilium Plebis: loss against LR, P and F"),
               T("LR al centro di ogni epoca; P da LR 30% alla fine (ep496), F dopo la salita delle prime 2 epoche",
                 "LR at the middle of each epoch; P from an LR of 30% to its end (epoch 496), F after its 2-epoch rise"))


def main():
    args = [a for a in sys.argv[1:] if a != "--it"]
    prow = load(args[0])
    frow = load_f(*args[1:-1])   # F metrics.csv of every version (each resume writes a new version)
    out_dir = args[-1]
    names = T(("c2_pf_complessivo.svg", "c2_pf_loss_vs_lr.svg", "c2_pf_zoom.svg"),
              ("c2_pf_overall.svg", "c2_pf_loss_vs_lr.svg", "c2_pf_zoom.svg"))
    for name, chart in zip(names, (chart_all, chart_lr, chart_zoom)):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            f.write(chart(prow, frow))
    print(f"P 0-{prow[-1][0]}, F 0-{frow[-1][0]}, F last {frow[-1][2]:.6f}")


if __name__ == "__main__":
    main()
