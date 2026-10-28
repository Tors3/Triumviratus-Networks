"""Consilium Plebis as a transposed copy of Consilium's training curve: fit and chart.

Consilium's P points (epochs up to 373, no F3; against Legio Septima) get a weighted polynomial in ln(epoch) (degree
--deg, default 3). Consilium Plebis's points (against Consilium, fixed 100k nodes) are then matched to that curve by
  shift   Plebis(e) = C(e + s)                (best s on a grid; the model that fits best so far)
  offset  Plebis(e) = C(e) + D                (weighted mean of the residuals)
Prints both models (weighted RMS in sigmas, epoch where Plebis crosses 0) and writes an SVG chart with the points, the
Consilium fit, both transposed curves and the crossing. Different references (Legio Septima at time control, Consilium
at fixed nodes): the comparison is of the shape of the growth.
Usage: python consilium_transpose_fit.py [--deg 3] [out.svg]   (PNG: msedge --headless, window 1200x700)
"""
import csv, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import consilium_elo_trajectory as c1   # noqa: E402

deg = int(sys.argv[sys.argv.index("--deg") + 1]) if "--deg" in sys.argv else 3
args = [a for i, a in enumerate(sys.argv[1:], 1) if a != "--deg" and sys.argv[i - 1] != "--deg"]
P1 = sorted(p for s in c1.SERIES for p in s["points"] if p[0] <= 373)
P2 = sorted((int(r["epoch"]), float(r["elo"]), float(r["margin95"]))
            for r in csv.DictReader(open(HERE / "consilium2_elo_points.csv", encoding="utf-8"))
            if r["condition"].startswith("fixed 100k"))


def arr(P):
    e, v, m = map(np.array, zip(*P))
    return e.astype(float), v, 1 / (m / 1.96) ** 2


e1, v1, w1 = arr(P1)
e2, v2, w2 = arr(P2)
coef = np.polyfit(np.log(e1), v1, deg, w=np.sqrt(w1))


def C(e):
    return float(np.polyval(coef, math.log(e)))


def zero(f, lo=20, hi=3000):
    return next((e for e in range(lo, hi) if f(e) >= 0), None)


def rms(pred):
    r = (v2 - np.array([pred(e) for e in e2])) * np.sqrt(w2)
    return math.sqrt(float(r @ r) / max(len(e2) - 1, 1))


shift = min(((s, rms(lambda e, s=s: C(e + s))) for s in range(-100, 301) if e2.min() + s >= 15), key=lambda t: t[1])
S = shift[0]
D = float(np.sum(w2 * (v2 - np.array([C(e) for e in e2]))) / np.sum(w2))
z_shift, z_off, z_c = zero(lambda e: C(e + S)), zero(lambda e: C(e) + D), zero(C)
# the same shift fitted on the recent exports only (from RECENT on): Plebis's lambda reached 0.75 at epoch 165,
# Consilium's only at 332, so the late points may run ahead of a shift fitted on the whole curve
RECENT = 107
mr = e2 >= RECENT


def rms_recent(pred):
    r = (v2[mr] - np.array([pred(e) for e in e2[mr]])) * np.sqrt(w2[mr])
    return math.sqrt(float(r @ r) / max(int(mr.sum()) - 1, 1))


shift_r = min(((s, rms_recent(lambda e, s=s: C(e + s))) for s in range(-100, 301)), key=lambda t: t[1])
S2 = shift_r[0]
z_shift2 = zero(lambda e: C(e + S2))
F3_POINTS = [(406, 1, 10)]   # Consilium F3 export at 30+0.3 (consilium_elo_trajectory.py)
F3_STAR = (c1.STAR["epoch"], c1.STAR["elo"], c1.STAR["margin"])
print(f"Consilium fit: degree {deg} in ln(epoch), {len(e1)} points, crosses 0 at epoch {z_c}")
print(f"shift  s = {S:+d}: rms {shift[1]:.2f}, Consilium Plebis crosses 0 at epoch {z_shift}")
print(f"shift from epoch {RECENT} ({int(mr.sum())} points) s = {S2:+d}: rms {shift_r[1]:.2f}, crosses 0 at epoch {z_shift2}")
print(f"offset D = {D:+.1f}: rms {rms(lambda e: C(e) + D):.2f}, Consilium Plebis crosses 0 at epoch {z_off}")

# ---- chart -----------------------------------------------------------------------------------------------------------
W, H = 1200, 700
L, R, T, B = 84, 1160, 100, 600
X0, X1, Y0, Y1 = 0, 450, -250, 40
C1C, C2C, INK, INK2, INK3, GRID = "#d9662b", "#178a6b", "#1d1c1a", "#55534e", "#8a877f", "#e6e3dc"
sx = lambda e: L + (e - X0) / (X1 - X0) * (R - L)
sy = lambda v: B - (v - Y0) / (Y1 - Y0) * (B - T)


def text(x, y, s, size=12, color=INK2, anchor="start", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def curve(f, a, b, color, width=2.0, dash=""):
    pts = [(e, f(e)) for e in np.linspace(a, b, 220)]
    pts = [(e, v) for e, v in pts if Y0 <= v <= Y1]
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<polyline fill="none" stroke="{color}" stroke-width="{width}"{d} points="'
            + " ".join(f"{sx(e):.1f},{sy(v):.1f}" for e, v in pts) + '"/>')


def points(P, color, shape):
    out = []
    for e, v, m in P:
        x, y = sx(e), sy(v)
        out.append(f'<line x1="{x:.1f}" y1="{sy(max(v - m, Y0)):.1f}" x2="{x:.1f}" y2="{sy(min(v + m, Y1)):.1f}" '
                   f'stroke="{color}" stroke-opacity="0.6"/>')
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{color}"/>' if shape == "c" else
                   f'<polygon points="{x:.1f},{y - 5.5:.1f} {x + 4.8:.1f},{y:.1f} {x:.1f},{y + 5.5:.1f} '
                   f'{x - 4.8:.1f},{y:.1f}" fill="{color}"/>')
    return out


s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
     f'font-family="Segoe UI, Helvetica Neue, Arial, sans-serif" role="img" aria-label="Consilium Plebis as Consilium '
     f'shifted by {S2} epochs (fit on the exports from epoch {RECENT}): projected parity at epoch {z_shift2}">',
     f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
     text(30, 38, "Consilium Plebis as a transposed Consilium curve", 20, INK, weight="600"),
     text(30, 60, f"Consilium (P, against Legio Septima) fitted with a degree-{deg} polynomial in ln(epoch); "
                  f"Consilium Plebis (against Consilium, fixed 100k nodes) matched to it", 13, INK2)]
for v in range(Y0, Y1 + 1, 50):
    s.append(f'<line x1="{L}" y1="{sy(v):.1f}" x2="{R}" y2="{sy(v):.1f}" stroke="{GRID}"/>')
    s.append(text(L - 10, sy(v) + 4, f"{v:+d}" if v else "0", 12, INK3, "end"))
s.append(f'<line x1="{L}" y1="{sy(0):.1f}" x2="{R}" y2="{sy(0):.1f}" stroke="{INK2}"/>')
for e in range(0, X1 + 1, 50):
    s.append(text(sx(e), B + 20, str(e), 12, INK3, "middle"))
s.append(f'<rect x="{L}" y="{T}" width="{R - L}" height="{B - T}" fill="none" stroke="{GRID}"/>')
# Consilium's F3 fine-tune (from epoch 373) and the level it reached
s.insert(4, f'<rect x="{sx(373):.1f}" y="{T}" width="{R - sx(373):.1f}" height="{B - T}" fill="#eef5f0"/>')
s.append(text(sx(373) + 8, B - 30, "Consilium F3 fine-tune", 12, INK2, weight="600"))
s.append(text(sx(373) + 8, B - 14, "(from epoch 373)", 11, INK3))
RED = "#c0392b"
s.append(f'<line x1="{L}" y1="{sy(F3_STAR[1]):.1f}" x2="{R}" y2="{sy(F3_STAR[1]):.1f}" stroke="{RED}" '
         f'stroke-width="1.2" stroke-dasharray="6 4"/>')
s.append(text(L + 10, sy(F3_STAR[1]) - 7, f"Consilium's final level after F3: +{F3_STAR[1]} against Legio Septima "
                                          f"(90+0.9, epoch {F3_STAR[0]})", 12, RED, weight="600"))
s.append(text((L + R) / 2, B + 44, "Training epoch (pretraining P)", 13, INK2, "middle"))
s.append(text(26, (T + B) / 2, "Elo against the previous network", 13, INK2, "middle")
         .replace("<text ", f'<text transform="rotate(-90 26 {(T + B) / 2:.0f})" '))
s.append(curve(C, 17, 373, C1C, 2.2))
s.append(curve(lambda e: C(e + S2), RECENT, X1, C2C, 2.2, "7 5"))
s += points(P1, C1C, "c")
s += points(F3_POINTS, C1C, "c")
s += points(P2, C2C, "d")
x, y = sx(F3_STAR[0]), sy(F3_STAR[1])
s.append(f'<line x1="{x:.1f}" y1="{sy(F3_STAR[1] - F3_STAR[2]):.1f}" x2="{x:.1f}" y2="{sy(min(F3_STAR[1] + F3_STAR[2], Y1)):.1f}" '
         f'stroke="{RED}" stroke-opacity="0.6"/>')
star = " ".join(f"{x + (7 if i % 2 == 0 else 3) * math.sin(i * math.pi / 5):.1f},"
                f"{y - (7 if i % 2 == 0 else 3) * math.cos(i * math.pi / 5):.1f}" for i in range(10))
s.append(f'<polygon points="{star}" fill="{RED}"/>')
for z, dy, lab in ((z_shift2, 40, f"projected parity: epoch {z_shift2}"),):
    if z:
        x = sx(z)
        s.append(f'<line x1="{x:.1f}" y1="{sy(0) - 30:.1f}" x2="{x:.1f}" y2="{sy(0) + 30:.1f}" stroke="{C2C}" '
                 f'stroke-width="1.5"/>')
        s.append(text(x, sy(0) + dy + (4 if dy > 0 else 0), lab, 12.5, C2C, "middle", "600"))
lx, ly = sx(220), sy(-150)
s.append(f'<rect x="{lx - 14:.0f}" y="{ly - 24:.0f}" width="560" height="112" rx="6" fill="#ffffff" stroke="{GRID}"/>')
rows = [(f'<line x1="{lx - 6:.0f}" y1="{ly - 4:.0f}" x2="{lx + 10:.0f}" y2="{ly - 4:.0f}" stroke="{C1C}" stroke-width="2.2"/>',
         f"Consilium points and its fit (crosses 0 at epoch {z_c})"),
        (f'<line x1="{lx - 6:.0f}" y1="{ly + 22:.0f}" x2="{lx + 10:.0f}" y2="{ly + 22:.0f}" stroke="{C2C}" stroke-width="2.2" stroke-dasharray="7 5"/>',
         f"Plebis = Consilium {S2:+d} epochs ahead (fit from epoch {RECENT}): parity at epoch {z_shift2}"),
        (f'<line x1="{lx - 6:.0f}" y1="{ly + 48:.0f}" x2="{lx + 10:.0f}" y2="{ly + 48:.0f}" stroke="{RED}" stroke-width="1.2" stroke-dasharray="6 4"/>',
         f"Consilium after its F3 fine-tune: +{F3_STAR[1]} (star; circle in the band: epoch 406 at 30+0.3)")]
for i, (mk, lab) in enumerate(rows):
    s.append(mk)
    s.append(text(lx + 18, ly + i * 26, lab, 12, INK))
s.append(text(lx + 18, ly + 78, "diamonds: Consilium Plebis exports; circles: Consilium exports", 12, INK3))
s.append(text(30, H - 16, "Different references (Legio Septima at time control, Consilium at fixed nodes): the shape of "
                          "the growth is compared, not the level. Consilium's gain above parity came in its F3 fine-tune.",
              12, INK3))
s.append("</svg>")
out = Path(args[0]) if args else Path(__file__).with_suffix(".svg")
out.write_text("\n".join(s), encoding="utf-8")
print("written", out)
