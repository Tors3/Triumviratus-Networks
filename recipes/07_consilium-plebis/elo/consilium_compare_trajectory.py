"""Consilium and Consilium Plebis on one chart: each network's epoch exports against the network it replaces.

Consilium (MoE-1024) against Legio Septima, at time control (20+0.2, then 30+0.3), points from consilium_elo_trajectory.py.
Consilium Plebis against the final Consilium, at fixed 100k nodes per move, rows of consilium2_elo_points.csv. Same
search within each series. The learning rate (% of its peak) is marked along the top for each run: Consilium's schedule
is the one of its P run (ticks of the September chart), Consilium Plebis's is the one-cycle of train_c2.sh (550 epochs,
+3% steps, 5% warmup, cosine), which falls more slowly.
Usage: python consilium_compare_trajectory.py [out.svg]   (PNG: msedge --headless as in the other trajectory scripts)
"""
import csv, math, sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import consilium_elo_trajectory as c1   # noqa: E402  (data of the Consilium chart)

C1_POINTS = [p for s in c1.SERIES for p in s["points"]]
C2_POINTS = [(int(r["epoch"]), float(r["elo"]), float(r["margin95"]))
             for r in csv.DictReader(open(HERE / "consilium2_elo_points.csv", encoding="utf-8"))
             if r["condition"].startswith(("fixed 100k", "8.1 dev fixed 100k"))]   # 26/10: also the 8.1 dev rows
C1_LR = c1.LR


def c2_lr(ep):
    t, tw = ep / (550 * 1.03), 0.05
    return 100 * t / tw if t < tw else 100 * (1 + math.cos(math.pi * (t - tw) / (1 - tw))) / 2


C2_LR = [(e, f"{c2_lr(e):.0f}%") for e in (28, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550)]
C1_COL, C2_COL = "#d9662b", "#178a6b"
W, H = 1200, 700
L, R, T, B = 84, 1160, 132, 600
X0, X1, Y0, Y1 = 0, 560, -250, 40   # 26/10/2026: to 560, the P run ends at 550
INK, INK2, INK3, GRID = "#1d1c1a", "#55534e", "#8a877f", "#e6e3dc"


def sx(e):
    return L + (e - X0) / (X1 - X0) * (R - L)


def sy(v):
    return B - (v - Y0) / (Y1 - Y0) * (B - T)


def text(x, y, s, size=12, color=INK2, anchor="start", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def series(points, color, shape):
    out = []
    pts = sorted(points)
    out.append('<polyline fill="none" stroke="%s" stroke-width="1.2" stroke-opacity="0.45" points="%s"/>'
               % (color, " ".join(f"{sx(e):.1f},{sy(v):.1f}" for e, v, _ in pts)))
    for e, v, m in pts:
        x, y = sx(e), sy(v)
        out.append(f'<line x1="{x:.1f}" y1="{sy(v - m):.1f}" x2="{x:.1f}" y2="{sy(v + m):.1f}" stroke="{color}" '
                   f'stroke-width="1.2" stroke-opacity="0.7"/>')
        if shape == "circle":
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"/>')
        else:
            out.append(f'<polygon points="{x:.1f},{y - 5.5:.1f} {x + 4.8:.1f},{y:.1f} {x:.1f},{y + 5.5:.1f} '
                       f'{x - 4.8:.1f},{y:.1f}" fill="{color}"/>')
    return out


def build():
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="Segoe UI, Helvetica Neue, Arial, sans-serif" role="img" aria-label="Consilium and Consilium '
         f'Plebis across training, each against the network it replaces">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         text(30, 38, "Consilium and Consilium Plebis across training", 20, INK, weight="600"),
         text(30, 60, "Elo of epoch exports against the network each one replaces: Consilium against Legio Septima "
                      "(time control), Consilium Plebis against Consilium (fixed 100k nodes)", 13, INK2)]
    for v in range(Y0, Y1 + 1, 50):
        s.append(f'<line x1="{L}" y1="{sy(v):.1f}" x2="{R}" y2="{sy(v):.1f}" stroke="{GRID}"/>')
        s.append(text(L - 10, sy(v) + 4, f"{v:+d}" if v else "0", 12, INK3, "end"))
    s.append(f'<line x1="{L}" y1="{sy(0):.1f}" x2="{R}" y2="{sy(0):.1f}" stroke="{INK2}" stroke-width="1"/>')
    for e in range(0, X1 + 1, 50):
        s.append(f'<line x1="{sx(e):.1f}" y1="{B}" x2="{sx(e):.1f}" y2="{B + 5}" stroke="{INK3}"/>')
        s.append(text(sx(e), B + 20, str(e), 12, INK3, "middle"))
    # Consilium's F3 fine-tune (from epoch 373): where its gain arrived
    s.append(f'<rect x="{sx(373):.1f}" y="{T}" width="{sx(X1) - sx(373):.1f}" height="{B - T}" fill="#e6f2ec"/>')
    s.append(text(sx(378), B - 30, "Consilium F3 fine-tune", 11, INK2))
    s.append(text(sx(378), B - 16, "(learning rate to 20%, then 0)", 11, INK3))
    s.append(f'<rect x="{L}" y="{T}" width="{R - L}" height="{B - T}" fill="none" stroke="{GRID}"/>')
    s.append(text((L + R) / 2, B + 44, "Training epoch (pretraining P, then Consilium's F3 from 373)", 13, INK2,
                  "middle"))
    s.append(text(26, (T + B) / 2, "Elo against the previous network", 13, INK2, "middle")
             .replace("<text ", f'<text transform="rotate(-90 26 {(T + B) / 2:.0f})" '))
    # learning-rate ticks along the top, one row per run
    s.append(text(L - 10, T - 34, "LR Consilium", 11, C1_COL, "end"))
    s.append(text(L - 10, T - 14, "LR Plebis", 11, C2_COL, "end"))
    for e, lab in C1_LR:
        s.append(text(sx(e), T - 34, lab, 11, C1_COL, "middle"))
    for e, lab in C2_LR:
        s.append(text(sx(e), T - 14, lab, 11, C2_COL, "middle"))
    s += series(C1_POINTS, C1_COL, "circle")
    s += series(C2_POINTS, C2_COL, "diamond")
    # Consilium's longer time controls: 60+0.6 at epoch 350 (hollow) and the F3 net of epoch 415 at 90+0.9 (star)
    lt, st = c1.LONG_TC, c1.STAR
    x, y = sx(lt["epoch"]), sy(lt["elo"])
    s.append(f'<line x1="{x:.1f}" y1="{sy(lt["elo"] - lt["margin"]):.1f}" x2="{x:.1f}" '
             f'y2="{sy(lt["elo"] + lt["margin"]):.1f}" stroke="{C1_COL}" stroke-opacity="0.7"/>')
    s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#ffffff" stroke="{C1_COL}" stroke-width="1.6"/>')
    s.append(text(x, y + 20, "60+0.6", 11, C1_COL, "middle"))
    x, y = sx(st["epoch"]), sy(st["elo"])
    s.append(f'<line x1="{x:.1f}" y1="{sy(st["elo"] - st["margin"]):.1f}" x2="{x:.1f}" '
             f'y2="{sy(st["elo"] + st["margin"]):.1f}" stroke="#c0392b" stroke-width="1.4"/>')
    pts = " ".join(f"{x + (9 if i % 2 == 0 else 4) * math.sin(i * math.pi / 5):.1f},"
                   f"{y - (9 if i % 2 == 0 else 4) * math.cos(i * math.pi / 5):.1f}" for i in range(10))
    s.append(f'<polygon points="{pts}" fill="#c0392b"/>')
    s.append(text(x - 12, y - 14, "+14.5 at 90+0.9 (F3)", 12, "#c0392b", "end", "600"))
    # legend
    lx, ly = sx(250), sy(-180)
    s.append(f'<rect x="{lx - 14:.0f}" y="{ly - 24:.0f}" width="430" height="76" rx="6" fill="#ffffff" stroke="{GRID}"/>')
    s.append(f'<circle cx="{lx:.0f}" cy="{ly - 4:.0f}" r="4.5" fill="{C1_COL}"/>')
    s.append(text(lx + 14, ly, "Consilium vs Legio Septima (20+0.2, 30+0.3)", 12, INK))
    s.append(f'<polygon points="{lx:.0f},{ly + 16:.0f} {lx + 4.8:.0f},{ly + 21.5:.0f} {lx:.0f},{ly + 27:.0f} '
             f'{lx - 4.8:.0f},{ly + 21.5:.0f}" fill="{C2_COL}"/>')
    s.append(text(lx + 14, ly + 26, "Consilium Plebis vs Consilium (fixed 100k nodes)", 12, INK))
    last = max(C2_POINTS)
    s.append(text(30, H - 34, f"Different references and conditions: read the shape, not the gap. Consilium Plebis: "
                              f"{len(C2_POINTS)} points up to epoch {last[0]}, where its learning rate is "
                              f"{c2_lr(last[0]):.0f}% of the peak.", 12, INK3))
    s.append(text(30, H - 16, "Consilium's later points (from epoch 157) use the evaluation scale calibrated per phase; "
                              "its gain over Legio Septima arrived with the F3 fine-tune (+14.5 at 90+0.9).", 12, INK3))
    s.append("</svg>")
    return "\n".join(s)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_suffix(".svg")
    out.write_text(build(), encoding="utf-8")
    print("written", out)
