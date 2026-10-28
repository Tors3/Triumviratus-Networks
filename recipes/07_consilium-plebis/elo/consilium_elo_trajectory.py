"""Elo trajectory of a network across training, drawn as a self-contained SVG (no plotting library).

Consilium (MoE-1024, pretraining P stopped after epoch 373, then the F3 fine-tune) against Legio Septima, same 8.0 search, only the
network changes. Values read from the September chart (match_moe/traiettoria_moe1024_v11.png, 29/09/2026 18:00);
error bars are the 95% margins of that chart, read to about +-2 Elo. A second network (Consilium Plebis) can be added as
another entry of SERIES with its own colour.

Usage: python consilium_elo_trajectory.py [out.svg]    (default: consilium_elo_trajectory.svg next to this file)
PNG:   msedge --headless --screenshot=out.png --window-size=1200,720 file:///<path>/consilium_elo_trajectory.svg
"""
import sys
from pathlib import Path

# ---- data: (epoch, elo, 95% margin) ------------------------------------------------------------------------------
SERIES = [
    {"label": "20+0.2", "color": "#2a78d6", "marker": "circle", "points": [
        (17, -230, 50), (21, -164, 45), (26, -144, 30), (30, -121, 21), (35, -113, 11), (44, -86, 11)]},
    {"label": "30+0.3, 1,000-2,000 games each", "color": "#eb6834", "marker": "triangle", "points": [
        (53, -76, 11), (64, -56, 14), (71, -70, 14), (80, -56, 11), (99, -44, 14), (108, -58, 11),
        (124, -40, 10), (134, -41, 12), (145, -36, 15)]},
    {"label": "30+0.3, eval scale calibrated per phase", "color": "#4a3aa7", "marker": "diamond", "points": [
        (157, -31, 13), (165, -24, 16), (171, -28, 10), (200, -18, 8), (253, -5, 11), (262, 5, 10), (272, -9, 14),
        (279, -10, 12), (292, -2, 12), (298, -10, 12), (310, -3, 10), (320, 6, 8), (332, 12, 10), (342, -3, 15),
        (359, 9, 10), (368, 3, 10), (406, 1, 10)]},
]
LONG_TC = {"epoch": 350, "elo": -11, "margin": 15, "label": "60+0.6"}
STAR = {"epoch": 415, "elo": 14.5, "margin": 12.5,
        "label": ["+14.5 at 90+0.9", "720 games, LOS 98.8%"]}
TREND = {"from": 53, "to": 340, "slope": 0.38, "zero": 240,
         "label": "Linear trend of the 30+0.3 points: +3.8 Elo per 10 epochs, level at about epoch 240"}
GATE = {"epoch": 225, "elo": -30, "label": "gate at epoch 225: -30"}
LR = [(24, "100%"), (80, "95%"), (135, "84%"), (180, "70%"), (225, "54%"), (270, "38%"), (314, "23%"), (359, "10%")]
BANDS = [  # (from, to, fill, label lines, label anchor)
    (22, 108, "#fbf3e4", ["learning rate above 90% of its peak"], "start"),
    (373, 450, "#e6f2ec", ["F3 fine-tune (from epoch 373):", "learning rate back to 20%, then to zero"], "start"),
]
LINES = [(292, "lambda 0.865 to 0.75"), (332, "lambda fixed")]
NOTES = ["Epochs 406 and 415 played with the SPSA vector of the search; pentanomial margins from epoch 200."]

TITLE = "Consilium against Legio Septima across training"
SUBTITLE = ("Elo of epoch exports, same 8.0 search, only the network changes. Pretraining P stopped after epoch 373; "
            "F3 epochs counted from there")
XLABEL = "Training epoch (P, then F3 from epoch 373)"
YLABEL = "Elo against Legio Septima"
NOTES_AT = (150, -148)                  # (epoch, elo) of the first note line
LEGEND_AT = (222, -178)                 # (epoch, elo) of the legend's top-left corner
LEGEND_TITLE = "Match time control"

# ---- layout -------------------------------------------------------------------------------------------------------
W, H = 1200, 690
L, R, T, B = 84, 1160, 92, 600          # plot box
X0, X1, Y0, Y1 = 0, 450, -260, 60       # data ranges
INK, INK2, INK3, GRID, SURF = "#1d1c1a", "#55534e", "#8a877f", "#e6e3dc", "#ffffff"
FONT = "Segoe UI, Helvetica Neue, Arial, sans-serif"


def sx(e):
    return L + (e - X0) / (X1 - X0) * (R - L)


def sy(v):
    return B - (v - Y0) / (Y1 - Y0) * (B - T)


def text(x, y, s, size=12, color=INK2, anchor="start", weight="normal", italic=False):
    st = ' font-style="italic"' if italic else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{st}>{s}</text>')


def marker(kind, x, y, color, r=5.5, hollow=False):
    fill = SURF if hollow else color
    if kind == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{color}" stroke-width="1.6"/>'
    if kind == "triangle":
        p = f"{x:.1f},{y - r * 1.15:.1f} {x - r:.1f},{y + r * 0.75:.1f} {x + r:.1f},{y + r * 0.75:.1f}"
        return f'<polygon points="{p}" fill="{fill}" stroke="{color}" stroke-width="1.2"/>'
    if kind == "diamond":
        p = f"{x:.1f},{y - r * 1.2:.1f} {x + r:.1f},{y:.1f} {x:.1f},{y + r * 1.2:.1f} {x - r:.1f},{y:.1f}"
        return f'<polygon points="{p}" fill="{fill}" stroke="{color}" stroke-width="1.2"/>'
    if kind == "star":
        import math
        pts = []
        for i in range(10):
            a = -math.pi / 2 + i * math.pi / 5
            rr = r * 1.6 if i % 2 == 0 else r * 0.7
            pts.append(f"{x + rr * math.cos(a):.1f},{y + rr * math.sin(a):.1f}")
        return f'<polygon points="{" ".join(pts)}" fill="{fill}" stroke="{color}" stroke-width="1"/>'
    raise ValueError(kind)


def errbar(x, v, m, color):
    y_lo, y_hi = sy(max(v - m, Y0)), sy(min(v + m, Y1))
    out = [f'<line x1="{x:.1f}" y1="{y_lo:.1f}" x2="{x:.1f}" y2="{y_hi:.1f}" stroke="{color}" stroke-width="1.4"/>']
    for yy, v_end in ((y_lo, v - m), (y_hi, v + m)):
        if Y0 <= v_end <= Y1:
            out.append(f'<line x1="{x - 4:.1f}" y1="{yy:.1f}" x2="{x + 4:.1f}" y2="{yy:.1f}" stroke="{color}" '
                       f'stroke-width="1.4"/>')
    return out


def build():
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="{FONT}" role="img" aria-label="{TITLE}: {SUBTITLE}">',
         f'<rect width="{W}" height="{H}" fill="{SURF}"/>']
    # bands
    for a, b, fill, lines, anchor in BANDS:
        o.append(f'<rect x="{sx(a):.1f}" y="{T}" width="{sx(min(b, X1)) - sx(a):.1f}" height="{B - T}" fill="{fill}"/>')
        for i, s in enumerate(lines):
            o.append(text(sx(a) + 8, T + 18 + 15 * i, s, 11.5, INK2))
    # grid and y axis
    for v in range(-(-Y0 // 50) * 50, Y1 + 1, 50):
        o.append(f'<line x1="{L}" y1="{sy(v):.1f}" x2="{R}" y2="{sy(v):.1f}" stroke="{GRID}" stroke-width="1"/>')
        o.append(text(L - 10, sy(v) + 4, f"{v:+d}" if v else "0", 11.5, INK3, "end"))
    o.append(f'<line x1="{L}" y1="{sy(0):.1f}" x2="{R}" y2="{sy(0):.1f}" stroke="{INK3}" stroke-width="1.2"/>')
    # x axis
    o.append(f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="{INK3}" stroke-width="1"/>')
    for e in range(0, X1 + 1, 50):
        o.append(f'<line x1="{sx(e):.1f}" y1="{B}" x2="{sx(e):.1f}" y2="{B + 5}" stroke="{INK3}" stroke-width="1"/>')
        o.append(text(sx(e), B + 19, str(e), 11.5, INK3, "middle"))
    o.append(text((L + R) / 2, B + (70 if LR else 48), XLABEL, 13, INK2, "middle"))
    o.append(f'<text x="{L - 54}" y="{(T + B) / 2:.1f}" font-size="13" fill="{INK2}" text-anchor="middle" '
             f'transform="rotate(-90 {L - 54} {(T + B) / 2:.1f})">{YLABEL}</text>')
    # learning-rate row
    if LR:
        o.append(text(L - 10, B + 42, "lr", 11, INK3, "end", italic=True))
    for e, s in LR:
        o.append(text(sx(e), B + 42, s, 11, "#9a7a3c", "middle"))
    # vertical markers (lambda)
    for e, s in LINES:
        o.append(f'<line x1="{sx(e):.1f}" y1="{T}" x2="{sx(e):.1f}" y2="{B}" stroke="#7d74c8" stroke-width="1.2" '
                 f'stroke-dasharray="3 4"/>')
        o.append(text(sx(e) - 6, T + 16, s, 11.5, "#5b52a8", "end"))
    # trend
    if TREND:
        t0, t1 = TREND["from"], TREND["to"]
        f = lambda e: TREND["slope"] * (e - TREND["zero"])
        o.append(f'<line x1="{sx(t0):.1f}" y1="{sy(f(t0)):.1f}" x2="{sx(t1):.1f}" y2="{sy(f(t1)):.1f}" '
                 f'stroke="#b5532c" stroke-width="1.6" stroke-dasharray="7 5"/>')
        o.append(text(sx(150), sy(-58), TREND["label"], 12, "#9a4424"))
    # gate tick
    if GATE:
        gx, gy = sx(GATE["epoch"]), sy(GATE["elo"])
        o.append(f'<line x1="{gx - 9:.1f}" y1="{gy:.1f}" x2="{gx + 9:.1f}" y2="{gy:.1f}" stroke="{INK2}" '
                 f'stroke-width="1.6"/>')
        o.append(f'<line x1="{gx:.1f}" y1="{gy:.1f}" x2="{gx:.1f}" y2="{gy - 8:.1f}" stroke="{INK2}" stroke-width="1.6"/>')
        o.append(text(gx, gy + 17, GATE["label"], 11.5, INK2, "middle"))
    # series
    for s in SERIES:
        for e, v, m in s["points"]:
            o += errbar(sx(e), v, m, s["color"])
        for e, v, m in s["points"]:
            o.append(marker(s["marker"], sx(e), sy(v), s["color"]))
            o.append(text(sx(e) + 8, sy(v) + 4, f"{v:+d}" if v else "0", 10.5, INK2))
    # longer time control and the star
    c, red = "#4a3aa7", "#d23f3e"
    rows = [(s["marker"], s["color"], s["label"], False) for s in SERIES]
    if LONG_TC:
        o += errbar(sx(LONG_TC["epoch"]), LONG_TC["elo"], LONG_TC["margin"], c)
        o.append(marker("circle", sx(LONG_TC["epoch"]), sy(LONG_TC["elo"]), c, hollow=True))
        o.append(text(sx(LONG_TC["epoch"]), sy(LONG_TC["elo"] - LONG_TC["margin"]) + 15,
                      f"{LONG_TC['elo']:+d} at {LONG_TC['label']}", 10.5, INK2, "middle"))
        rows.append(("circle", c, f"Longer time control ({LONG_TC['label']})", True))
    if STAR:
        o += errbar(sx(STAR["epoch"]), STAR["elo"], STAR["margin"], red)
        o.append(marker("star", sx(STAR["epoch"]), sy(STAR["elo"]), red))
        for i, s in enumerate(STAR["label"]):
            o.append(text(R - 10, sy(-32) + 16 * i, s, 12.5 if i == 0 else 11.5, red if i == 0 else INK2,
                          "end", "bold" if i == 0 else "normal"))
        rows.append(("star", red, "F3 net of epoch 415 at 90+0.9", False))
    # notes
    for i, s in enumerate(NOTES):
        o.append(text(sx(NOTES_AT[0]), sy(NOTES_AT[1]) + 15 * i, s, 11.5, INK3))
    # legend
    lx, ly = sx(LEGEND_AT[0]), sy(LEGEND_AT[1])
    o.append(f'<rect x="{lx:.1f}" y="{ly:.1f}" width="330" height="{22 * len(rows) + 30}" rx="4" fill="{SURF}" '
             f'stroke="{GRID}"/>')
    o.append(text(lx + 14, ly + 20, LEGEND_TITLE, 11.5, INK, weight="bold"))
    for i, (k, col, lab, hol) in enumerate(rows):
        yy = ly + 42 + 22 * i
        o.append(marker(k, lx + 22, yy - 4, col, 5, hol))
        o.append(text(lx + 40, yy, lab, 12, INK2))
    # titles
    o.append(text(L - 54, 36, TITLE, 20, INK, weight="bold"))
    o.append(text(L - 54, 60, SUBTITLE, 13, INK2))
    o.append(f'<rect x="{L}" y="{T}" width="{R - L}" height="{B - T}" fill="none" stroke="{GRID}"/>')
    o.append("</svg>")
    return "\n".join(o)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_suffix(".svg")
    out.write_text(build(), encoding="utf-8")
    print(f"written {out}")
