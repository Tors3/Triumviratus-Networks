"""Elo trajectory of Consilium Plebis across training, against the final Consilium with the same search.

Same drawing code as consilium_elo_trajectory.py (imported); the points come from consilium2_elo_points.csv, one
series per match condition (fixed nodes, time control). Add a row to the CSV after each match and rerun.
Usage: python consilium2_elo_trajectory.py [out.svg]    (PNG: msedge --headless as in consilium_elo_trajectory.py)
"""
import csv
import sys
from pathlib import Path

import consilium_elo_trajectory as base

HERE = Path(__file__).parent
COLORS = {"nodes": ("#1baf7a", "diamond"), "time": ("#2a78d6", "circle")}
TIME_COLORS = ["#2a78d6", "#eb6834", "#4a3aa7", "#d23f3e"]   # one per time control, in order of appearance


def load():
    groups = {}
    for row in csv.DictReader(open(HERE / "consilium2_elo_points.csv", encoding="utf-8")):
        cond = row["condition"]
        # 24/10/2026 (author): only the fixed 100k-node series is drawn; the time-control rows stay in the CSV.
        # 25/10/2026: second fixed-node series, both sides 8.1 dev builds of the same source and C2 search levers on
        # (condition "8.1 dev fixed 100k ..."), drawn apart because its opponent and search differ.
        if cond.startswith("8.1 dev fixed 100k"):
            label = cond.split(" vs ")[0]
            if label not in groups:
                groups[label] = {"label": label, "color": "#eb6834", "marker": "circle", "points": []}
            groups[label]["points"].append((int(row["epoch"]), round(float(row["elo"])), float(row["margin95"])))
            continue
        if not cond.startswith("fixed 100k"):
            continue
        key = "nodes" if "nodes" in cond else "time"
        label = cond.split(" vs ")[0]
        if label not in groups:
            n_time = sum(1 for g in groups.values() if g["marker"] == "circle")
            color = COLORS[key][0] if key == "nodes" else TIME_COLORS[n_time % len(TIME_COLORS)]
            groups[label] = {"label": label, "color": color, "marker": COLORS[key][1], "points": []}
        g = groups[label]
        g["points"].append((int(row["epoch"]), round(float(row["elo"])), float(row["margin95"])))
    return list(groups.values())


def linear_trend(series):
    """Least-squares line through the fixed-node points: (slope per epoch, epoch where it reaches 0, first, last)."""
    pts = [p for s in series if "nodes" in s["label"] for p in s["points"]]
    if len(pts) < 3:
        return None
    n = len(pts)
    mx = sum(e for e, _, _ in pts) / n
    my = sum(v for _, v, _ in pts) / n
    sxx = sum((e - mx) ** 2 for e, _, _ in pts)
    slope = sum((e - mx) * (v - my) for e, v, _ in pts) / sxx
    if slope <= 0:
        return None
    return slope, mx - my / slope, min(e for e, _, _ in pts), max(e for e, _, _ in pts)


def configure():
    base.SERIES = load()
    base.LONG_TC = base.STAR = base.TREND = base.GATE = None
    # 24/10/2026: the straight line through all points is superseded by the non-linear fits (consilium2_elo_fit.py)
    # and the transposition chart; kept as an option
    t = linear_trend(base.SERIES) if "--trend" in sys.argv else None
    if t:
        slope, zero, first, _ = t
        base.TREND = {"from": first, "to": min(zero, 300), "slope": slope, "zero": zero,
                      "label": f"Linear trend (fixed nodes): +{slope * 10:.1f} per 10 epochs, level at epoch "
                               f"~{zero:.0f} (an early bound: curves flatten)"}
    base.LR, base.BANDS, base.LINES = [], [], []
    base.NOTES = ["Opponent: the final Consilium (8.0 release network), same merged 8.1 search."]
    base.TITLE = "Consilium Plebis against Consilium across training"
    base.SUBTITLE = "Elo of epoch exports against the final Consilium; pretraining P of 550 epochs, started 21/10/2026"
    base.X0, base.X1, base.Y0, base.Y1 = 0, 560, -160, 40   # x up to 560 (26/10: ep435 past 400; the run ends at 550)
    base.XLABEL = "Training epoch (P)"
    base.YLABEL = "Elo against Consilium"
    base.NOTES_AT = (20, -150)
    base.LEGEND_AT = (160, -95)
    base.LEGEND_TITLE = "Match condition"


if __name__ == "__main__":
    configure()
    args = [a for a in sys.argv[1:] if a != "--trend"]
    out = Path(args[0]) if args else Path(__file__).with_suffix(".svg")
    out.write_text(base.build(), encoding="utf-8")
    print(f"written {out}")
