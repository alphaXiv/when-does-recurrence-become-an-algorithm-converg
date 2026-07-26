#!/usr/bin/env python3
"""Render dependency-free SVG figures from the frozen aggregate evidence."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).parent
DATA = json.loads((ROOT / "data" / "results.json").read_text())
OUT = ROOT / "images"
OUT.mkdir(exist_ok=True)

INK = "#17212b"
MUTED = "#64748b"
GRID = "#dbe4ee"
BLUE = "#1464f4"
TEAL = "#009a88"
ORANGE = "#e87920"
RED = "#c9364f"
BG = "#fbfdff"


def esc(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def svg_start(title: str, subtitle: str = "") -> list[str]:
    out = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="520" viewBox="0 0 900 520">',
        f'<rect width="900" height="520" fill="{BG}"/>',
        f'<text x="54" y="42" font-family="Inter,Arial,sans-serif" font-size="25" font-weight="700" fill="{INK}">{esc(title)}</text>',
    ]
    if subtitle:
        out.append(
            f'<text x="54" y="68" font-family="Inter,Arial,sans-serif" font-size="14" fill="{MUTED}">{esc(subtitle)}</text>'
        )
    return out


def axes(
    out: list[str],
    x_ticks: list[float],
    y_ticks: list[float],
    xmap,
    ymap,
    xlabel: str,
    ylabel: str,
) -> None:
    left, top, right, bottom = 92, 92, 850, 438
    for y in y_ticks:
        py = ymap(y)
        out.append(f'<line x1="{left}" y1="{py:.1f}" x2="{right}" y2="{py:.1f}" stroke="{GRID}"/>')
        out.append(f'<text x="{left-12}" y="{py+5:.1f}" text-anchor="end" font-family="Inter,Arial,sans-serif" font-size="13" fill="{MUTED}">{y:g}</text>')
    for x in x_ticks:
        px = xmap(x)
        out.append(f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{bottom}" stroke="{GRID}"/>')
        out.append(f'<text x="{px:.1f}" y="{bottom+24}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="13" fill="{MUTED}">{x:g}</text>')
    out.extend(
        [
            f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="{INK}" stroke-width="1.5"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="{INK}" stroke-width="1.5"/>',
            f'<text x="{(left+right)/2}" y="500" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="15" fill="{INK}">{esc(xlabel)}</text>',
            f'<text x="22" y="{(top+bottom)/2}" text-anchor="middle" transform="rotate(-90 22 {(top+bottom)/2})" font-family="Inter,Arial,sans-serif" font-size="15" fill="{INK}">{esc(ylabel)}</text>',
        ]
    )


def save(name: str, out: list[str]) -> None:
    out.append("</svg>")
    (OUT / name).write_text("\n".join(out) + "\n")


def budget_law() -> None:
    out = svg_start(
        "The learned frontier tracks the training contract",
        "Filled circles: 0.44M-parameter S4 models; open squares: 1.72M-parameter robustness series",
    )
    lo, hi = math.log(0.4), math.log(4.5)
    xmap = lambda x: 92 + (math.log(x) - lo) / (hi - lo) * (850 - 92)
    ymap = lambda y: 438 - (math.log(y) - lo) / (hi - lo) * (438 - 92)
    axes(out, [0.5, 1, 2, 4], [0.5, 1, 2, 4], xmap, ymap, "contract demand  n / T", "measured positions per loop")
    out.append(f'<line x1="{xmap(.4):.1f}" y1="{ymap(.4):.1f}" x2="{xmap(4.5):.1f}" y2="{ymap(4.5):.1f}" stroke="{MUTED}" stroke-width="2" stroke-dasharray="7 6"/>')
    for row in DATA["primary_s4"]:
        x, y, seeds = row["demand"], row["median_speed"], row["seed_speeds"]
        out.append(f'<line x1="{xmap(x):.1f}" y1="{ymap(min(seeds)):.1f}" x2="{xmap(x):.1f}" y2="{ymap(max(seeds)):.1f}" stroke="{BLUE}" stroke-width="2"/>')
        out.append(f'<circle cx="{xmap(x):.1f}" cy="{ymap(y):.1f}" r="7" fill="{BLUE}" stroke="white" stroke-width="2"/>')
    for row in DATA["medium_width"]:
        x, y = row["demand"], row["median_speed"]
        px, py = xmap(x), ymap(y)
        out.append(f'<rect x="{px-6:.1f}" y="{py-6:.1f}" width="12" height="12" fill="{BG}" stroke="{TEAL}" stroke-width="3"/>')
    fit = DATA["power_fit"]
    out.append(f'<rect x="585" y="340" width="245" height="70" rx="10" fill="white" stroke="{GRID}"/>')
    out.append(f'<text x="602" y="368" font-family="Inter,Arial,sans-serif" font-size="15" font-weight="700" fill="{INK}">S4 fit: exponent {fit["exponent"]:.3f} ± {fit["standard_error"]:.3f}</text>')
    out.append(f'<text x="602" y="392" font-family="Inter,Arial,sans-serif" font-size="14" fill="{MUTED}">log-space R² = {fit["log_r2"]:.3f}; dashed = ideal</text>')
    save("budget-law.svg", out)


def rescue_curve() -> None:
    out = svg_start(
        "Extra loops rescue the late prefix—until overthinking",
        "S4 demand 2 at fixed length 32; line is the four-seed median, whiskers span seeds",
    )
    xmap = lambda x: 92 + (x - 8) / 32 * (850 - 92)
    ymap = lambda y: 438 - y * (438 - 92)
    axes(out, [8, 16, 24, 32, 40], [0, .25, .5, .75, 1], xmap, ymap, "loops", "late-quartile accuracy")
    rows = DATA["fixed_n32_rescue"]
    med = [statistics.median(v) for v in rows["late_accuracy_by_seed"]]
    path = " ".join(("M" if i == 0 else "L") + f" {xmap(x):.1f} {ymap(y):.1f}" for i, (x, y) in enumerate(zip(rows["loops"], med)))
    out.append(f'<path d="{path}" fill="none" stroke="{BLUE}" stroke-width="4"/>')
    for x, vals, y in zip(rows["loops"], rows["late_accuracy_by_seed"], med):
        out.append(f'<line x1="{xmap(x):.1f}" y1="{ymap(min(vals)):.1f}" x2="{xmap(x):.1f}" y2="{ymap(max(vals)):.1f}" stroke="{BLUE}" stroke-width="2"/>')
        out.append(f'<circle cx="{xmap(x):.1f}" cy="{ymap(y):.1f}" r="6" fill="{BLUE}"/>')
    px = xmap(rows["predicted_loops"])
    py = ymap(statistics.median(rows["predicted_late_accuracy_by_seed"]))
    out.append(f'<line x1="{px:.1f}" y1="92" x2="{px:.1f}" y2="438" stroke="{ORANGE}" stroke-width="2" stroke-dasharray="6 5"/>')
    out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="8" fill="{ORANGE}" stroke="white" stroke-width="2"/>')
    out.append(f'<text x="{px+10:.1f}" y="118" font-family="Inter,Arial,sans-serif" font-size="14" font-weight="700" fill="{ORANGE}">speed rule: 17 loops</text>')
    save("loop-rescue.svg", out)


def ood_halting() -> None:
    out = svg_start(
        "The stopping rule transfers out of distribution only selectively",
        "Length 64 was never trained; each dot is one seed evaluated at its speed-predicted loop count",
    )
    labels = [row["demand"] for row in DATA["ood_n64_predicted_halting"]]
    xmap = lambda x: 150 + labels.index(x) * 200
    ymap = lambda y: 438 - y * (438 - 92)
    axes(out, labels, [0, .25, .5, .75, 1], xmap, ymap, "contract demand  n / T", "late-quartile accuracy at predicted halt")
    jitter = [-18, -6, 6, 18]
    for row in DATA["ood_n64_predicted_halting"]:
        vals = row["late_accuracy_by_seed"]
        for dx, value in zip(jitter, vals):
            out.append(f'<circle cx="{xmap(row["demand"])+dx:.1f}" cy="{ymap(value):.1f}" r="6" fill="{TEAL}" opacity=".82"/>')
        median = statistics.median(vals)
        out.append(f'<line x1="{xmap(row["demand"])-29:.1f}" y1="{ymap(median):.1f}" x2="{xmap(row["demand"])+29:.1f}" y2="{ymap(median):.1f}" stroke="{INK}" stroke-width="4"/>')
    out.append(f'<text x="282" y="112" font-family="Inter,Arial,sans-serif" font-size="14" font-weight="700" fill="{TEAL}">Demand 2/3: all seeds 0.80–0.97</text>')
    save("ood-halting.svg", out)


def thresholds() -> None:
    out = svg_start(
        "The frontier estimate is robust at 80–90%, brittle at 95%",
        "Median length-32 slope across four S4 seeds under three accuracy thresholds",
    )
    xvals = [row["demand"] for row in DATA["threshold_sensitivity"]]
    xmap = lambda x: 120 + xvals.index(x) * 220
    ymap = lambda y: 438 - y / 2.1 * (438 - 92)
    axes(out, xvals, [0, .5, 1, 1.5, 2], xmap, ymap, "contract demand  n / T", "measured positions per loop")
    series = [("q80", "80%", BLUE), ("q90", "90%", TEAL), ("q95", "95%", RED)]
    offsets = [-18, 0, 18]
    for (key, label, color), dx in zip(series, offsets):
        for row in DATA["threshold_sensitivity"]:
            out.append(f'<circle cx="{xmap(row["demand"])+dx:.1f}" cy="{ymap(row[key]):.1f}" r="6" fill="{color}"/>')
    for i, (_, label, color) in enumerate(series):
        x = 360 + i * 78
        out.append(f'<circle cx="{x}" cy="112" r="5" fill="{color}"/><text x="{x+10}" y="117" font-family="Inter,Arial,sans-serif" font-size="13" fill="{INK}">{label}</text>')
    save("thresholds.svg", out)


def causal() -> None:
    out = svg_start(
        "Small activation damage did not reveal the claimed cone",
        "Fitted cone slopes by seed; the measured undamaged frontier is ≈2 positions per loop",
    )
    rows = DATA["activation_damage"]
    xmap = lambda i: 145 + i * 155
    ymap = lambda y: 438 - (y + .6) / 2.8 * (438 - 92)
    axes(out, [], [-.5, 0, .5, 1, 1.5, 2], xmap, ymap, "intervention condition", "fitted damage-cone slope")
    out.append(f'<line x1="92" y1="{ymap(2):.1f}" x2="850" y2="{ymap(2):.1f}" stroke="{BLUE}" stroke-width="2" stroke-dasharray="7 5"/>')
    out.append(f'<text x="670" y="{ymap(2)-8:.1f}" font-family="Inter,Arial,sans-serif" font-size="13" fill="{BLUE}">frontier speed ≈ 2</text>')
    jitter = [-18, -6, 6, 18]
    for i, row in enumerate(rows):
        for dx, value in zip(jitter, row["slopes"]):
            out.append(f'<circle cx="{xmap(i)+dx:.1f}" cy="{ymap(value):.1f}" r="6" fill="{RED}" opacity=".8"/>')
        words = row["condition"].split()
        out.append(f'<text x="{xmap(i):.1f}" y="463" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="12" fill="{MUTED}">{esc(words[0])}</text>')
        out.append(f'<text x="{xmap(i):.1f}" y="479" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="12" fill="{MUTED}">{esc(" ".join(words[1:]))}</text>')
    save("causal-cones.svg", out)


if __name__ == "__main__":
    budget_law()
    rescue_curve()
    ood_halting()
    thresholds()
    causal()
    print(f"wrote 5 figures to {OUT}")
