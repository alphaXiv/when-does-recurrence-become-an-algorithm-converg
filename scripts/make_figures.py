"""Generate the publication SVGs from the fresh recovery-run summaries.

The values below are compact summaries of terminal ORX_FINAL_EVIDENCE_JSON
records. Raw run identifiers and full measurements remain in `orx exp desc`.
No plotting dependency is required.
"""

from pathlib import Path
from math import log


OUT = Path("reports/convergence-budget/images")
W, H = 920, 520
INK = "#152238"
MUTED = "#64748b"
BLUE = "#2563eb"
ORANGE = "#ea580c"
GREEN = "#15803d"
RED = "#be123c"
GRID = "#dbe3ee"


def esc(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg_start(title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="72" y="42" font-family="system-ui,sans-serif" font-size="24" font-weight="700" fill="{INK}">{esc(title)}</text>',
        f'<text x="72" y="68" font-family="system-ui,sans-serif" font-size="14" fill="{MUTED}">{esc(subtitle)}</text>',
    ]


def text(x: float, y: float, value: object, size: int = 13, color: str = INK, anchor: str = "middle", weight: int = 400) -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="system-ui,sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}">{esc(value)}</text>'


def axes(lines: list[str], xmin: float, xmax: float, ymin: float, ymax: float, xticks: list[float], yticks: list[float], xlabel: str, ylabel: str):
    left, right, top, bottom = 88, 884, 92, 448
    sx = lambda x: left + (x - xmin) / (xmax - xmin) * (right - left)
    sy = lambda y: bottom - (y - ymin) / (ymax - ymin) * (bottom - top)
    for y in yticks:
        lines.append(f'<line x1="{left}" x2="{right}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{GRID}" stroke-width="1"/>')
        lines.append(text(left - 12, sy(y) + 5, f"{y:g}", 12, MUTED, "end"))
    for x in xticks:
        lines.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{top}" y2="{bottom}" stroke="{GRID}" stroke-width="1"/>')
        lines.append(text(sx(x), bottom + 24, f"{x:g}", 12, MUTED))
    lines.extend([
        f'<line x1="{left}" x2="{right}" y1="{bottom}" y2="{bottom}" stroke="{INK}" stroke-width="1.5"/>',
        f'<line x1="{left}" x2="{left}" y1="{top}" y2="{bottom}" stroke="{INK}" stroke-width="1.5"/>',
        text((left + right) / 2, 495, xlabel, 14, INK),
        f'<text x="22" y="{(top + bottom) / 2}" transform="rotate(-90 22 {(top + bottom) / 2})" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" fill="{INK}">{esc(ylabel)}</text>',
    ])
    return sx, sy


def polyline(lines: list[str], points: list[tuple[float, float]], sx, sy, color: str, width: int = 3, dash: str | None = None):
    coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    lines.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"{extra}/>')


def circles(lines: list[str], points: list[tuple[float, float]], sx, sy, color: str, radius: int = 6):
    for x, y in points:
        lines.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{radius}" fill="{color}" stroke="#fff" stroke-width="2"/>')


def budget_law():
    lines = svg_start(
        "Measured convergence speed follows training demand",
        "S4 prefix products; q=0.90 frontier. Dashed line is the paper's ideal speed = n_train / T_train.",
    )
    sx, sy = axes(lines, 0, 4.25, 0, 4.25, [0.5, 1, 2, 3, 4], [0, 1, 2, 3, 4], "contract demand  n_train / T_train", "measured frontier speed")
    ideal = [(0, 0), (4.15, 4.15)]
    small = [(0.5, 0.509), (2 / 3, 0.710), (1, 0.991), (2, 1.990), (4, 1.771)]
    medium = [(0.5, 0.516), (2 / 3, 0.733), (1, 0.985), (2, 1.966), (4, 3.367)]
    polyline(lines, ideal, sx, sy, MUTED, 2, "7 6")
    polyline(lines, small, sx, sy, BLUE, 3)
    polyline(lines, medium, sx, sy, ORANGE, 3)
    circles(lines, small, sx, sy, BLUE)
    circles(lines, medium, sx, sy, ORANGE)
    lines += [
        f'<line x1="620" x2="648" y1="112" y2="112" stroke="{BLUE}" stroke-width="4"/>', text(657, 117, "0.44M parameters", 13, BLUE, "start", 600),
        f'<line x1="620" x2="648" y1="136" y2="136" stroke="{ORANGE}" stroke-width="4"/>', text(657, 141, "1.72M parameters", 13, ORANGE, "start", 600),
        text(sx(2.05), sy(2.0) - 13, "near-linear through demand 2", 13, GREEN, "start", 600),
        text(sx(4) - 8, sy(1.771) + 24, "small model saturates", 12, BLUE, "end"),
        "</svg>",
    ]
    (OUT / "budget_law.svg").write_text("\n".join(lines))


def rescue():
    lines = svg_start(
        "Extra loops move the correctness frontier—but too many can erase it",
        "Median token accuracy on length-64 S4 inputs; each curve uses a different training contract.",
    )
    sx, sy = axes(lines, 0, 320, 0, 1.02, [0, 64, 128, 192, 256, 320], [0, 0.25, 0.5, 0.75, 1], "loops used at evaluation", "token accuracy")
    curves = [
        ("demand 0.5", GREEN, [(64, .558), (96, .749), (128, .881), (160, .883), (192, .893), (256, .893), (320, .876)]),
        ("demand 0.67", BLUE, [(48, .577), (72, .843), (96, .953), (120, .941), (144, .930), (192, .904), (240, .881)]),
        ("demand 1", ORANGE, [(32, .546), (48, .751), (64, .725), (80, .541), (96, .358), (128, .129), (160, .078)]),
        ("demand 2", RED, [(16, .536), (24, .685), (32, .291), (40, .130), (48, .090), (64, .061)]),
    ]
    for _, color, pts in curves:
        polyline(lines, pts, sx, sy, color, 3)
        circles(lines, pts, sx, sy, color, 5)
    for i, (name, color, _) in enumerate(curves):
        y = 109 + i * 23
        lines += [f'<line x1="674" x2="702" y1="{y}" y2="{y}" stroke="{color}" stroke-width="4"/>', text(711, y + 5, name, 12, color, "start", 600)]
    lines += [
        text(sx(127), sy(.88) - 12, "rescued near predicted budget", 12, GREEN, "start", 600),
        text(sx(76), sy(.43), "overthinking", 12, RED, "start", 600),
        "</svg>",
    ]
    (OUT / "loop_rescue.svg").write_text("\n".join(lines))


def thresholds():
    lines = svg_start(
        "The exponent is robust in solved contracts, not at the capacity edge",
        "Measured speed divided by requested speed for three correctness thresholds; 1.0 is exact delivery.",
    )
    demands = [0.5, 2 / 3, 1, 2, 4]
    vals = {
        "80%": [1.038, 1.071, 1.000, 1.000, 0.750],
        "90%": [1.012, 1.070, 0.994, 1.000, 0.443],
        "95%": [0.276, 1.056, 0.985, 1.000, 0.403],
    }
    left, right, top, bottom = 100, 884, 104, 448
    sy = lambda y: bottom - y / 1.2 * (bottom - top)
    for y in [0, .25, .5, .75, 1]:
        lines += [f'<line x1="{left}" x2="{right}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{GRID}"/>', text(left - 12, sy(y) + 4, y, 12, MUTED, "end")]
    colors = [GREEN, BLUE, ORANGE]
    group_w = (right - left) / len(demands)
    bw = 34
    for i, d in enumerate(demands):
        center = left + group_w * (i + .5)
        for j, (label, data) in enumerate(vals.items()):
            x = center + (j - 1) * (bw + 4)
            y = sy(data[i])
            lines.append(f'<rect x="{x-bw/2:.1f}" y="{y:.1f}" width="{bw}" height="{bottom-y:.1f}" rx="3" fill="{colors[j]}"/>')
        lines.append(text(center, bottom + 24, f"{d:.2g}", 12, MUTED))
    lines += [
        f'<line x1="{left}" x2="{right}" y1="{sy(1):.1f}" y2="{sy(1):.1f}" stroke="{INK}" stroke-width="2" stroke-dasharray="6 5"/>',
        text((left + right) / 2, 494, "contract demand", 14),
        f'<text x="25" y="276" transform="rotate(-90 25 276)" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" fill="{INK}">speed / requested speed</text>',
    ]
    for j, label in enumerate(vals):
        x = 650 + j * 76
        lines += [f'<rect x="{x}" y="82" width="14" height="14" fill="{colors[j]}"/>', text(x + 20, 94, label, 12, colors[j], "start", 600)]
    lines += [text(805, 296, "capacity edge", 12, RED, "middle", 600), "</svg>"]
    (OUT / "threshold_robustness.svg").write_text("\n".join(lines))


def capacity():
    lines = svg_start(
        "The four-step-per-loop contract is capacity sensitive",
        "Median q=0.90 speed at demand 4. Longer training helps little; doubling width closes much of the gap.",
    )
    labels = ["paper target", "small", "small + longer dwell", "medium width"]
    values = [4.0, 1.771, 2.011, 3.367]
    colors = [MUTED, BLUE, BLUE, ORANGE]
    left, right, top, bottom = 130, 870, 100, 440
    sx = lambda v: left + v / 4.2 * (right - left)
    for x in [0, 1, 2, 3, 4]:
        lines += [f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{top}" y2="{bottom}" stroke="{GRID}"/>', text(sx(x), bottom + 24, x, 12, MUTED)]
    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        y = 126 + i * 78
        lines += [
            text(left - 16, y + 18, label, 13, INK, "end", 600),
            f'<rect x="{left}" y="{y}" width="{sx(value)-left:.1f}" height="34" rx="5" fill="{color}"/>',
            text(sx(value) + 12, y + 23, f"{value:.2f}", 13, color, "start", 700),
        ]
    lines += [text((left + right) / 2, 496, "measured convergence speed", 14), "</svg>"]
    (OUT / "capacity_edge.svg").write_text("\n".join(lines))


def causal():
    lines = svg_start(
        "Activation damage is directed, but its cone speed is not stable",
        "Cone slopes from full-resampling patches. Dashed segments mark each model's measured frontier speed.",
    )
    sx, sy = axes(lines, 0.5, 12.5, 0, 2.2, [1, 4, 8, 12], [0, .5, 1, 1.5, 2], "patch condition × seed", "measured damage-cone speed")
    slopes = [.354, .143, .071, .077, 0, 0, 0, 0, 2.052, .034, 0, 0]
    colors = [ORANGE] * 4 + [BLUE] * 8
    lines += [
        f'<line x1="{sx(.7):.1f}" x2="{sx(4.3):.1f}" y1="{sy(.984):.1f}" y2="{sy(.984):.1f}" stroke="{ORANGE}" stroke-width="2" stroke-dasharray="6 5"/>',
        f'<line x1="{sx(4.7):.1f}" x2="{sx(12.3):.1f}" y1="{sy(1.99):.1f}" y2="{sy(1.99):.1f}" stroke="{BLUE}" stroke-width="2" stroke-dasharray="6 5"/>',
    ]
    for i, (value, color) in enumerate(zip(slopes, colors), start=1):
        lines.append(f'<circle cx="{sx(i):.1f}" cy="{sy(value):.1f}" r="7" fill="{color}" stroke="#fff" stroke-width="2"/>')
    lines += [
        text(sx(2.5), sy(.984) - 9, "unit-speed frontier", 12, ORANGE),
        text(sx(8.2), sy(1.99) - 9, "demand-2 frontier", 12, BLUE),
        text(sx(10), sy(2.052) + 27, "1 of 12 compatible", 12, GREEN, "middle", 700),
        text(sx(2.5), 476, "unit, source pos. 1", 11, ORANGE),
        text(sx(6.5), 476, "demand 2, pos. 1", 11, BLUE),
        text(sx(10.5), 476, "demand 2, pos. 8", 11, BLUE),
        "</svg>",
    ]
    (OUT / "damage_cone.svg").write_text("\n".join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    budget_law()
    rescue()
    thresholds()
    capacity()
    causal()


if __name__ == "__main__":
    main()
