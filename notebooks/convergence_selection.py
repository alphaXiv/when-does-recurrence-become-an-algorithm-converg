import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    evidence = {
        "s4": [
            {"demand": 0.5, "median": 0.508877, "seeds": [0.505627, 0.512127, 0.588091, 0.465202]},
            {"demand": 2 / 3, "median": 0.709572, "seeds": [0.699313, 0.659750, 0.719830, 0.745344]},
            {"demand": 1.0, "median": 0.983699, "seeds": [1.0, 0.967397, 0.927675, 1.0]},
            {"demand": 2.0, "median": 1.994485, "seeds": [1.979412, 2.0, 2.0, 1.988971]},
        ],
        "fit": {"exponent": 0.969341, "se": 0.034192, "r2": 0.997518},
        "thresholds": {
            "80%": [0.522716, 0.721862, 1.0, 1.964286],
            "90%": [0.517754, 0.716717, 0.993771, 1.988971],
            "95%": [0.128413, 0.335205, 0.966407, 1.924265],
        },
        "rescue": {
            "loops": [8, 12, 16, 20, 24, 32, 40],
            "median_late": [0.046875, 0.174561, 0.986084, 0.996338, 0.987061, 0.251709, 0.047119],
            "predicted_loops": 17,
            "predicted_late": 0.996338,
        },
        "ood": {
            "0.5": [0.927979, 0.545898, 0.908691, 0.050049],
            "0.667": [0.874, 0.852, 0.967, 0.796],
            "1": [0.139, 0.599, 0.736, 0.049],
            "2": [0.046, 0.043, 0.043, 0.041],
        },
        "small_patch_slopes": [0.0] * 8,
        "full_mid_patch_slopes": [2.052, 0.034, 0.0, 0.0],
    }
    return (evidence,)


@app.cell
def _(evidence):
    import math

    def budget_svg(rows):
        width, height = 760, 390
        left, top, right, bottom = 70, 35, 725, 330
        lo, hi = math.log(0.4), math.log(2.3)
        xmap = lambda x: left + (math.log(x) - lo) / (hi - lo) * (right - left)
        ymap = lambda y: bottom - (math.log(y) - lo) / (hi - lo) * (bottom - top)
        parts = [
            f'<svg viewBox="0 0 {width} {height}" style="width:100%;background:#fbfdff;border:1px solid #dbe4ee;border-radius:12px">',
            '<text x="22" y="24" font-family="sans-serif" font-size="17" font-weight="700" fill="#17212b">Measured convergence speed follows contract demand</text>',
        ]
        for value in (0.5, 1, 2):
            x, y = xmap(value), ymap(value)
            parts += [
                f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#dbe4ee"/>',
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="#dbe4ee"/>',
                f'<text x="{left-9}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#64748b">{value:g}</text>',
                f'<text x="{x:.1f}" y="{bottom+20}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#64748b">{value:g}</text>',
            ]
        parts.append(f'<line x1="{xmap(.4):.1f}" y1="{ymap(.4):.1f}" x2="{xmap(2.3):.1f}" y2="{ymap(2.3):.1f}" stroke="#64748b" stroke-width="2" stroke-dasharray="6 5"/>')
        for row in rows:
            x, y = xmap(row["demand"]), ymap(row["median"])
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="#1464f4" stroke="white" stroke-width="2"/>')
        parts += [
            f'<text x="{(left+right)/2}" y="382" text-anchor="middle" font-family="sans-serif" font-size="13">contract demand n / T</text>',
            f'<text x="17" y="{(top+bottom)/2}" text-anchor="middle" transform="rotate(-90 17 {(top+bottom)/2})" font-family="sans-serif" font-size="13">positions per loop</text>',
            "</svg>",
        ]
        return "".join(parts)

    headline_svg = budget_svg(evidence["s4"])
    return (headline_svg,)


@app.cell
def _(evidence, headline_svg, mo):
    mo.vstack(
        [
            mo.md(
                f"""
                # When does recurrence become an algorithm?

                A looped transformer reuses the same computation block. The paper asks whether
                training teaches that block a predictable *rate* for moving a correct prefix
                forward, and whether more loops can extend computation after training.

                **Verdict: partially reproduced.** On fresh Kubernetes runs, four competent S4
                contracts followed the budget law with exponent **{evidence["fit"]["exponent"]:.3f}
                ± {evidence["fit"]["se"]:.3f}** (log-space R²
                **{evidence["fit"]["r2"]:.3f}**). Loop rescue was strong in a controlled fixed-length
                test and selective at unseen length 64; the small activation-damage cone was not
                observed.
                """
            ),
            mo.Html(headline_svg),
            mo.md(
                "The dashed diagonal is the paper’s prediction: a contract demanding two positions "
                "per loop should learn a frontier speed of two. Each blue point is a four-seed median."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    threshold = mo.ui.dropdown(
        options=["80%", "90%", "95%"],
        value="90%",
        label="Convergence threshold",
    )
    threshold
    return (threshold,)


@app.cell
def _(evidence, mo, threshold):
    demands = [0.5, 2 / 3, 1.0, 2.0]
    values = evidence["thresholds"][threshold.value]
    rows = [
        {
            "demand": f"{demand:.3g}",
            "measured speed": f"{value:.3f}",
            "difference": f"{value - demand:+.3f}",
        }
        for demand, value in zip(demands, values)
    ]
    mo.vstack(
        [
            mo.md(
                """
                ## How the frontier was measured

                At every loop, we measured token accuracy by position and found the longest
                contiguous prefix over the selected threshold. A line fitted to prefix length
                versus loop count gives positions per loop. The preregistered 80% and 90%
                thresholds agree; 95% is brittle for the two slow contracts.
                """
            ),
            mo.ui.table(rows, selection=None),
        ]
    )
    return


@app.cell
def _(evidence):
    def rescue_svg(rescue):
        left, top, right, bottom = 62, 25, 735, 285
        xmap = lambda x: left + (x - 8) / 32 * (right - left)
        ymap = lambda y: bottom - y * (bottom - top)
        pts = " ".join(
            f"{xmap(x):.1f},{ymap(y):.1f}"
            for x, y in zip(rescue["loops"], rescue["median_late"])
        )
        pred_x, pred_y = xmap(rescue["predicted_loops"]), ymap(rescue["predicted_late"])
        return f"""
        <svg viewBox="0 0 760 340" style="width:100%;background:#fbfdff;border:1px solid #dbe4ee;border-radius:12px">
          <line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#17212b"/>
          <line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#17212b"/>
          <polyline points="{pts}" fill="none" stroke="#1464f4" stroke-width="4"/>
          <line x1="{pred_x:.1f}" y1="{top}" x2="{pred_x:.1f}" y2="{bottom}" stroke="#e87920" stroke-width="2" stroke-dasharray="6 5"/>
          <circle cx="{pred_x:.1f}" cy="{pred_y:.1f}" r="7" fill="#e87920"/>
          <text x="{pred_x+10:.1f}" y="48" font-family="sans-serif" font-size="13" fill="#e87920">predicted: 17 loops</text>
          <text x="385" y="330" text-anchor="middle" font-family="sans-serif" font-size="13">loops (8 → 40)</text>
          <text x="17" y="155" text-anchor="middle" transform="rotate(-90 17 155)" font-family="sans-serif" font-size="13">late-prefix accuracy</text>
        </svg>
        """

    rescue_chart = rescue_svg(evidence["rescue"])
    return (rescue_chart,)


@app.cell
def _(evidence, mo, rescue_chart):
    ood_rows = [
        {
            "demand": demand,
            "seed range at predicted halt": f"{min(values):.3f}–{max(values):.3f}",
        }
        for demand, values in evidence["ood"].items()
    ]
    mo.vstack(
        [
            mo.md(
                """
                ## Loops are a useful—and finite—computation budget

                For the demand-2 model at length 32, late-prefix accuracy rose from 0.047 at
                8 loops to 0.986 at 16. The in-distribution speed predicted 17 loops and achieved
                0.996. Continuing to 32 or 40 loops caused overthinking.
                """
            ),
            mo.Html(rescue_chart),
            mo.md(
                """
                Length 64 was unseen during training. The demand-2/3 contract transferred best:
                all four seeds scored 0.80–0.97 at the predicted halt. Other contracts were
                seed-variable or remained near chance, so the broad out-of-distribution claim is
                only partially aligned here.
                """
            ),
            mo.ui.table(ood_rows, selection=None),
        ]
    )
    return


@app.cell
def _(evidence, mo):
    mo.md(
        f"""
        ## The causal cone was the boundary of the reproduction

        A 25% donor-state patch produced slope **0 in all
        {len(evidence["small_patch_slopes"])} tested seed/configuration cases**, including a
        no-input-reinjection control. Full replacement at a middle position produced one
        compatible slope (2.052) but three self-healing responses
        ({", ".join(f"{x:.3f}" for x in evidence["full_mid_patch_slopes"][1:])}).
        That isolated, stronger intervention does not reproduce the requested *small-damage*
        propagation cone.

        ## What was reconstructed

        The formal task was the named S4 prefix-product benchmark. Four independent seeds per
        contract trained a two-layer, width-128, weight-tied transformer through lengths
        4/8/16/32 with AdamW; a 1.72M-parameter width-256 series checked scale sensitivity.
        The public paper did not provide author code or a full hyperparameter table, so group
        encoding, initialization, input reinjection, and damage thresholds are documented
        reconstruction choices. A small A5 extension did not establish competence and is not
        used as evidence.

        Every formal result shown here came from Kubernetes using NVIDIA RTX PRO 6000
        Blackwell GPUs, with a peak of 16 GPUs allocated concurrently and an actual campaign
        wall time of **1.376 hours**. The notebook embeds the measured aggregates, so Molab
        readers do not need the expensive checkpoints or repository-relative data files.
        """
    )
    return


if __name__ == "__main__":
    app.run()
