# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo>=0.14.0",
# ]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    import statistics
    import marimo as mo

    return math, mo


@app.cell
def _(mo):
    mo.md(r"""
    # When does recurrence become an algorithm?

    A looped transformer repeats the same learned block. The paper
    [arXiv:2607.20594](https://arxiv.org/abs/2607.20594) proposes that the
    loop budget used during training selects how quickly correct answers
    advance through a sequence. This notebook walks through fresh S4
    evidence produced on Kubernetes; all measurements are embedded, so
    opening it does **not** retrain a model.

    **Verdict: partially reproduced.** The compact model's frontier followed
    demand from 0.5 through 2 with exponent **0.968** versus the paper's
    **0.98 ± 0.04**. Rescue by extra loops was strong, while a compatible
    activation-damage slope appeared in only **1 of 12** tests.

    Compute: Kubernetes, NVIDIA RTX PRO 6000 Blackwell, four GPUs per run,
    16 GPUs at peak concurrency, and 1.376 hours of fresh campaign wall time.
    """)
    return


@app.cell
def _():
    compact = [
        {"demand": 0.5, "speed": 0.509},
        {"demand": 2 / 3, "speed": 0.710},
        {"demand": 1.0, "speed": 0.991},
        {"demand": 2.0, "speed": 1.990},
        {"demand": 4.0, "speed": 1.771},
    ]
    medium = [
        {"demand": 0.5, "speed": 0.516},
        {"demand": 2 / 3, "speed": 0.733},
        {"demand": 1.0, "speed": 0.985},
        {"demand": 2.0, "speed": 1.966},
        {"demand": 4.0, "speed": 3.367},
    ]
    return compact, medium


@app.cell
def _(compact, medium, mo):
    def chart_points(rows, color):
        def sx(value):
            return 55 + value / 4.2 * 560

        def sy(value):
            return 350 - value / 4.2 * 300

        path = " ".join(f"{sx(row['demand']):.1f},{sy(row['speed']):.1f}" for row in rows)
        dots = "".join(
            f"<circle cx='{sx(row['demand']):.1f}' cy='{sy(row['speed']):.1f}' r='5' fill='{color}'/>"
            for row in rows
        )
        return f"<polyline points='{path}' fill='none' stroke='{color}' stroke-width='3'/>{dots}"

    budget_svg = f"""
    <svg viewBox="0 0 660 410" style="width:100%;max-width:760px;background:white">
      <text x="55" y="25" font-family="sans-serif" font-size="19" font-weight="700">Frontier speed tracks the training contract</text>
      <text x="55" y="47" font-family="sans-serif" font-size="12" fill="#64748b">Dashed: exact delivery. Blue: 0.44M. Orange: 1.72M parameters.</text>
      <line x1="55" y1="350" x2="625" y2="350" stroke="#152238"/>
      <line x1="55" y1="350" x2="55" y2="50" stroke="#152238"/>
      <line x1="55" y1="350" x2="615" y2="50" stroke="#64748b" stroke-dasharray="6 5" stroke-width="2"/>
      {chart_points(compact, "#2563eb")}
      {chart_points(medium, "#ea580c")}
      <text x="335" y="397" text-anchor="middle" font-family="sans-serif" font-size="13">training demand n / T</text>
      <text x="17" y="210" transform="rotate(-90 17 210)" text-anchor="middle" font-family="sans-serif" font-size="13">measured speed</text>
      <text x="595" y="230" text-anchor="end" font-family="sans-serif" font-size="11" fill="#2563eb">small model saturates</text>
    </svg>
    """
    mo.Html(budget_svg)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The budget law

    For an input of length \(n\), training uses \(T=n/d\) loops. Demand
    \(d=2\), for example, asks the tied block to advance correctness by two
    positions on every application. A loopwise accuracy trace defines the
    longest contiguous prefix above 90% accuracy; the slope of that prefix
    is the measured speed.

    The first four compact conditions span a fourfold demand range and hug
    the identity line. Demand 4 is the useful exception: longer training
    only raises compact speed from 1.77 to 2.01, whereas doubling width
    reaches 3.37.
    """)
    return


@app.cell
def _(mo):
    fit_count = mo.ui.slider(3, 5, value=4, step=1, label="Number of compact contracts included in the power-law fit")
    fit_count
    return (fit_count,)


@app.cell
def _(compact, fit_count, math, mo):
    selected = compact[: fit_count.value]
    lx = [math.log(row["demand"]) for row in selected]
    ly = [math.log(row["speed"]) for row in selected]
    mean_x = sum(lx) / len(lx)
    mean_y = sum(ly) / len(ly)
    exponent = sum((x - mean_x) * (y - mean_y) for x, y in zip(lx, ly)) / sum(
        (x - mean_x) ** 2 for x in lx
    )
    intercept = mean_y - exponent * mean_x
    predictions = [intercept + exponent * x for x in lx]
    residual = sum((y - pred) ** 2 for y, pred in zip(ly, predictions))
    total = sum((y - mean_y) ** 2 for y in ly)
    r_squared = 1 - residual / total
    mo.md(
        rf"""
        **Fit through demand {selected[-1]['demand']:g}:**
        exponent **{exponent:.3f}**, \(R^2=\)**{r_squared:.3f}**.

        Including the compact demand-4 capacity failure changes the exponent;
        this control is why the reported 0.968 fit is explicitly restricted to
        the solved 0.5–2 regime.
        """
    )
    return


@app.cell
def _():
    rescue = [
        {"contract": "d=0.5", "early": 0.558, "best": 0.893, "predicted_loops": "109–138"},
        {"contract": "d=0.67", "early": 0.577, "best": 0.953, "predicted_loops": "89–94"},
        {"contract": "d=1", "early": 0.546, "best": 0.751, "predicted_loops": "63–69"},
        {"contract": "d=2, fixed n=32", "early": 0.548, "best": 0.996, "predicted_loops": "17"},
    ]
    return (rescue,)


@app.cell
def _(mo, rescue):
    mo.md(
        """
        ## 2. Extra loops rescue late positions

        Slower contracts begin length-64 evaluation before the frontier can
        reach the end. Giving them their speed-predicted budget produces large
        gains:
        """
    )
    mo.ui.table(rescue, selection=None)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Rescue is not monotone. The demand-2 model at fixed length 32 reaches
    0.996 token accuracy near its predicted 17 loops, but falls to 0.465 by
    32 loops. The measured frontier is therefore a **halting rule**, not a
    promise that arbitrary extra recurrence is safe.

    ## 3. The causal intervention is less stable

    Full-resampling patches were applied at selected loop-position
    coordinates. Upstream argmax damage was consistently zero, which is
    directionally compatible with prefix computation. But unit-speed cone
    slopes were only 0.07–0.35 versus frontier speed 0.98; demand-2 patches
    gave one compatible slope (2.05 versus 1.98) among twelve
    position-by-seed tests. Early patches were often overwritten and late
    patches damaged only the source position.

    ## What the fresh evidence says

    * **Aligned:** the S4 budget law through demand 2 and approximately unit
      speed when \(T=n\).
    * **Aligned with a warning:** extra loops rescue unfinished late
      positions near the predicted budget, then may cause overthinking.
    * **Partial:** widening improves the demand-4 frontier, but exact 4×
      delivery is not reliable.
    * **Inconclusive here:** a damage cone with a consistently compatible
      slope.

    The reduced A5 horizon extension reached q90 speed 0.61 and chance-like
    long-sequence performance, so it is not evidence for the paper's A5
    result. Exact author hyperparameters, paired rescue batches, and a
    larger preregistered patch sweep remain the most important next steps.
    """)
    return


if __name__ == "__main__":
    app.run()
