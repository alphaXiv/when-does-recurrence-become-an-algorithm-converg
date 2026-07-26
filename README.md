# Convergence Selection in Looped Transformers — Fresh Reproduction

This repository reproduces the central claim of [*When Does Recurrence Become an Algorithm?* (arXiv:2607.20594)](https://arxiv.org/abs/2607.20594): a weight-tied transformer trained with \(T\) loops on length \(n\) learns a convergence speed near \(n/T\), and that speed predicts how many extra loops can finish longer sequences.

**Assessment: partially reproduced.** On synthetic S4 prefix products, the compact model followed demand 0.5–2 with exponent **0.968** (\(R^2=0.998\)), compared with the paper's **0.98 ± 0.04** (\(R^2=0.99\)); the \(T=n\) contract measured **0.991** across eight seeds. Extra loops rescued length-64 token accuracy from **0.56 to 0.88** at demand 0.5 and **0.58 to 0.95** at demand 0.67. The compact demand-4 model reached only 1.77, a wider model reached 3.37, and a frontier-compatible activation-damage slope appeared in only 1 of 12 tests.

The primary evidence is downscaled to two-layer, 0.44M- and 1.72M-parameter models trained through length 32; exact optimizer, encoding, curriculum, and convergence details were reconstructed because author code was unavailable. A reduced A5 extension did not solve long sequences and is reported as a negative extension, not replacement evidence.

All fresh formal runs used **Kubernetes** on **NVIDIA RTX PRO 6000 Blackwell** GPUs, four GPUs per experiment and **16 GPUs at peak concurrency**. The successful compute campaign spanned **1.243 wall hours** from the first successful contract start to the final successful run.

## Read and explore

- [Tutorial-style scientific report](reports/convergence-budget/report.md)
- [Self-contained marimo notebook](notebooks/convergence_budget_reproduction.py)
- [Figure-generation source](scripts/make_figures.py)
- Local notebook: `marimo edit notebooks/convergence_budget_reproduction.py`

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alphaXiv/when-does-recurrence-become-an-algorithm-converg/blob/main/notebooks/convergence_budget_reproduction.py)

## Primary result

![Measured convergence speed follows training demand](reports/convergence-budget/images/budget_law.svg)

The dashed line is exact delivery. Both widths follow it through demand 2; demand 4 reveals a capacity boundary.

## Experiment log

The exact fixed command copied from `orx exp status` is shown for every formal branch. `main` was **not run as an experiment (publication surface)**.

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| `main` | Polished public report, notebook, and canonical implementation | Not run as an experiment (publication surface) | Publication surface only | — |
| [S4 demand 0.5](https://github.com/alphaXiv/when-does-recurrence-become-an-algorithm-converg/tree/orx/fresh-s4-t-2n-demand-0-5) | Compact model, \(T=2n\), four seeds | `python -m torch.distributed.run --standalone --nproc_per_node=4 train_repro.py` | Speed 0.509; strong late-position rescue | Kubernetes, 4× RTX PRO 6000 Blackwell |
| [S4 demand 0.67](https://github.com/alphaXiv/when-does-recurrence-become-an-algorithm-converg/tree/orx/s4-t-1-5n-demand-two-thirds) | Compact model, \(T=1.5n\), four seeds | `python -m torch.distributed.run --standalone --nproc_per_node=4 train_repro.py` | Speed 0.710; rescue to 0.953 | Kubernetes, 4× RTX PRO 6000 Blackwell |
| [S4 demand 1](https://github.com/alphaXiv/when-does-recurrence-become-an-algorithm-converg/tree/orx/fresh-s4-t-n-kubernetes-fix) | Compact model, \(T=n\), four seeds | `python -m torch.distributed.run --standalone --nproc_per_node=4 train_repro.py` | Speed 0.984; independent replication gives combined 0.991 | Kubernetes, 4× RTX PRO 6000 Blackwell |
| [S4 demand 2](https://github.com/alphaXiv/when-does-recurrence-become-an-algorithm-converg/tree/orx/fresh-s4-t-n-2-demand-2) | Compact model, \(T=n/2\), four seeds | `python -m torch.distributed.run --standalone --nproc_per_node=4 train_repro.py` | Speed 2.00; length-64 overthinking | Kubernetes, 4× RTX PRO 6000 Blackwell |
| [Fixed-length rescue](https://github.com/alphaXiv/when-does-recurrence-become-an-algorithm-converg/tree/orx/demand-2-fixed-length-n-32-rescue) | Demand 2 evaluated around the predicted length-32 boundary | `python -m torch.distributed.run --standalone --nproc_per_node=4 train_repro.py` | Accuracy 0.548→0.996 near 17 loops, then 0.465 at 32 | Kubernetes, 4× RTX PRO 6000 Blackwell |
| [S4 demand 4](https://github.com/alphaXiv/when-does-recurrence-become-an-algorithm-converg/tree/orx/fresh-s4-t-n-4-demand-4) | Compact model, \(T=n/4\) | `python -m torch.distributed.run --standalone --nproc_per_node=4 train_repro.py` | Speed 1.77: capacity shortfall | Kubernetes, 4× RTX PRO 6000 Blackwell |
| [Medium demand 4](https://github.com/alphaXiv/when-does-recurrence-become-an-algorithm-converg/tree/orx/demand-4-medium-width-diagnostic) | Double width to 256 / eight heads | `python -m torch.distributed.run --standalone --nproc_per_node=4 train_repro.py` | Speed 3.37; partial recovery | Kubernetes, 4× RTX PRO 6000 Blackwell |
| [Medium demand 0.5](https://github.com/alphaXiv/when-does-recurrence-become-an-algorithm-converg/tree/orx/medium-width-demand-0-5-2) | Complete the wide-model scaling matrix | `python -m torch.distributed.run --standalone --nproc_per_node=4 train_repro.py` | Median speed 0.516; 3/4 seeds track 0.5 | Kubernetes, 4× RTX PRO 6000 Blackwell |
| [Unit damage cone](https://github.com/alphaXiv/when-does-recurrence-become-an-algorithm-converg/tree/orx/unit-speed-full-activation-resampling) | Full-resampling activation patch, source position 1 | `python -m torch.distributed.run --standalone --nproc_per_node=4 train_repro.py` | Directed but shallow slopes 0.07–0.35 vs 0.98 | Kubernetes, 4× RTX PRO 6000 Blackwell |
| [Demand-2 mid-prefix cone](https://github.com/alphaXiv/when-does-recurrence-become-an-algorithm-converg/tree/orx/demand-2-mid-prefix-damage-cone) | Patch source position 8 across loop times | `python -m torch.distributed.run --standalone --nproc_per_node=4 train_repro.py` | One compatible 2.05 slope; 3/4 absent/shallow | Kubernetes, 4× RTX PRO 6000 Blackwell |
| [A5 extension](https://github.com/alphaXiv/when-does-recurrence-become-an-algorithm-converg/tree/orx/a5-unit-speed-horizon-extension) | Harder alternating-group extension | `python -m torch.distributed.run --standalone --nproc_per_node=4 train_repro.py` | q90 speed 0.61; OOD near chance | Kubernetes, 4× RTX PRO 6000 Blackwell |

## Reproduce the implementation

The fixed experiment entrypoint is:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 train_repro.py
```

Each experiment branch changes committed `config.json`, never command-line knobs. The Kubernetes manifest allocates four GPUs and launches one independent seed per distributed rank. The formal evidence is stored in terminal logs as `ORX_FINAL_EVIDENCE_JSON`; the public notebook embeds the decisive summaries so readers do not need access to the OpenResearch run database or expensive retraining.

See the [report](reports/convergence-budget/report.md) for methods, five evidence-bearing figures, claim-by-claim assessments, and limitations.
