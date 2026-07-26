# Convergence Selection in Looped Transformers — Reproduction

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alphaXiv/when-does-recurrence-become-an-algorithm-converg/blob/main/notebooks/convergence_selection.py)

This repository reproduces the central claim of [arXiv:2607.20594, *When Does Recurrence Become an Algorithm?*](https://arxiv.org/abs/2607.20594): a weight-tied looped transformer learns a convergence speed set by training length divided by training loops, and that speed can prescribe useful extra computation.

**Assessment: partially reproduced.** On fresh S4 prefix-product runs, the paper’s scaling exponent **0.98 ± 0.04** became **0.969 ± 0.034** (log-space R² 0.998). At demands 0.5, 0.667, 1, and 2, measured speeds were **0.509, 0.710, 0.984, and 1.994**. A speed-predicted 17-loop halt rescued late length-32 positions from 0.047 to 0.996 accuracy, while transfer to unseen length 64 was contract- and seed-dependent. The requested 25% activation-damage cone was not observed.

The primary model is a reconstructed 0.44M-parameter, two-layer, width-128 S4 transformer trained through lengths 4/8/16/32 with four seeds per contract. The paper reports roughly 1.6M parameters for its small setting and provides no author code or complete hyperparameter table, so initialization, input reinjection, group encoding, and thresholds are documented substitutions. A 1.72M-parameter capacity series was added; a bounded A5 extension was inconclusive and is not primary evidence.

- [Tutorial-style detailed report](reports/convergence-selection/report.md)
- [Self-contained marimo notebook](notebooks/convergence_selection.py)
- [Frozen aggregate evidence](reports/convergence-selection/data/results.json)
- [Exact Molab notebook URL](https://molab.marimo.io/github/alphaXiv/when-does-recurrence-become-an-algorithm-converg/blob/main/notebooks/convergence_selection.py)

Formal experiments ran on Kubernetes using four NVIDIA RTX PRO 6000 Blackwell GPUs per run and a peak of 16 GPUs concurrently. The fresh campaign’s actual elapsed wall time was **1.376 hours** (2026-07-26 12:24:34–13:47:07 UTC). Every claimed measurement comes from a nonempty terminal log created after the recovery cutoff.

## Experiment log

The command column is copied verbatim from `orx exp status`. Raw run IDs and fuller lineage notes remain in `orx exp desc`.

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| `main` | Public report, figures, notebook, and reusable harness | Not run as an experiment (publication surface) | Presentation-only | No experiment |
| [`orx/fresh-s4-t-n-baseline`](https://github.com/alphaXiv/when-does-recurrence-become-an-algorithm-converg/tree/orx/fresh-s4-t-n-baseline) | Frozen recovery root | `python -m torch.distributed.run --standalone --nproc_per_node=4 train_repro.py` | Startup failed because the manifest did not evaluate the injected script; repaired on child branches | Kubernetes; 4× RTX PRO 6000 Blackwell; 20s |
| [`orx/fresh-s4-t-2n-demand-0-5`](https://github.com/alphaXiv/when-does-recurrence-become-an-algorithm-converg/tree/orx/fresh-s4-t-2n-demand-0-5) | Slow S4 endpoint, demand 0.5 | `python -m torch.distributed.run --standalone --nproc_per_node=4 train_repro.py` | All seeds competent; speed 0.509 vs paper 0.51 | Kubernetes; 4× RTX PRO 6000 Blackwell; 19m06s |
| [`orx/s4-t-1-5n-demand-two-thirds`](https://github.com/alphaXiv/when-does-recurrence-become-an-algorithm-converg/tree/orx/s4-t-1-5n-demand-two-thirds) | Bridge contract, demand 0.667 | `python -m torch.distributed.run --standalone --nproc_per_node=4 train_repro.py` | All seeds competent; speed 0.710; strongest unseen-length rescue | Kubernetes; 4× RTX PRO 6000 Blackwell; 10m15s |
| [`orx/fresh-s4-t-n-kubernetes-fix`](https://github.com/alphaXiv/when-does-recurrence-become-an-algorithm-converg/tree/orx/fresh-s4-t-n-kubernetes-fix) | Unit-speed S4 contract | `python -m torch.distributed.run --standalone --nproc_per_node=4 train_repro.py` | All seeds competent; speed 0.984 vs paper 1.00 | Kubernetes; 4× RTX PRO 6000 Blackwell; 8m06s |
| [`orx/fresh-s4-t-n-2-demand-2`](https://github.com/alphaXiv/when-does-recurrence-become-an-algorithm-converg/tree/orx/fresh-s4-t-n-2-demand-2) | Fast S4 endpoint, demand 2 | `python -m torch.distributed.run --standalone --nproc_per_node=4 train_repro.py` | All seeds competent; speed 1.994 vs paper 2.00 | Kubernetes; 4× RTX PRO 6000 Blackwell; 4m21s |
| [`orx/demand-2-fixed-length-n-32-rescue`](https://github.com/alphaXiv/when-does-recurrence-become-an-algorithm-converg/tree/orx/demand-2-fixed-length-n-32-rescue) | Predicted stopping and overthinking control | `python -m torch.distributed.run --standalone --nproc_per_node=4 train_repro.py` | Predicted 17 loops: late accuracy 0.996; too many loops collapse | Kubernetes; 4× RTX PRO 6000 Blackwell; 4m15s |
| [`orx/demand-4-medium-width-diagnostic`](https://github.com/alphaXiv/when-does-recurrence-become-an-algorithm-converg/tree/orx/demand-4-medium-width-diagnostic) | 1.72M-parameter high-demand capacity check | `python -m torch.distributed.run --standalone --nproc_per_node=4 train_repro.py` | Median speed 3.367 vs paper 4.00; one unstable seed | Kubernetes; 4× RTX PRO 6000 Blackwell; 5m18s |
| [`orx/demand-2-mid-prefix-damage-cone`](https://github.com/alphaXiv/when-does-recurrence-become-an-algorithm-converg/tree/orx/demand-2-mid-prefix-damage-cone) | Full-resampling causal placement diagnostic | `python -m torch.distributed.run --standalone --nproc_per_node=4 train_repro.py` | One of four seeds matched slope ≈2; three self-healed. Small 25% patches were all zero on the parent test | Kubernetes; 4× RTX PRO 6000 Blackwell; 7m29s |
| [`orx/a5-unit-speed-horizon-extension`](https://github.com/alphaXiv/when-does-recurrence-become-an-algorithm-converg/tree/orx/a5-unit-speed-horizon-extension) | Harder A5 extension | `python -m torch.distributed.run --standalone --nproc_per_node=4 train_repro.py` | Only one of four seeds completed the curriculum; inconclusive and excluded | Kubernetes; 4× RTX PRO 6000 Blackwell; 8m51s |

## Reproduce the harness

The formal run command is:

```bash
orx exp run <experiment-id> --backend k8s
```

The committed Kubernetes manifest requests four GPUs and executes:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 train_repro.py
```

For a cheap local structural check:

```bash
python train_repro.py --smoke
marimo check notebooks/convergence_selection.py
marimo edit notebooks/convergence_selection.py
```

Formal evidence is the terminal `ORX_FINAL_EVIDENCE_JSON` record. Figures are regenerated without external services:

```bash
python reports/convergence-selection/make_figures.py
```
