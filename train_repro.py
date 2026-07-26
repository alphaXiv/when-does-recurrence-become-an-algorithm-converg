#!/usr/bin/env python3
"""Fresh group-prefix-product reproduction for arXiv:2607.20594.

One torchrun worker trains one independent seed. Rank 0 combines the four
machine-readable summaries so the terminal log is the complete evidence record.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def permutation_is_even(perm: tuple[int, ...]) -> bool:
    inversions = sum(
        perm[i] > perm[j]
        for i in range(len(perm))
        for j in range(i + 1, len(perm))
    )
    return inversions % 2 == 0


def make_group_table(
    group: str, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Return S4 or A5 composition table with (a ∘ b)(x) = a(b(x))."""
    if group == "S4":
        perms = list(itertools.permutations(range(4)))
    elif group == "A5":
        perms = [
            perm
            for perm in itertools.permutations(range(5))
            if permutation_is_even(perm)
        ]
    else:
        raise ValueError(f"Unsupported group: {group}")
    index = {p: i for i, p in enumerate(perms)}
    table = torch.empty((len(perms), len(perms)), dtype=torch.long)
    for ia, a in enumerate(perms):
        for ib, b in enumerate(perms):
            table[ia, ib] = index[tuple(a[b[x]] for x in range(len(a)))]
    return table.to(device)


def prefix_targets(tokens: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    batch, length = tokens.shape
    out = torch.empty_like(tokens)
    acc = tokens[:, 0]
    out[:, 0] = acc
    for pos in range(1, length):
        acc = table[acc, tokens[:, pos]]
        out[:, pos] = acc
    return out


class TiedLoopedTransformer(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        d = cfg["d_model"]
        self.vocab_size = cfg["vocab_size"]
        self.embed = nn.Embedding(self.vocab_size, d)
        self.inject = nn.Linear(2 * d, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg["n_heads"],
            dim_feedforward=cfg["ff_multiplier"] * d,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.block = nn.TransformerEncoder(layer, num_layers=cfg["n_layers"])
        self.readout_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, self.vocab_size)
        # Start as a stable recurrent map while leaving input reinjection learnable.
        nn.init.zeros_(self.inject.bias)
        with torch.no_grad():
            self.inject.weight.zero_()
            self.inject.weight[:, :d].copy_(torch.eye(d))
            self.inject.weight[:, d:].copy_(0.1 * torch.eye(d))

    @staticmethod
    def causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
        )

    def step(self, hidden: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        z = self.inject(torch.cat((hidden, embedding), dim=-1))
        return self.block(z, mask=self.causal_mask(z.shape[1], z.device))

    def forward(
        self,
        tokens: torch.Tensor,
        loops: int,
        return_all: bool = False,
        patch: tuple[int, int, torch.Tensor, float] | None = None,
    ) -> torch.Tensor:
        embedding = self.embed(tokens)
        hidden = embedding
        logits = []
        for loop in range(1, loops + 1):
            hidden = self.step(hidden, embedding)
            if patch is not None and loop == patch[0]:
                _, position, donor_state, alpha = patch
                hidden = hidden.clone()
                hidden[:, position] = (
                    (1.0 - alpha) * hidden[:, position]
                    + alpha * donor_state[:, position]
                )
            if return_all:
                logits.append(self.head(self.readout_norm(hidden)))
        final = self.head(self.readout_norm(hidden))
        return torch.stack(logits) if return_all else final

    @torch.no_grad()
    def hidden_at(self, tokens: torch.Tensor, loops: int) -> torch.Tensor:
        embedding = self.embed(tokens)
        hidden = embedding
        for _ in range(loops):
            hidden = self.step(hidden, embedding)
        return hidden


def contract_loops(length: int, multiplier: float) -> int:
    return max(1, math.ceil(length * multiplier))


def sample_batch(
    batch_size: int,
    max_length: int,
    table: torch.Tensor,
    device: torch.device,
    fixed_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    length = fixed_length or random.randint(2, max_length)
    x = torch.randint(0, table.shape[0], (batch_size, length), device=device)
    return x, prefix_targets(x, table)


def fit_slope(xs: list[float], ys: list[float]) -> tuple[float, float]:
    if len(xs) < 3 or np.std(xs) < 1e-8:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(np.asarray(xs), np.asarray(ys), deg=1)
    pred = slope * np.asarray(xs) + intercept
    denom = float(np.sum((np.asarray(ys) - np.mean(ys)) ** 2))
    r2 = 1.0 - float(np.sum((np.asarray(ys) - pred) ** 2)) / max(denom, 1e-12)
    return float(slope), r2


def frontier_from_accuracy(acc: torch.Tensor, threshold: float) -> list[int]:
    """Contiguous correct-prefix length per loop, monotonized for fitting."""
    values = []
    for row in acc:
        good = row >= threshold
        bad = torch.where(~good)[0]
        values.append(int(bad[0]) if len(bad) else int(row.numel()))
    return [int(x) for x in np.maximum.accumulate(values)]


def measure_frontier(
    logits: torch.Tensor,
    target: torch.Tensor,
    thresholds: list[float],
) -> dict[str, Any]:
    acc = (logits.argmax(-1) == target.unsqueeze(0)).float().mean(1)
    length = target.shape[1]
    out: dict[str, Any] = {
        "position_accuracy_final": [round(float(x), 5) for x in acc[-1].cpu()],
    }
    for threshold in thresholds:
        frontier = frontier_from_accuracy(acc, threshold)
        xs, ys = [], []
        for loop, value in enumerate(frontier, start=1):
            if 0 < value < length:
                xs.append(float(loop))
                ys.append(float(value))
        slope, r2 = fit_slope(xs, ys)
        out[f"q{threshold:.2f}"] = {
            "speed": slope,
            "r2": r2,
            "frontier_by_loop": frontier,
        }
    # Solved-input convergence time, matching the paper's forward-only tau.
    pred = logits.argmax(-1)
    correct = pred == target.unsqueeze(0)
    stable = torch.flip(
        torch.cumprod(torch.flip(correct.to(torch.int8), dims=[0]), dim=0), dims=[0]
    ).bool()
    solved = correct[-1].all(1)
    tau_positions: list[float] = []
    if int(solved.sum()) >= 8:
        stable_solved = stable[:, solved]
        for pos in range(length):
            per_example = []
            for sample in range(stable_solved.shape[1]):
                hits = torch.where(stable_solved[:, sample, pos])[0]
                per_example.append(int(hits[0]) + 1 if len(hits) else logits.shape[0])
            tau_positions.append(float(np.median(per_example)))
    if len(tau_positions) >= 3:
        tau_slope, tau_r2 = fit_slope(
            list(range(1, length + 1)), tau_positions
        )
        tau_speed = 1.0 / tau_slope if tau_slope > 1e-8 else float("nan")
    else:
        tau_slope = tau_r2 = tau_speed = float("nan")
    out["tau"] = {
        "solved_inputs": int(solved.sum()),
        "median_by_position": tau_positions,
        "tau_per_position_slope": tau_slope,
        "speed": tau_speed,
        "r2": tau_r2,
    }
    return out


@torch.no_grad()
def eval_accuracy(
    model: TiedLoopedTransformer,
    table: torch.Tensor,
    length: int,
    loops: int,
    batch_size: int,
    seed: int,
) -> dict[str, float]:
    torch.manual_seed(seed)
    x, y = sample_batch(batch_size, length, table, x_device(model), fixed_length=length)
    pred = model(x, loops).argmax(-1)
    correct = pred == y
    return {
        "token_accuracy": float(correct.float().mean()),
        "late_quartile_accuracy": float(correct[:, 3 * length // 4 :].float().mean()),
        "exact_match": float(correct.all(1).float().mean()),
    }


def x_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def activation_damage(
    model: TiedLoopedTransformer,
    table: torch.Tensor,
    cfg: dict[str, Any],
    loops: int,
    seed: int,
) -> dict[str, Any]:
    """Bounded donor-state intervention; fit cone extent against loops remaining."""
    device = x_device(model)
    length = min(32, cfg["rescue_length"])
    batch = min(128, cfg["eval_batch_size"])
    torch.manual_seed(seed)
    x, y = sample_batch(batch, length, table, device, fixed_length=length)
    donor_x, _ = sample_batch(batch, length, table, device, fixed_length=length)
    clean = model(x, loops).argmax(-1)
    position = 1
    patch_loops = sorted(set(max(1, int(loops * f)) for f in (0.2, 0.4, 0.6, 0.8)))
    rows = []
    for patch_loop in patch_loops:
        donor = model.hidden_at(donor_x, patch_loop)
        patched = model(
            x,
            loops,
            patch=(patch_loop, position, donor, cfg["patch_alpha"]),
        ).argmax(-1)
        damage = (patched != clean).float().mean(0)
        active = torch.where(damage >= 0.05)[0]
        extent = int(active.max()) - position if len(active) else 0
        upstream = float(damage[:position].mean()) if position else 0.0
        rows.append(
            {
                "patch_loop": patch_loop,
                "loops_remaining": loops - patch_loop,
                "extent": extent,
                "upstream_damage": upstream,
                "damage_by_position": [round(float(z), 5) for z in damage.cpu()],
            }
        )
    slope, r2 = fit_slope(
        [r["loops_remaining"] for r in rows], [r["extent"] for r in rows]
    )
    return {
        "alpha": cfg["patch_alpha"],
        "source_position_zero_based": position,
        "cone_speed": slope,
        "cone_r2": r2,
        "rows": rows,
    }


def train_one(cfg: dict[str, Any], rank: int) -> dict[str, Any]:
    cfg = dict(cfg)
    seed = int(cfg["seeds"][rank])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    table = make_group_table(cfg["group"], device)
    cfg["vocab_size"] = int(table.shape[0])
    model = TiedLoopedTransformer(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scaler = None
    started = time.monotonic()
    global_step = 0
    training_trace = []
    stage_summaries = []
    for stage, max_length in enumerate(cfg["stage_max_lengths"], start=1):
        model.train()
        interval_loss = 0.0
        interval_acc = 0.0
        promoted = False
        final_interval_acc = 0.0
        for local_step in range(1, cfg["max_steps_per_stage"] + 1):
            x, y = sample_batch(cfg["batch_size"], max_length, table, device)
            loops = contract_loops(x.shape[1], cfg["contract_multiplier"])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x, loops)
                if cfg.get("horizon_curriculum", False):
                    horizon = min(
                        x.shape[1],
                        max(
                            2,
                            math.ceil(
                                4
                                + (max_length - 4)
                                * local_step
                                / cfg["max_steps_per_stage"]
                            ),
                        ),
                    )
                else:
                    horizon = x.shape[1]
                supervised_logits = logits[:, :horizon]
                supervised_y = y[:, :horizon]
                loss = F.cross_entropy(
                    supervised_logits.reshape(-1, cfg["vocab_size"]),
                    supervised_y.reshape(-1),
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            global_step += 1
            interval_loss += float(loss)
            interval_acc += float(
                (supervised_logits.argmax(-1) == supervised_y).float().mean()
            )
            if local_step % 500 == 0:
                final_interval_acc = interval_acc / 500
                record = {
                    "rank": rank,
                    "seed": seed,
                    "stage": stage,
                    "max_length": max_length,
                    "global_step": global_step,
                    "loss": interval_loss / 500,
                    "token_accuracy": final_interval_acc,
                }
                training_trace.append(record)
                print("TRAIN_JSON " + json.dumps(record, sort_keys=True), flush=True)
                interval_loss = interval_acc = 0.0
                if (
                    local_step >= cfg["min_steps_per_stage"]
                    and final_interval_acc >= cfg["promotion_accuracy"]
                ):
                    promoted = True
                    break
        stage_summaries.append(
            {
                "stage": stage,
                "max_length": max_length,
                "steps": local_step,
                "promoted_at_threshold": promoted,
                "final_interval_accuracy": final_interval_acc,
            }
        )

    model.eval()
    thresholds = [float(q) for q in cfg["frontier_thresholds"]]
    frontier: dict[str, Any] = {}
    for length in cfg["eval_lengths"]:
        max_loops = max(
            length,
            math.ceil(2.5 * contract_loops(length, cfg["contract_multiplier"])),
        )
        torch.manual_seed(seed + 20_000 + length)
        x, y = sample_batch(
            cfg["eval_batch_size"], length, table, device, fixed_length=length
        )
        with torch.no_grad():
            all_logits = model(x, max_loops, return_all=True)
        frontier[str(length)] = {
            "max_eval_loops": max_loops,
            **measure_frontier(all_logits, y, thresholds),
        }
        del all_logits

    rescue_length = int(cfg["rescue_length"])
    train_budget = contract_loops(rescue_length, cfg["contract_multiplier"])
    loop_grid = sorted(
        set(
            max(1, int(round(train_budget * factor)))
            for factor in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5)
        )
    )
    rescue = {
        str(loops): eval_accuracy(
            model,
            table,
            rescue_length,
            loops,
            cfg["eval_batch_size"],
            seed + 30_000 + loops,
        )
        for loops in loop_grid
    }

    id_speeds = []
    for length in (16, 32):
        speed = frontier[str(length)]["q0.90"]["speed"]
        if math.isfinite(speed) and speed > 0:
            id_speeds.append(speed)
    id_speed = float(np.median(id_speeds)) if id_speeds else float("nan")
    predicted_loops = (
        max(1, math.ceil(rescue_length / id_speed))
        if math.isfinite(id_speed) and id_speed > 0
        else train_budget
    )
    predicted = eval_accuracy(
        model,
        table,
        rescue_length,
        predicted_loops,
        cfg["eval_batch_size"],
        seed + 40_000,
    )
    damage_loops = max(train_budget, predicted_loops)
    damage = activation_damage(model, table, cfg, damage_loops, seed + 50_000)
    elapsed = (time.monotonic() - started) / 3600
    return {
        "evidence_version": 1,
        "paper_id": "2607.20594",
        "group": cfg["group"],
        "backend": "kubernetes",
        "gpu_model": "NVIDIA RTX PRO 6000 Blackwell",
        "rank": rank,
        "seed": seed,
        "contract_multiplier_T_over_n": cfg["contract_multiplier"],
        "contract_demand_n_over_T": 1.0 / cfg["contract_multiplier"],
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "config": cfg,
        "training_trace": training_trace,
        "stage_summaries": stage_summaries,
        "frontier": frontier,
        "id_speed_q0.90": id_speed,
        "rescue_length": rescue_length,
        "training_budget_at_rescue_length": train_budget,
        "rescue": rescue,
        "predicted_halting_loops": predicted_loops,
        "predicted_halting_accuracy": predicted,
        "activation_damage": damage,
        "elapsed_hours": elapsed,
        "finished_utc": utc_now(),
    }


def smoke() -> None:
    table = make_group_table("S4")
    identity = next(
        i for i, p in enumerate(itertools.permutations(range(4))) if p == (0, 1, 2, 3)
    )
    assert torch.equal(table[identity], torch.arange(24))
    x = torch.tensor([[identity, 1, 2]])
    y = prefix_targets(x, table)
    assert y.shape == x.shape
    print("smoke-ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    cfg = json.loads(Path("config.json").read_text())
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != 4:
        raise RuntimeError(f"Formal run requires exactly 4 torchrun workers, got {world}")
    dist.init_process_group("gloo")
    if rank == 0:
        print(
            "RUN_CONTRACT_JSON "
            + json.dumps(
                {
                    "started_utc": utc_now(),
                    "backend": "kubernetes",
                    "gpu_model": "NVIDIA RTX PRO 6000 Blackwell",
                    "allocated_gpus": world,
                    "config": cfg,
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    result = train_one(cfg, rank)
    result_path = Path(f"/tmp/orx_result_rank_{rank}.json")
    result_path.write_text(json.dumps(result, allow_nan=True))
    dist.barrier()
    if rank == 0:
        combined = [
            json.loads(Path(f"/tmp/orx_result_rank_{r}.json").read_text())
            for r in range(world)
        ]
        speeds = [
            row["id_speed_q0.90"]
            for row in combined
            if math.isfinite(row["id_speed_q0.90"])
        ]
        summary = {
            "paper_id": "2607.20594",
            "backend": "kubernetes",
            "gpu_model": "NVIDIA RTX PRO 6000 Blackwell",
            "allocated_gpus": world,
            "contract_multiplier_T_over_n": cfg["contract_multiplier"],
            "contract_demand_n_over_T": 1.0 / cfg["contract_multiplier"],
            "seeds": cfg["seeds"],
            "median_id_speed_q0.90": float(np.median(speeds)) if speeds else None,
            "per_seed": combined,
            "finished_utc": utc_now(),
        }
        print("ORX_FINAL_EVIDENCE_JSON " + json.dumps(summary, allow_nan=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
