"""Empirical basin-of-attraction study for simulated annealing on a fixed discrete landscape.

The current concrete instantiation uses the custom fixed f5 landscape from the
Kirkpatrick module, but the core experiment is generic over:

- a finite list of start states
- a fixed energy landscape
- a proposal kernel
- a success criterion
"""

from __future__ import annotations

import argparse
import math
import os
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_DIR))

MPL_CONFIG_DIR = PAPER_DIR / "benchmarks" / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(PAPER_DIR / "benchmarks" / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark_simulated_annealing_f5_landscape import (  # noqa: E402
    FixedLandscape,
    build_fixed_landscape,
    run_sa,
    save_csv,
)


@dataclass(frozen=True)
class BasinSummary:
    initial_state: int
    runs_per_start: int
    hit_probability_exact: float
    hit_probability_epsilon: float
    mean_best_energy: float
    stderr_best_energy: float
    mean_gap_to_global_minimum: float
    stderr_gap_to_global_minimum: float
    mean_best_state: float


def clamp_runs_per_start(value: int) -> int:
    return max(10, min(150, value))


def choose_start_states(M: int, start_step: int) -> list[int]:
    if start_step <= 0:
        raise ValueError("start-step must be positive")
    starts = list(range(0, M, start_step))
    if starts[-1] != M - 1:
        starts.append(M - 1)
    return starts


def energy_hit(best_energy: float, global_minimum_energy: float, epsilon: float) -> bool:
    return best_energy <= global_minimum_energy + epsilon


def summarize_basin(
    *,
    landscape: FixedLandscape,
    initial_state: int,
    runs_per_start: int,
    initial_temperature: float,
    alpha: float,
    steps: int,
    seed_offset: int,
    success_epsilon: float,
) -> BasinSummary:
    best_energies: list[float] = []
    best_states: list[int] = []
    exact_hits = 0
    epsilon_hits = 0
    global_minimum_state = landscape.global_minimum_state()
    global_minimum_energy = landscape.global_minimum_energy()

    for run_index in range(runs_per_start):
        seed = seed_offset + run_index
        result = run_sa(
            landscape=landscape,
            initial_state=initial_state,
            steps=steps,
            seed=seed,
            initial_temperature=initial_temperature,
            alpha=alpha,
        )
        best_energies.append(result.best_energy)
        best_states.append(result.best_state)
        if result.best_state == global_minimum_state:
            exact_hits += 1
        if energy_hit(result.best_energy, global_minimum_energy, success_epsilon):
            epsilon_hits += 1

    gaps = [energy - global_minimum_energy for energy in best_energies]
    return BasinSummary(
        initial_state=initial_state,
        runs_per_start=runs_per_start,
        hit_probability_exact=exact_hits / runs_per_start,
        hit_probability_epsilon=epsilon_hits / runs_per_start,
        mean_best_energy=statistics.fmean(best_energies),
        stderr_best_energy=statistics.pstdev(best_energies) / math.sqrt(runs_per_start) if runs_per_start > 1 else 0.0,
        mean_gap_to_global_minimum=statistics.fmean(gaps),
        stderr_gap_to_global_minimum=statistics.pstdev(gaps) / math.sqrt(runs_per_start) if runs_per_start > 1 else 0.0,
        mean_best_state=statistics.fmean(best_states),
    )


def plot_hit_probability(rows: list[BasinSummary], output_dir: Path) -> None:
    starts = [row.initial_state for row in rows]
    exact = [row.hit_probability_exact for row in rows]
    eps = [row.hit_probability_epsilon for row in rows]

    plt.figure(figsize=(10, 5))
    plt.plot(starts, exact, "o-", label="Exact global-minimum hit probability", markersize=4)
    plt.plot(starts, eps, "o-", label="Epsilon-hit probability", markersize=4)
    plt.xlabel("Initial state")
    plt.ylabel("Probability")
    plt.title("Empirical SA Basin: Hit Probability vs Initial State")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "hit_probability_vs_initial_state.png", dpi=160)
    plt.close()


def plot_gap(rows: list[BasinSummary], output_dir: Path) -> None:
    starts = [row.initial_state for row in rows]
    mean_gap = [row.mean_gap_to_global_minimum for row in rows]
    stderr_gap = [row.stderr_gap_to_global_minimum for row in rows]

    plt.figure(figsize=(10, 5))
    plt.errorbar(starts, mean_gap, yerr=stderr_gap, fmt="o-", capsize=4, markersize=4)
    plt.xlabel("Initial state")
    plt.ylabel("Mean gap to global minimum")
    plt.title("Empirical SA Basin: Mean Gap vs Initial State")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "mean_gap_vs_initial_state.png", dpi=160)
    plt.close()


def plot_best_energy(rows: list[BasinSummary], global_minimum_energy: float, output_dir: Path) -> None:
    starts = [row.initial_state for row in rows]
    means = [row.mean_best_energy for row in rows]
    stderrs = [row.stderr_best_energy for row in rows]

    plt.figure(figsize=(10, 5))
    plt.errorbar(starts, means, yerr=stderrs, fmt="o-", capsize=4, markersize=4)
    plt.axhline(global_minimum_energy, linestyle="--", color="black", alpha=0.7, label="Global minimum energy")
    plt.xlabel("Initial state")
    plt.ylabel("Mean best energy")
    plt.title("Empirical SA Basin: Mean Best Energy vs Initial State")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "mean_best_energy_vs_initial_state.png", dpi=160)
    plt.close()


def plot_landscape_with_basin(landscape: FixedLandscape, rows: list[BasinSummary], output_dir: Path) -> None:
    xs = list(range(landscape.M))
    plt.figure(figsize=(11, 5))
    plt.plot(xs, landscape.base_values, label="Base f5 landscape", linewidth=1.4, alpha=0.7)
    plt.plot(xs, landscape.total_values, label="Frozen landscape realization", linewidth=1.6, color="tab:red")
    plt.axvline(landscape.R, linestyle=":", color="black", alpha=0.5, label=f"R={landscape.R}")

    starts = [row.initial_state for row in rows]
    hit_prob = [row.hit_probability_exact for row in rows]
    hit_energy = [landscape.energy(x) for x in starts]
    scatter = plt.scatter(starts, hit_energy, c=hit_prob, cmap="viridis", s=55, edgecolors="black", linewidths=0.3)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Exact hit probability")

    plt.xlabel("Initial state")
    plt.ylabel("Energy")
    plt.title("Frozen f5 Landscape with Empirical SA Basin Overlay")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "frozen_landscape_with_empirical_basin.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Empirical basin-of-attraction study for SA on a fixed f5 landscape.")
    parser.add_argument("--M", type=int, default=120, help="Number of states in {0, ..., M-1}.")
    parser.add_argument("--k", type=int, default=20, help="Block-width parameter. Must divide M.")
    parser.add_argument("--c", type=float, default=0.4, help="Ruggedness scale in the f5 family.")
    parser.add_argument("--R", type=int, default=60, help="Shift parameter.")
    parser.add_argument("--steps", type=int, default=8000, help="Iterations per SA run.")
    parser.add_argument("--initial-temperature", type=float, default=4000.0, help="Initial SA temperature.")
    parser.add_argument("--alpha", type=float, default=0.98, help="Geometric cooling factor.")
    parser.add_argument(
        "--runs-per-start",
        type=int,
        default=30,
        help="Requested number of SA runs per initial state. Clamped into [10, 150].",
    )
    parser.add_argument("--start-step", type=int, default=1, help="Evaluate initial states 0, start-step, 2*start-step, ...")
    parser.add_argument(
        "--success-epsilon",
        type=float,
        default=0.0,
        help="Energy tolerance above the global minimum that still counts as a success.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Base seed for the frozen landscape and SA runs.")
    parser.add_argument("--gaussian-noise", action="store_true", help="Add one fixed Gaussian perturbation to the landscape.")
    parser.add_argument("--noise-mean", type=float, default=None, help="Mean of the Gaussian perturbation if enabled.")
    parser.add_argument("--noise-sd", type=float, default=None, help="Standard deviation of the Gaussian perturbation if enabled.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "f5_empirical_basin_outputs",
        help="Directory for CSV and PNG outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.M <= 1:
        raise ValueError("M must be greater than 1")
    if args.k <= 0 or args.M % args.k != 0:
        raise ValueError("k must be positive and divide M")
    if not 0 <= args.R < args.M:
        raise ValueError("R must lie in [0, M-1]")
    if args.steps < 0:
        raise ValueError("steps must be non-negative")
    if args.initial_temperature <= 0.0:
        raise ValueError("initial-temperature must be positive")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if args.success_epsilon < 0.0:
        raise ValueError("success-epsilon must be non-negative")
    if args.gaussian_noise and (args.noise_mean is None or args.noise_sd is None):
        raise ValueError("--noise-mean and --noise-sd are required when --gaussian-noise is set")
    if not args.gaussian_noise:
        args.noise_mean = 0.0
        args.noise_sd = 0.0
    if args.noise_sd < 0.0:
        raise ValueError("noise-sd must be non-negative")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_per_start = clamp_runs_per_start(args.runs_per_start)
    starts = choose_start_states(args.M, args.start_step)
    landscape = build_fixed_landscape(
        M=args.M,
        k=args.k,
        c=args.c,
        R=args.R,
        gaussian_noise=args.gaussian_noise,
        noise_mean=args.noise_mean,
        noise_sd=args.noise_sd,
        seed=args.seed,
    )

    summaries: list[BasinSummary] = []
    for index, initial_state in enumerate(starts):
        summaries.append(
            summarize_basin(
                landscape=landscape,
                initial_state=initial_state,
                runs_per_start=runs_per_start,
                initial_temperature=args.initial_temperature,
                alpha=args.alpha,
                steps=args.steps,
                seed_offset=args.seed + 100000 + index * 1000,
                success_epsilon=args.success_epsilon,
            )
        )

    rows = [
        {
            "initial_state": row.initial_state,
            "runs_per_start": row.runs_per_start,
            "hit_probability_exact": row.hit_probability_exact,
            "hit_probability_epsilon": row.hit_probability_epsilon,
            "mean_best_energy": row.mean_best_energy,
            "stderr_best_energy": row.stderr_best_energy,
            "mean_gap_to_global_minimum": row.mean_gap_to_global_minimum,
            "stderr_gap_to_global_minimum": row.stderr_gap_to_global_minimum,
            "mean_best_state": row.mean_best_state,
            "global_minimum_state": landscape.global_minimum_state(),
            "global_minimum_energy": landscape.global_minimum_energy(),
        }
        for row in summaries
    ]
    save_csv(output_dir / "empirical_basin_results.csv", rows)
    save_csv(
        output_dir / "empirical_basin_config.csv",
        [
            {
                "M": args.M,
                "k": args.k,
                "c": args.c,
                "R": args.R,
                "steps": args.steps,
                "initial_temperature": args.initial_temperature,
                "alpha": args.alpha,
                "requested_runs_per_start": args.runs_per_start,
                "effective_runs_per_start": runs_per_start,
                "start_step": args.start_step,
                "success_epsilon": args.success_epsilon,
                "seed": args.seed,
                "gaussian_noise": args.gaussian_noise,
                "noise_mean": args.noise_mean,
                "noise_sd": args.noise_sd,
                "global_minimum_state": landscape.global_minimum_state(),
                "global_minimum_energy": landscape.global_minimum_energy(),
            }
        ],
    )

    plot_hit_probability(summaries, output_dir)
    plot_gap(summaries, output_dir)
    plot_best_energy(summaries, landscape.global_minimum_energy(), output_dir)
    plot_landscape_with_basin(landscape, summaries, output_dir)

    mean_exact_hit = statistics.fmean(row.hit_probability_exact for row in summaries)
    mean_gap = statistics.fmean(row.mean_gap_to_global_minimum for row in summaries)
    print("effective_runs_per_start\tmean_exact_hit_probability\tmean_gap_across_starts")
    print(f"{runs_per_start}\t{mean_exact_hit:.6f}\t{mean_gap:.6f}")


if __name__ == "__main__":
    main()
