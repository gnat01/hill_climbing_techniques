"""Compare greedy hill climbing and simulated annealing over many starts on one fixed f5 landscape."""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import statistics
import sys
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
    build_fixed_landscape,
    run_hill_climb,
    run_sa,
    save_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-start SA vs greedy benchmark on a fixed f5 landscape.")
    parser.add_argument("--M", type=int, default=120, help="Number of states in {0, ..., M-1}.")
    parser.add_argument("--k", type=int, default=20, help="Block-width parameter. Must divide M.")
    parser.add_argument("--c", type=float, default=0.4, help="Ruggedness scale in the f5 family.")
    parser.add_argument("--R", type=int, default=60, help="Shift parameter.")
    parser.add_argument("--steps", type=int, default=800, help="Iterations per run.")
    parser.add_argument("--initial-temperature", type=float, default=40.0, help="Initial SA temperature.")
    parser.add_argument("--alpha", type=float, default=0.98, help="Geometric cooling factor for SA.")
    parser.add_argument("--num-starts", type=int, default=100, help="Number of distinct starting states.")
    parser.add_argument("--seed", type=int, default=123, help="Base seed for the fixed landscape and start selection.")
    parser.add_argument("--gaussian-noise", action="store_true", help="Add one fixed Gaussian perturbation to the landscape.")
    parser.add_argument("--noise-mean", type=float, default=None, help="Mean of the Gaussian perturbation if enabled.")
    parser.add_argument("--noise-sd", type=float, default=None, help="Standard deviation of the Gaussian perturbation if enabled.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "f5_multi_start_outputs",
        help="Directory for CSV and PNG outputs.",
    )
    return parser.parse_args()


def choose_start_states(M: int, num_starts: int, seed: int) -> list[int]:
    if num_starts > M:
        raise ValueError("num-starts cannot exceed M when distinct starts are required")
    rng = random.Random(seed)
    starts = list(range(M))
    rng.shuffle(starts)
    return sorted(starts[:num_starts])


def plot_best_energy_vs_start(rows: list[dict[str, float | int]], global_minimum_energy: float, output_dir: Path) -> None:
    starts = [row["initial_state"] for row in rows]
    greedy_best = [row["greedy_best_energy"] for row in rows]
    sa_best = [row["sa_best_energy"] for row in rows]

    plt.figure(figsize=(10, 5))
    plt.plot(starts, greedy_best, "o-", label="Greedy best energy", markersize=4)
    plt.plot(starts, sa_best, "o-", label="SA best energy", markersize=4)
    plt.axhline(global_minimum_energy, color="black", linestyle="--", alpha=0.7, label="Global minimum energy")
    plt.xlabel("Initial state")
    plt.ylabel("Best energy reached")
    plt.title("Best Energy Reached vs Initial State")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "best_energy_vs_initial_state.png", dpi=160)
    plt.close()


def plot_gap_vs_start(rows: list[dict[str, float | int]], output_dir: Path) -> None:
    starts = [row["initial_state"] for row in rows]
    greedy_gap = [row["greedy_gap_to_global_minimum"] for row in rows]
    sa_gap = [row["sa_gap_to_global_minimum"] for row in rows]

    plt.figure(figsize=(10, 5))
    plt.plot(starts, greedy_gap, "o-", label="Greedy gap", markersize=4)
    plt.plot(starts, sa_gap, "o-", label="SA gap", markersize=4)
    plt.xlabel("Initial state")
    plt.ylabel("Gap to global minimum")
    plt.title("Gap to Global Minimum vs Initial State")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "gap_to_global_minimum_vs_initial_state.png", dpi=160)
    plt.close()


def plot_sa_advantage(rows: list[dict[str, float | int]], output_dir: Path) -> None:
    starts = [row["initial_state"] for row in rows]
    advantage = [row["greedy_best_energy"] - row["sa_best_energy"] for row in rows]

    plt.figure(figsize=(10, 5))
    plt.axhline(0.0, color="black", linestyle="--", alpha=0.7)
    plt.plot(starts, advantage, "o-", markersize=4, color="tab:green")
    plt.xlabel("Initial state")
    plt.ylabel("Greedy best energy - SA best energy")
    plt.title("SA Advantage by Initial State")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "sa_advantage_vs_initial_state.png", dpi=160)
    plt.close()


def plot_summary(summary_rows: list[dict[str, float | str]], output_dir: Path) -> None:
    labels = [row["algorithm"] for row in summary_rows]
    x = list(range(len(labels)))

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    axes[0].errorbar(
        x,
        [row["mean_best_energy"] for row in summary_rows],
        yerr=[row["stderr_best_energy"] for row in summary_rows],
        fmt="o",
        capsize=5,
    )
    axes[0].set_ylabel("Mean best energy")
    axes[0].set_title("Mean Best Energy Across Starts")
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(
        x,
        [row["mean_gap_to_global_minimum"] for row in summary_rows],
        yerr=[row["stderr_gap_to_global_minimum"] for row in summary_rows],
        fmt="o",
        capsize=5,
        color="tab:red",
    )
    axes[1].set_ylabel("Mean gap")
    axes[1].set_title("Mean Gap to Global Minimum Across Starts")
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(
        x,
        [row["hit_rate_global_minimum"] for row in summary_rows],
        color=["tab:blue", "tab:orange"],
        alpha=0.7,
    )
    axes[2].set_xticks(x, labels)
    axes[2].set_ylabel("Hit rate")
    axes[2].set_title("Global Minimum Hit Rate Across Starts")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "multi_start_summary.png", dpi=160)
    plt.close(fig)


def summarize(rows: list[dict[str, float | int]], global_minimum_state: int, global_minimum_energy: float) -> list[dict[str, float | str]]:
    summary_rows: list[dict[str, float | str]] = []
    for prefix, label in [("greedy", "greedy"), ("sa", "simulated_annealing")]:
        best_energies = [row[f"{prefix}_best_energy"] for row in rows]
        gaps = [row[f"{prefix}_gap_to_global_minimum"] for row in rows]
        hit_rate = statistics.fmean(
            [1.0 if row[f"{prefix}_best_state"] == global_minimum_state else 0.0 for row in rows]
        )
        summary_rows.append(
            {
                "algorithm": label,
                "mean_best_energy": statistics.fmean(best_energies),
                "stderr_best_energy": statistics.pstdev(best_energies) / math.sqrt(len(best_energies))
                if len(best_energies) > 1
                else 0.0,
                "mean_gap_to_global_minimum": statistics.fmean(gaps),
                "stderr_gap_to_global_minimum": statistics.pstdev(gaps) / math.sqrt(len(gaps))
                if len(gaps) > 1
                else 0.0,
                "hit_rate_global_minimum": hit_rate,
                "global_minimum_state": global_minimum_state,
                "global_minimum_energy": global_minimum_energy,
            }
        )
    return summary_rows


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
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if args.initial_temperature <= 0.0:
        raise ValueError("initial-temperature must be positive")
    if not 1 <= args.num_starts <= args.M:
        raise ValueError("num-starts must lie in [1, M]")
    if args.gaussian_noise and (args.noise_mean is None or args.noise_sd is None):
        raise ValueError("--noise-mean and --noise-sd are required when --gaussian-noise is set")
    if not args.gaussian_noise:
        args.noise_mean = 0.0
        args.noise_sd = 0.0
    if args.noise_sd < 0.0:
        raise ValueError("noise-sd must be non-negative")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

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
    starts = choose_start_states(args.M, args.num_starts, args.seed + 500)
    global_minimum_state = landscape.global_minimum_state()
    global_minimum_energy = landscape.global_minimum_energy()

    rows: list[dict[str, float | int]] = []
    for index, initial_state in enumerate(starts):
        greedy_seed = args.seed + 10000 + index
        sa_seed = args.seed + 20000 + index
        greedy_result = run_hill_climb(
            landscape=landscape,
            initial_state=initial_state,
            steps=args.steps,
            seed=greedy_seed,
        )
        sa_result = run_sa(
            landscape=landscape,
            initial_state=initial_state,
            steps=args.steps,
            seed=sa_seed,
            initial_temperature=args.initial_temperature,
            alpha=args.alpha,
        )
        rows.append(
            {
                "initial_state": initial_state,
                "greedy_best_state": greedy_result.best_state,
                "greedy_best_energy": greedy_result.best_energy,
                "greedy_gap_to_global_minimum": greedy_result.best_energy - global_minimum_energy,
                "sa_best_state": sa_result.best_state,
                "sa_best_energy": sa_result.best_energy,
                "sa_gap_to_global_minimum": sa_result.best_energy - global_minimum_energy,
                "sa_minus_greedy_best_energy": sa_result.best_energy - greedy_result.best_energy,
                "global_minimum_state": global_minimum_state,
                "global_minimum_energy": global_minimum_energy,
            }
        )

    summary_rows = summarize(rows, global_minimum_state, global_minimum_energy)
    save_csv(output_dir / "multi_start_results.csv", rows)
    save_csv(output_dir / "multi_start_summary.csv", summary_rows)
    save_csv(
        output_dir / "multi_start_config.csv",
        [
            {
                "M": args.M,
                "k": args.k,
                "c": args.c,
                "R": args.R,
                "steps": args.steps,
                "initial_temperature": args.initial_temperature,
                "alpha": args.alpha,
                "num_starts": args.num_starts,
                "seed": args.seed,
                "gaussian_noise": args.gaussian_noise,
                "noise_mean": args.noise_mean,
                "noise_sd": args.noise_sd,
                "global_minimum_state": global_minimum_state,
                "global_minimum_energy": global_minimum_energy,
            }
        ],
    )

    plot_best_energy_vs_start(rows, global_minimum_energy, output_dir)
    plot_gap_vs_start(rows, output_dir)
    plot_sa_advantage(rows, output_dir)
    plot_summary(summary_rows, output_dir)

    print("algorithm\tmean_best_energy\tmean_gap\thit_rate_global_minimum")
    for row in summary_rows:
        print(
            f"{row['algorithm']}\t"
            f"{row['mean_best_energy']:.6f}\t"
            f"{row['mean_gap_to_global_minimum']:.6f}\t"
            f"{row['hit_rate_global_minimum']:.4f}"
        )


if __name__ == "__main__":
    main()
