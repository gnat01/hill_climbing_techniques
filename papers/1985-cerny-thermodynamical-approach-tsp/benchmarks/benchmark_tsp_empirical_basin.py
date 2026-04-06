"""Empirical basin-of-attraction study for Cerny-style TSP annealing neighborhoods.

This benchmark fixes the V1 Euclidean TSP instance and samples many distinct
initial tours. For each initial tour, it repeatedly runs simulated annealing
with each neighborhood strategy:

- swap
- insert
- two_opt

and estimates, as functions of the initial tour, quantities such as:

- exact optimum hit probability
- epsilon-hit probability
- mean best-route gap to optimum
"""

from __future__ import annotations

import argparse
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

from benchmark_tsp_annealing_v1 import cerny_v1_points, save_csv  # noqa: E402
from python.tsp_annealing import (  # noqa: E402
    TspSimulatedAnnealing,
    build_distance_matrix,
    canonical_tour,
    exact_tsp_optimum,
    geometric_schedule,
    random_insert_proposal,
    random_swap_proposal,
    random_two_opt_proposal,
    route_length,
)


MOVE_MAP = {
    "swap": random_swap_proposal,
    "insert": random_insert_proposal,
    "two_opt": random_two_opt_proposal,
}


def clamp_runs_per_start(value: int) -> int:
    return max(10, min(150, value))


def sample_distinct_initial_tours(num_cities: int, num_starts: int, seed: int) -> list[tuple[int, ...]]:
    if num_starts <= 0:
        raise ValueError("num-starts must be positive")
    rng = random.Random(seed)
    seen: set[tuple[int, ...]] = set()
    tours: list[tuple[int, ...]] = []
    base = list(range(num_cities))
    while len(tours) < num_starts:
        candidate = base[:]
        rng.shuffle(candidate)
        canonical = canonical_tour(candidate)
        if canonical not in seen:
            seen.add(canonical)
            tours.append(canonical)
    return tours


def run_sa_tsp(*, distance_matrix, proposal_fn, initial_tour, steps: int, seed: int, initial_temperature: float, alpha: float):
    optimizer = TspSimulatedAnnealing(
        distance_matrix=distance_matrix,
        proposal_fn=proposal_fn,
        schedule_fn=geometric_schedule(initial_temperature, alpha),
        rng=random.Random(seed),
    )
    return optimizer.run(initial_tour=initial_tour, steps=steps)


def plot_hit_probability_vs_start(rows, output_dir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for axis, move_name in zip(axes, MOVE_MAP.keys()):
        move_rows = [row for row in rows if row["move_type"] == move_name]
        axis.plot(
            [row["start_index"] for row in move_rows],
            [row["hit_probability_exact"] for row in move_rows],
            "o-",
            label="Exact hit probability",
            markersize=4,
        )
        axis.plot(
            [row["start_index"] for row in move_rows],
            [row["hit_probability_epsilon"] for row in move_rows],
            "o-",
            label="Epsilon-hit probability",
            markersize=4,
        )
        axis.set_ylabel("Probability")
        axis.set_title(f"{move_name} empirical basin by sampled start")
        axis.grid(True, alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel("Sampled initial-tour index")
    fig.tight_layout()
    fig.savefig(output_dir / "hit_probability_vs_start_index_by_move.png", dpi=160)
    plt.close(fig)


def plot_gap_vs_start(rows, output_dir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for axis, move_name in zip(axes, MOVE_MAP.keys()):
        move_rows = [row for row in rows if row["move_type"] == move_name]
        axis.errorbar(
            [row["start_index"] for row in move_rows],
            [row["mean_gap_to_optimum"] for row in move_rows],
            yerr=[row["stderr_gap_to_optimum"] for row in move_rows],
            fmt="o-",
            capsize=4,
            markersize=4,
        )
        axis.set_ylabel("Mean gap")
        axis.set_title(f"{move_name} mean gap by sampled start")
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Sampled initial-tour index")
    fig.tight_layout()
    fig.savefig(output_dir / "mean_gap_vs_start_index_by_move.png", dpi=160)
    plt.close(fig)


def plot_hit_probability_vs_initial_gap(rows, output_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    for move_name, color in [("swap", "tab:blue"), ("insert", "tab:orange"), ("two_opt", "tab:green")]:
        move_rows = [row for row in rows if row["move_type"] == move_name]
        plt.scatter(
            [row["initial_gap_to_optimum"] for row in move_rows],
            [row["hit_probability_exact"] for row in move_rows],
            s=30,
            alpha=0.75,
            label=move_name,
            color=color,
        )
    plt.xlabel("Initial tour gap to optimum")
    plt.ylabel("Exact optimum hit probability")
    plt.title("Empirical Basin vs Initial Tour Difficulty")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "hit_probability_vs_initial_gap_by_move.png", dpi=160)
    plt.close()


def plot_mean_gap_vs_initial_gap(rows, output_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    for move_name, color in [("swap", "tab:blue"), ("insert", "tab:orange"), ("two_opt", "tab:green")]:
        move_rows = [row for row in rows if row["move_type"] == move_name]
        plt.scatter(
            [row["initial_gap_to_optimum"] for row in move_rows],
            [row["mean_gap_to_optimum"] for row in move_rows],
            s=30,
            alpha=0.75,
            label=move_name,
            color=color,
        )
    plt.xlabel("Initial tour gap to optimum")
    plt.ylabel("Mean best-route gap to optimum")
    plt.title("Recovered Gap vs Initial Tour Difficulty")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "mean_gap_vs_initial_gap_by_move.png", dpi=160)
    plt.close()


def plot_move_summary(summary_rows, output_dir: Path) -> None:
    x = list(range(len(summary_rows)))
    labels = [row["move_type"] for row in summary_rows]

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    axes[0].errorbar(
        x,
        [row["mean_best_gap_over_starts"] for row in summary_rows],
        yerr=[row["stderr_best_gap_over_starts"] for row in summary_rows],
        fmt="o",
        capsize=5,
    )
    axes[0].set_ylabel("Mean gap")
    axes[0].set_title("Mean Best-Route Gap Across Sampled Starts")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, [row["mean_exact_hit_probability"] for row in summary_rows], "o-", color="tab:red")
    axes[1].set_ylabel("Mean exact hit probability")
    axes[1].set_title("Mean Exact Hit Probability Across Starts")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, [row["mean_epsilon_hit_probability"] for row in summary_rows], "o-", color="tab:green")
    axes[2].set_xticks(x, labels)
    axes[2].set_ylabel("Mean epsilon-hit probability")
    axes[2].set_title("Mean Epsilon-Hit Probability Across Starts")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "move_type_basin_summary.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Empirical basin study for Cerny TSP move strategies.")
    parser.add_argument("--steps", type=int, default=800, help="Annealing steps per run.")
    parser.add_argument("--initial-temperature", type=float, default=6.0, help="Initial temperature.")
    parser.add_argument("--alpha", type=float, default=0.997, help="Geometric cooling factor.")
    parser.add_argument("--num-starts", type=int, default=60, help="Number of distinct sampled initial tours.")
    parser.add_argument(
        "--runs-per-start",
        type=int,
        default=30,
        help="Requested SA runs per initial tour per move type. Clamped into [10, 150].",
    )
    parser.add_argument(
        "--success-epsilon",
        type=float,
        default=0.0,
        help="Route-length tolerance above the exact optimum that still counts as success.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Base random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PAPER_DIR / "benchmarks" / "empirical_basin_outputs",
        help="Directory for CSV and PNG outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 0:
        raise ValueError("steps must be non-negative")
    if args.initial_temperature <= 0.0:
        raise ValueError("initial-temperature must be positive")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if args.num_starts <= 0:
        raise ValueError("num-starts must be positive")
    if args.success_epsilon < 0.0:
        raise ValueError("success-epsilon must be non-negative")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    points = cerny_v1_points()
    distance_matrix = build_distance_matrix(points)
    optimum_tour, optimum_length = exact_tsp_optimum(distance_matrix)
    runs_per_start = clamp_runs_per_start(args.runs_per_start)
    initial_tours = sample_distinct_initial_tours(len(points), args.num_starts, args.seed)

    rows = []
    for start_index, initial_tour in enumerate(initial_tours):
        initial_length = route_length(initial_tour, distance_matrix)
        initial_gap = initial_length - optimum_length
        for move_offset, (move_name, proposal_fn) in enumerate(MOVE_MAP.items()):
            best_lengths: list[float] = []
            best_tours: list[tuple[int, ...]] = []
            exact_hits = 0
            epsilon_hits = 0
            for run_index in range(runs_per_start):
                seed = args.seed + 100000 * (move_offset + 1) + 1000 * start_index + run_index
                result = run_sa_tsp(
                    distance_matrix=distance_matrix,
                    proposal_fn=proposal_fn,
                    initial_tour=initial_tour,
                    steps=args.steps,
                    seed=seed,
                    initial_temperature=args.initial_temperature,
                    alpha=args.alpha,
                )
                best_lengths.append(result.best_route_length)
                best_tours.append(canonical_tour(result.best_tour))
                if canonical_tour(result.best_tour) == optimum_tour:
                    exact_hits += 1
                if result.best_route_length <= optimum_length + args.success_epsilon:
                    epsilon_hits += 1

            mean_best = statistics.fmean(best_lengths)
            std_best = statistics.pstdev(best_lengths) if len(best_lengths) > 1 else 0.0
            gaps = [value - optimum_length for value in best_lengths]
            rows.append(
                {
                    "start_index": start_index,
                    "move_type": move_name,
                    "initial_tour": "-".join(str(city) for city in initial_tour),
                    "initial_route_length": initial_length,
                    "initial_gap_to_optimum": initial_gap,
                    "runs_per_start": runs_per_start,
                    "hit_probability_exact": exact_hits / runs_per_start,
                    "hit_probability_epsilon": epsilon_hits / runs_per_start,
                    "mean_best_route_length": mean_best,
                    "stderr_best_route_length": std_best / math.sqrt(runs_per_start) if runs_per_start > 1 else 0.0,
                    "mean_gap_to_optimum": statistics.fmean(gaps),
                    "stderr_gap_to_optimum": statistics.pstdev(gaps) / math.sqrt(runs_per_start) if len(gaps) > 1 else 0.0,
                    "mean_percent_improvement_over_initial": 100.0 * (initial_length - mean_best) / initial_length,
                    "optimum_length": optimum_length,
                }
            )

    summary_rows = []
    for move_name in MOVE_MAP:
        move_rows = [row for row in rows if row["move_type"] == move_name]
        mean_gaps = [row["mean_gap_to_optimum"] for row in move_rows]
        exact_hit_probs = [row["hit_probability_exact"] for row in move_rows]
        epsilon_hit_probs = [row["hit_probability_epsilon"] for row in move_rows]
        summary_rows.append(
            {
                "move_type": move_name,
                "num_starts": args.num_starts,
                "runs_per_start": runs_per_start,
                "mean_best_gap_over_starts": statistics.fmean(mean_gaps),
                "stderr_best_gap_over_starts": statistics.pstdev(mean_gaps) / math.sqrt(len(mean_gaps)) if len(mean_gaps) > 1 else 0.0,
                "mean_exact_hit_probability": statistics.fmean(exact_hit_probs),
                "mean_epsilon_hit_probability": statistics.fmean(epsilon_hit_probs),
                "optimum_length": optimum_length,
            }
        )

    save_csv(output_dir / "empirical_basin_results.csv", rows)
    save_csv(output_dir / "empirical_basin_summary.csv", summary_rows)
    save_csv(
        output_dir / "empirical_basin_config.csv",
        [
            {
                "steps": args.steps,
                "initial_temperature": args.initial_temperature,
                "alpha": args.alpha,
                "num_starts": args.num_starts,
                "requested_runs_per_start": args.runs_per_start,
                "effective_runs_per_start": runs_per_start,
                "success_epsilon": args.success_epsilon,
                "seed": args.seed,
                "optimum_tour": "-".join(str(city) for city in optimum_tour),
                "optimum_length": optimum_length,
            }
        ],
    )

    plot_hit_probability_vs_start(rows, output_dir)
    plot_gap_vs_start(rows, output_dir)
    plot_hit_probability_vs_initial_gap(rows, output_dir)
    plot_mean_gap_vs_initial_gap(rows, output_dir)
    plot_move_summary(summary_rows, output_dir)

    print("move_type\tmean_best_gap_over_starts\tmean_exact_hit_probability\tmean_epsilon_hit_probability")
    for row in summary_rows:
        print(
            f"{row['move_type']}\t{row['mean_best_gap_over_starts']:.6f}\t"
            f"{row['mean_exact_hit_probability']:.6f}\t{row['mean_epsilon_hit_probability']:.6f}"
        )
    print()
    print(f"Exact optimum length: {optimum_length:.6f}")
    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
