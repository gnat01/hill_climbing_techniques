"""Tightened Cerny benchmark with a harder TSP teaching instance.

This leaves the original benchmark file intact as a simpler V0 and uses a
deliberately poor initial tour plus an exact optimum reference to make the
annealing effect visually obvious.
"""

from __future__ import annotations

import argparse
import csv
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

from python.tsp_annealing import (  # noqa: E402
    TspSimulatedAnnealing,
    build_distance_matrix,
    exact_tsp_optimum,
    geometric_schedule,
    random_insert_proposal,
    random_swap_proposal,
    random_two_opt_proposal,
    route_length,
)


def save_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def cerny_v1_points() -> tuple[tuple[float, float], ...]:
    return (
        (0.0, 0.2),
        (1.0, 2.8),
        (2.0, 0.0),
        (3.0, 3.0),
        (4.0, 0.1),
        (5.0, 2.7),
        (6.0, -0.1),
        (6.4, 3.4),
        (5.1, 6.0),
        (3.0, 5.3),
    )


def bad_zigzag_initial_tour() -> tuple[int, ...]:
    return (0, 9, 1, 8, 2, 7, 3, 6, 4, 5)


def rolling_mean(values: list[float], window: int) -> list[float]:
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(statistics.fmean(values[start : i + 1]))
    return result


def plot_tour(points, tour, title: str, output_path: Path) -> None:
    ordered = [points[index] for index in tour] + [points[tour[0]]]
    xs = [point[0] for point in ordered]
    ys = [point[1] for point in ordered]
    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, marker="o")
    for index, point in enumerate(points):
        plt.text(point[0] + 0.03, point[1] + 0.03, str(index), fontsize=8)
    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def run_instance(proposal_fn, steps: int, seed: int):
    points = cerny_v1_points()
    distances = build_distance_matrix(points)
    initial_tour = bad_zigzag_initial_tour()
    optimizer = TspSimulatedAnnealing(
        distance_matrix=distances,
        proposal_fn=proposal_fn,
        schedule_fn=geometric_schedule(6.0, 0.997),
        rng=random.Random(seed),
    )
    return points, distances, initial_tour, optimizer.run(initial_tour, steps)


def plot_route_length_and_gap(result, optimum_length: float, output_dir: Path) -> None:
    xs = [step.step_index for step in result.trajectory]
    current_length = [step.route_length for step in result.trajectory]
    best_length = [step.best_route_length for step in result.trajectory]
    best_gap = [value - optimum_length for value in best_length]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(xs, current_length, label="Current route length", alpha=0.8)
    ax1.plot(xs, best_length, label="Best route length", linewidth=2.0)
    ax1.axhline(optimum_length, color="black", linestyle="--", label="Exact optimum")
    ax1.set_ylabel("Route length")
    ax1.set_title("Route Length vs Iteration")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(xs, best_gap, color="#d62728")
    ax2.axhline(0.0, color="black", linestyle="--")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Best-length gap to optimum")
    ax2.set_title("Gap to Exact Optimum vs Iteration")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "route_length_and_gap_vs_iteration.png", dpi=160)
    plt.close(fig)


def plot_temperature_and_acceptance(result, output_dir: Path) -> None:
    xs = [step.step_index for step in result.trajectory]
    temperatures = [step.temperature for step in result.trajectory]
    accepted = [1.0 if step.accepted else 0.0 for step in result.trajectory]
    uphill = [1.0 if step.uphill_move_accepted else 0.0 for step in result.trajectory]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(xs, temperatures)
    ax1.set_ylabel("Temperature")
    ax1.set_title("Temperature vs Iteration")
    ax1.grid(True, alpha=0.3)

    ax2.plot(xs, rolling_mean(accepted, 100), label="Rolling acceptance rate")
    ax2.plot(xs, rolling_mean(uphill, 100), label="Rolling uphill acceptance rate")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Rate")
    ax2.set_title("Acceptance Metrics vs Iteration")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "temperature_and_acceptance_vs_iteration.png", dpi=160)
    plt.close(fig)


def plot_tour_snapshots(points, result, output_dir: Path) -> None:
    snapshot_indices = [0, len(result.trajectory) // 4, len(result.trajectory) // 2, len(result.trajectory) - 1]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, idx in zip(axes.flat, snapshot_indices):
        step = result.trajectory[idx]
        ordered = [points[index] for index in step.tour] + [points[step.tour[0]]]
        ax.plot([p[0] for p in ordered], [p[1] for p in ordered], marker="o")
        ax.set_title(f"Step {step.step_index}, len={step.route_length:.3f}")
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Tour Snapshots Through Annealing")
    fig.tight_layout()
    fig.savefig(output_dir / "tour_snapshots_through_annealing_v1.png", dpi=160)
    plt.close(fig)


def repeated_move_study(steps: int, repeats: int, base_seed: int, optimum_length: float, initial_length: float):
    move_map = {
        "swap": random_swap_proposal,
        "insert": random_insert_proposal,
        "two_opt": random_two_opt_proposal,
    }
    rows = []
    for move_index, (move_name, proposal_fn) in enumerate(move_map.items()):
        best_lengths = []
        for repeat in range(repeats):
            _, _, initial_tour, result = run_instance(
                proposal_fn,
                steps,
                base_seed + 1000 * (move_index + 1) + repeat,
            )
            assert initial_tour == bad_zigzag_initial_tour()
            best_lengths.append(result.best_route_length)
        mean_best = statistics.fmean(best_lengths)
        std_best = statistics.pstdev(best_lengths)
        rows.append(
            {
                "move_type": move_name,
                "runs": repeats,
                "mean_best_route_length": mean_best,
                "std_best_route_length": std_best,
                "stderr_best_route_length": std_best / (len(best_lengths) ** 0.5),
                "mean_gap_to_optimum": mean_best - optimum_length,
                "mean_percent_improvement_over_initial": 100.0 * (initial_length - mean_best) / initial_length,
            }
        )
    return rows


def plot_move_type_summary(rows, output_dir: Path) -> None:
    x = list(range(len(rows)))
    labels = [row["move_type"] for row in rows]
    best = [row["mean_best_route_length"] for row in rows]
    err = [row["stderr_best_route_length"] for row in rows]
    gap = [row["mean_gap_to_optimum"] for row in rows]
    improvement = [row["mean_percent_improvement_over_initial"] for row in rows]

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axes[0].errorbar(x, best, yerr=err, fmt="o", capsize=5)
    axes[0].set_ylabel("Mean best route length")
    axes[0].set_title("Move-Type Comparison on Harder TSP Instance")
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(x, gap)
    axes[1].set_ylabel("Gap to exact optimum")
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(x, improvement)
    axes[2].set_xticks(x, labels)
    axes[2].set_ylabel("% improvement over initial")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "move_type_summary_v1.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tightened Cerny TSP benchmark.")
    parser.add_argument("--steps", type=int, default=800, help="Annealing steps.")
    parser.add_argument("--seed", type=int, default=123, help="Base random seed.")
    parser.add_argument("--repeats", type=int, default=100, help="Repeated runs per move type for summary statistics.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PAPER_DIR / "benchmarks" / "outputs_v1",
        help="Directory for CSV and PNG outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats <= 1:
        raise ValueError("repeats must be greater than 1")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    points = cerny_v1_points()
    distances = build_distance_matrix(points)
    initial_tour = bad_zigzag_initial_tour()
    initial_length = route_length(initial_tour, distances)
    optimum_tour, optimum_length = exact_tsp_optimum(distances)

    plot_tour(points, initial_tour, f"Initial bad tour ({initial_length:.3f})", output_dir / "initial_bad_tour.png")
    plot_tour(points, optimum_tour, f"Exact optimum tour ({optimum_length:.3f})", output_dir / "exact_optimum_tour.png")

    move_map = {
        "swap": random_swap_proposal,
        "insert": random_insert_proposal,
        "two_opt": random_two_opt_proposal,
    }
    results_by_move = {}
    for move_index, (move_name, proposal_fn) in enumerate(move_map.items()):
        _, _, _, result = run_instance(proposal_fn, args.steps, args.seed + move_index)
        results_by_move[move_name] = result
        plot_tour(
            points,
            result.best_tour,
            f"{move_name} best tour ({result.best_route_length:.3f})",
            output_dir / f"best_tour_{move_name}.png",
        )

    plot_route_length_and_gap(results_by_move["two_opt"], optimum_length, output_dir)
    plot_temperature_and_acceptance(results_by_move["two_opt"], output_dir)
    plot_tour_snapshots(points, results_by_move["two_opt"], output_dir)

    move_rows = repeated_move_study(args.steps, args.repeats, args.seed + 1000, optimum_length, initial_length)
    plot_move_type_summary(move_rows, output_dir)
    save_csv(output_dir / "move_type_summary_v1.csv", move_rows)
    save_csv(
        output_dir / "optimum_reference_v1.csv",
        [
            {
                "initial_route_length": initial_length,
                "exact_optimum_route_length": optimum_length,
                "initial_gap_to_optimum": initial_length - optimum_length,
            }
        ],
    )
    save_csv(
        output_dir / "two_opt_single_run_trajectory_v1.csv",
        [
            {
                "step_index": step.step_index,
                "temperature": step.temperature,
                "route_length": step.route_length,
                "best_route_length": step.best_route_length,
                "accepted": int(step.accepted),
                "uphill_move_accepted": int(step.uphill_move_accepted),
                "gap_to_optimum": step.best_route_length - optimum_length,
            }
            for step in results_by_move["two_opt"].trajectory
        ],
    )

    print("Repeated-run summary")
    print("move_type\truns\tmean_best_route_length\tstderr_best_route_length\tmean_gap_to_optimum\tmean_percent_improvement_over_initial")
    for row in move_rows:
        print(
            f"{row['move_type']}\t{row['runs']}\t{row['mean_best_route_length']:.6f}\t"
            f"{row['stderr_best_route_length']:.6f}\t{row['mean_gap_to_optimum']:.6f}\t"
            f"{row['mean_percent_improvement_over_initial']:.2f}"
        )
    print()
    print("Illustrative single-run results")
    print("move_type\tbest_route_length\tgap_to_optimum\tpercent_improvement_over_initial")
    for move_name, result in results_by_move.items():
        gap = result.best_route_length - optimum_length
        improvement = 100.0 * (initial_length - result.best_route_length) / initial_length
        print(f"{move_name}\t{result.best_route_length:.6f}\t{gap:.6f}\t{improvement:.2f}")
    print()
    print(f"Initial route length: {initial_length:.6f}")
    print(f"Exact optimum length: {optimum_length:.6f}")
    print(f"Initial gap to optimum: {initial_length - optimum_length:.6f}")
    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
