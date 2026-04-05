"""Benchmark and visualize Euclidean TSP simulated annealing."""

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
    medium_benchmark_points,
    nearest_neighbor_tour,
    random_insert_proposal,
    random_swap_proposal,
    random_two_opt_proposal,
    route_length,
    small_benchmark_points,
)


def save_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def rolling_mean(values: list[float], window: int) -> list[float]:
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(statistics.fmean(values[start : i + 1]))
    return out


def run_medium_instance(move_name: str, proposal_fn, steps: int, seed: int):
    points = medium_benchmark_points()
    distance_matrix = build_distance_matrix(points)
    initial_tour = nearest_neighbor_tour(distance_matrix)
    optimizer = TspSimulatedAnnealing(
        distance_matrix=distance_matrix,
        proposal_fn=proposal_fn,
        schedule_fn=geometric_schedule(5.0, 0.997),
        rng=random.Random(seed),
    )
    return points, distance_matrix, initial_tour, optimizer.run(initial_tour, steps)


def plot_route_length_traces(results_by_move: dict[str, object], output_dir: Path) -> None:
    plt.figure(figsize=(9, 5.5))
    for move_name, result in results_by_move.items():
        xs = [step.step_index for step in result.trajectory]
        ys = [step.best_route_length for step in result.trajectory]
        plt.plot(xs, ys, label=move_name)
    plt.xlabel("Iteration")
    plt.ylabel("Best route length")
    plt.title("Best Route Length vs Iteration by Move Type")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "best_route_length_vs_iteration_by_move.png", dpi=160)
    plt.close()


def plot_temperature_and_acceptance(result, output_dir: Path) -> None:
    xs = [step.step_index for step in result.trajectory]
    temperatures = [step.temperature for step in result.trajectory]
    accepted = [1.0 if step.accepted else 0.0 for step in result.trajectory]
    uphill = [1.0 if step.uphill_move_accepted else 0.0 for step in result.trajectory]

    plt.figure(figsize=(9, 5))
    plt.plot(xs, temperatures)
    plt.xlabel("Iteration")
    plt.ylabel("Temperature")
    plt.title("Temperature vs Iteration")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "temperature_vs_iteration.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(xs, rolling_mean(accepted, 100), label="Rolling acceptance rate")
    plt.plot(xs, rolling_mean(uphill, 100), label="Rolling uphill acceptance rate")
    plt.xlabel("Iteration")
    plt.ylabel("Rate")
    plt.title("Acceptance Metrics vs Iteration")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "acceptance_metrics_vs_iteration.png", dpi=160)
    plt.close()


def plot_route_snapshots(points, result, output_dir: Path) -> None:
    snapshot_indices = [0, len(result.trajectory) // 3, 2 * len(result.trajectory) // 3, len(result.trajectory) - 1]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, idx in zip(axes.flat, snapshot_indices):
        step = result.trajectory[idx]
        ordered = [points[index] for index in step.tour] + [points[step.tour[0]]]
        ax.plot([p[0] for p in ordered], [p[1] for p in ordered], marker="o")
        ax.set_title(f"Step {step.step_index}, length={step.route_length:.3f}")
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Tour Snapshots Through Annealing")
    fig.tight_layout()
    fig.savefig(output_dir / "tour_snapshots_through_annealing.png", dpi=160)
    plt.close(fig)


def repeated_move_study(steps: int, base_seed: int, medium_initial_length: float):
    move_map = {
        "swap": random_swap_proposal,
        "insert": random_insert_proposal,
        "two_opt": random_two_opt_proposal,
    }
    rows = []
    for move_index, (move_name, proposal_fn) in enumerate(move_map.items()):
        best_lengths = []
        for repeat in range(40):
            _, distance_matrix, initial_tour, result = run_medium_instance(
                move_name,
                proposal_fn,
                steps,
                base_seed + 100 * move_index + repeat,
            )
            best_lengths.append(result.best_route_length)
        rows.append(
            {
                "move_type": move_name,
                "mean_best_route_length": statistics.fmean(best_lengths),
                "std_best_route_length": statistics.pstdev(best_lengths),
                "stderr_best_route_length": statistics.pstdev(best_lengths) / (len(best_lengths) ** 0.5),
                "mean_improvement_over_medium_initial_tour": statistics.fmean(
                    medium_initial_length - length for length in best_lengths
                ),
            }
        )
    return rows


def plot_move_type_summary(rows, output_dir: Path) -> None:
    x = list(range(len(rows)))
    labels = [row["move_type"] for row in rows]
    best = [row["mean_best_route_length"] for row in rows]
    err = [row["stderr_best_route_length"] for row in rows]
    gain = [row["mean_improvement_over_medium_initial_tour"] for row in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1.errorbar(x, best, yerr=err, fmt="o", capsize=5)
    ax1.set_ylabel("Mean best route length")
    ax1.set_title("Move-Type Comparison")
    ax1.grid(True, alpha=0.3)

    ax2.bar(x, gain)
    ax2.set_xticks(x, labels)
    ax2.set_ylabel("Mean improvement over initial tour")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "move_type_summary.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Euclidean TSP simulated annealing.")
    parser.add_argument("--steps", type=int, default=4000, help="Annealing steps for the medium instance.")
    parser.add_argument("--seed", type=int, default=123, help="Base random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PAPER_DIR / "benchmarks" / "outputs",
        help="Directory for CSV and PNG outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    small_points = small_benchmark_points()
    small_distances = build_distance_matrix(small_points)
    exact_tour, exact_length = exact_tsp_optimum(small_distances)
    save_csv(
        output_dir / "small_instance_optimum.csv",
        [{"exact_optimum_route_length": exact_length, "tour": "-".join(map(str, exact_tour))}],
    )
    plot_tour(small_points, exact_tour, f"Small-instance exact optimum ({exact_length:.3f})", output_dir / "small_instance_exact_optimum.png")

    move_map = {
        "swap": random_swap_proposal,
        "insert": random_insert_proposal,
        "two_opt": random_two_opt_proposal,
    }
    results_by_move = {}
    medium_points = medium_benchmark_points()
    medium_distances = build_distance_matrix(medium_points)
    initial_tour = nearest_neighbor_tour(medium_distances)
    plot_tour(medium_points, initial_tour, f"Initial nearest-neighbor tour ({route_length(initial_tour, medium_distances):.3f})", output_dir / "medium_instance_initial_tour.png")

    for move_index, (move_name, proposal_fn) in enumerate(move_map.items()):
        _, _, _, result = run_medium_instance(move_name, proposal_fn, args.steps, args.seed + move_index)
        results_by_move[move_name] = result
        plot_tour(
            medium_points,
            result.best_tour,
            f"{move_name} best tour ({result.best_route_length:.3f})",
            output_dir / f"medium_instance_best_tour_{move_name}.png",
        )

    plot_route_length_traces(results_by_move, output_dir)
    plot_temperature_and_acceptance(results_by_move["two_opt"], output_dir)
    plot_route_snapshots(medium_points, results_by_move["two_opt"], output_dir)

    medium_initial_length = route_length(initial_tour, medium_distances)
    move_rows = repeated_move_study(args.steps, args.seed + 1000, medium_initial_length)
    plot_move_type_summary(move_rows, output_dir)
    save_csv(output_dir / "move_type_summary.csv", move_rows)

    trajectory_rows = [
        {
            "step_index": step.step_index,
            "temperature": step.temperature,
            "route_length": step.route_length,
            "best_route_length": step.best_route_length,
            "accepted": int(step.accepted),
            "uphill_move_accepted": int(step.uphill_move_accepted),
        }
        for step in results_by_move["two_opt"].trajectory
    ]
    save_csv(output_dir / "two_opt_single_run_trajectory.csv", trajectory_rows)

    print("move_type\tbest_route_length\tacceptance_rate\tuphill_moves_accepted")
    for move_name, result in results_by_move.items():
        print(
            f"{move_name}\t{result.best_route_length:.6f}\t"
            f"{result.acceptance_rate:.6f}\t{result.uphill_moves_accepted}"
        )
    print()
    print(f"Small-instance exact optimum length: {exact_length:.6f}")
    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
