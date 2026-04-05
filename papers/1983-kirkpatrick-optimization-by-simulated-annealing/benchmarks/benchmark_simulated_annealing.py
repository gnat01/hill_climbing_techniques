"""Benchmark and visualize simulated annealing on a rugged 1D landscape."""

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

from python.simulated_annealing import (  # noqa: E402
    SimulatedAnnealingOptimizer,
    geometric_schedule,
    greedy_hill_climb,
    random_walk_proposal,
    rugged_landscape_energy,
)

GEOMETRIC_ALPHAS = [0.99, 0.95, 0.9, 0.8, 0.7, 0.65, 0.6, 0.5, 0.4, 0.3]


def rolling_mean(values: list[float], window: int) -> list[float]:
    result: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        chunk = values[start : index + 1]
        result.append(statistics.fmean(chunk))
    return result


def run_sa(
    *,
    initial_temperature: float,
    cooling_rate: float,
    step_size: float,
    steps: int,
    seed: int,
    initial_state: float,
) -> object:
    optimizer = SimulatedAnnealingOptimizer(
        energy_fn=rugged_landscape_energy,
        proposal_fn=random_walk_proposal(step_size=step_size),
        schedule_fn=geometric_schedule(initial_temperature=initial_temperature, cooling_rate=cooling_rate),
        rng=random.Random(seed),
    )
    return optimizer.run(initial_state=initial_state, steps=steps)


def run_hill_climb(*, step_size: float, steps: int, seed: int, initial_state: float) -> object:
    return greedy_hill_climb(
        initial_state=initial_state,
        energy_fn=rugged_landscape_energy,
        proposal_fn=random_walk_proposal(step_size=step_size),
        steps=steps,
        rng=random.Random(seed),
    )


def save_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_temperature_trace(result, output_dir: Path) -> None:
    steps = [step.step_index for step in result.trajectory]
    temperatures = [step.temperature for step in result.trajectory]
    plt.figure(figsize=(8, 5))
    plt.plot(steps, temperatures, linewidth=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("Temperature")
    plt.title("Temperature vs Iteration")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "temperature_vs_iteration.png", dpi=160)
    plt.close()


def plot_energy_traces(sa_result, hc_result, output_dir: Path) -> None:
    steps = [step.step_index for step in sa_result.trajectory]
    sa_energy = [step.energy for step in sa_result.trajectory]
    sa_best = [step.best_energy for step in sa_result.trajectory]
    hc_energy = [step.energy for step in hc_result.trajectory]
    hc_best = [step.best_energy for step in hc_result.trajectory]

    plt.figure(figsize=(9, 5))
    plt.plot(steps, sa_energy, label="SA current energy", alpha=0.8)
    plt.plot(steps, sa_best, label="SA best energy", linewidth=2.0)
    plt.plot(steps, hc_energy, label="Hill climb current energy", alpha=0.7)
    plt.plot(steps, hc_best, label="Hill climb best energy", linestyle="--", linewidth=2.0)
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title("Current and Best Energy vs Iteration")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "energy_traces_vs_iteration.png", dpi=160)
    plt.close()


def plot_single_run_sa_traces(sa_result, output_dir: Path) -> None:
    steps = [step.step_index for step in sa_result.trajectory]
    energies = [step.energy for step in sa_result.trajectory]
    best_energies = [step.best_energy for step in sa_result.trajectory]
    states = [step.state for step in sa_result.trajectory]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, energies, linewidth=1.3)
    plt.xlabel("Iteration")
    plt.ylabel("Current energy")
    plt.title("SA Current Energy vs Iteration")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "sa_current_energy_vs_iteration.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(steps, best_energies, linewidth=1.6)
    plt.xlabel("Iteration")
    plt.ylabel("Best-so-far energy")
    plt.title("SA Best-So-Far Energy vs Iteration")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "sa_best_energy_vs_iteration.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(steps, states, linewidth=1.1)
    plt.xlabel("Iteration")
    plt.ylabel("State")
    plt.title("SA State vs Iteration")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "sa_state_vs_iteration.png", dpi=160)
    plt.close()


def plot_acceptance_metrics(sa_result, output_dir: Path, window: int = 100) -> None:
    steps = [step.step_index for step in sa_result.trajectory]
    accepted = [1.0 if step.accepted else 0.0 for step in sa_result.trajectory]
    uphill = [1.0 if step.uphill_move_accepted else 0.0 for step in sa_result.trajectory]
    rolling_acceptance = rolling_mean(accepted, window=window)
    rolling_uphill = rolling_mean(uphill, window=window)

    plt.figure(figsize=(9, 5))
    plt.plot(steps, rolling_acceptance, label="Rolling acceptance rate")
    plt.plot(steps, rolling_uphill, label="Rolling uphill acceptance rate")
    plt.xlabel("Iteration")
    plt.ylabel("Rate")
    plt.title(f"Rolling Acceptance Metrics (window={window})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "rolling_acceptance_metrics.png", dpi=160)
    plt.close()

    rolling_uphill_count = []
    for index in range(len(uphill)):
        start = max(0, index - window + 1)
        rolling_uphill_count.append(sum(uphill[start : index + 1]))

    plt.figure(figsize=(9, 5))
    plt.plot(steps, rolling_uphill_count, color="#d62728")
    plt.xlabel("Iteration")
    plt.ylabel("Accepted uphill moves in window")
    plt.title(f"Rolling Uphill Accepted Moves (window={window})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "rolling_uphill_accepted_count.png", dpi=160)
    plt.close()


def plot_landscape_trajectory(sa_result, hc_result, output_dir: Path) -> None:
    xs = [(-3.0 + 6.0 * i / 1200) for i in range(1201)]
    ys = [rugged_landscape_energy(x) for x in xs]

    sa_points_x = [step.state for step in sa_result.trajectory[:: max(1, len(sa_result.trajectory) // 150)]]
    sa_points_y = [rugged_landscape_energy(x) for x in sa_points_x]
    hc_points_x = [step.state for step in hc_result.trajectory[:: max(1, len(hc_result.trajectory) // 150)]]
    hc_points_y = [rugged_landscape_energy(x) for x in hc_points_x]

    plt.figure(figsize=(9, 5))
    plt.plot(xs, ys, color="black", linewidth=1.4, label="Landscape")
    plt.scatter(sa_points_x, sa_points_y, s=18, alpha=0.7, label="SA trajectory")
    plt.scatter(hc_points_x, hc_points_y, s=18, alpha=0.7, label="Hill climb trajectory")
    plt.xlabel("State x")
    plt.ylabel("Energy")
    plt.title("State Trajectory on the Rugged Landscape")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "state_trajectory_on_landscape.png", dpi=160)
    plt.close()


def plot_energy_histograms_by_temperature(sa_result, output_dir: Path) -> None:
    buckets = {"high": [], "mid": [], "low": []}
    for step in sa_result.trajectory:
        if step.temperature >= 1.5:
            buckets["high"].append(step.energy)
        elif step.temperature >= 0.4:
            buckets["mid"].append(step.energy)
        else:
            buckets["low"].append(step.energy)

    plt.figure(figsize=(9, 5))
    for label, color in [("high", "#1f77b4"), ("mid", "#ff7f0e"), ("low", "#2ca02c")]:
        if buckets[label]:
            plt.hist(buckets[label], bins=30, alpha=0.45, label=f"{label} temperature band", color=color)
    plt.xlabel("Energy")
    plt.ylabel("Frequency")
    plt.title("Energy Histogram by Temperature Band")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "energy_histogram_by_temperature_band.png", dpi=160)
    plt.close()


def compute_uphill_fraction_by_temperature_band(results) -> list[dict[str, float | str | int]]:
    labels = ["high", "mid", "low"]
    per_band = {label: [] for label in labels}

    for result in results:
        attempts = {label: 0 for label in labels}
        accepted = {label: 0 for label in labels}
        for step in result.trajectory:
            if step.temperature >= 1.5:
                key = "high"
            elif step.temperature >= 0.4:
                key = "mid"
            else:
                key = "low"

            if step.delta_energy > 0.0:
                attempts[key] += 1
                if step.uphill_move_accepted:
                    accepted[key] += 1

        for label in labels:
            fraction = accepted[label] / attempts[label] if attempts[label] else 0.0
            per_band[label].append(fraction)

    rows = []
    for x_position, label in enumerate(labels):
        values = per_band[label]
        mean_value = statistics.fmean(values)
        stderr = statistics.pstdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
        rows.append(
            {
                "temperature_band": label,
                "x_position": x_position,
                "mean_uphill_fraction": mean_value,
                "stderr_uphill_fraction": stderr,
            }
        )
    return rows


def plot_uphill_fraction_by_temperature_band(rows, output_dir: Path) -> None:
    x_values = [row["x_position"] for row in rows]
    y_values = [row["mean_uphill_fraction"] for row in rows]
    y_err = [row["stderr_uphill_fraction"] for row in rows]
    labels = [row["temperature_band"] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.errorbar(x_values, y_values, yerr=y_err, fmt="o", capsize=5, markersize=7, linewidth=1.5)
    plt.xticks(x_values, labels)
    plt.xlabel("Temperature band")
    plt.ylabel("Accepted uphill fraction")
    plt.title("Accepted Uphill Fraction by Temperature Band")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "uphill_fraction_by_temperature_band.png", dpi=160)
    plt.close()


def plot_final_state_distribution(results, output_dir: Path) -> None:
    best_states = [result.best_state for result in results]
    plt.figure(figsize=(8, 5))
    plt.hist(best_states, bins=20, color="#4c78a8", alpha=0.8)
    plt.xlabel("Best state reached")
    plt.ylabel("Count")
    plt.title("Distribution of Final Best States Across Repeated Runs")
    plt.tight_layout()
    plt.savefig(output_dir / "final_best_state_distribution.png", dpi=160)
    plt.close()


def plot_parameter_sweep(rows, parameter_key: str, title: str, xlabel: str, filename: str, output_dir: Path) -> None:
    x_values = [row[parameter_key] for row in rows]
    y_values = [row["mean_best_energy"] for row in rows]
    y_err = [row["std_best_energy"] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.errorbar(x_values, y_values, yerr=y_err, marker="o", capsize=4)
    plt.xlabel(xlabel)
    plt.ylabel("Mean final best energy")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=160)
    plt.close()


def plot_geometric_alpha_study(rows, output_dir: Path) -> None:
    x_values = [row["cooling_rate"] for row in rows]
    best_energy = [row["mean_best_energy"] for row in rows]
    best_energy_err = [row["stderr_best_energy"] for row in rows]
    global_gap = [row["mean_best_energy_gap_to_global_minimum"] for row in rows]
    global_gap_err = [row["stderr_best_energy_gap_to_global_minimum"] for row in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
    ax1.errorbar(
        x_values,
        best_energy,
        yerr=best_energy_err,
        marker="o",
        capsize=4,
        linewidth=1.6,
        color="#1f77b4",
    )
    ax1.set_ylabel("Final best energy")
    ax1.set_title("Final Best Energy vs Geometric Schedule Alpha")
    ax1.grid(True, alpha=0.3)

    ax2.errorbar(
        x_values,
        global_gap,
        yerr=global_gap_err,
        marker="s",
        capsize=4,
        linewidth=1.6,
        color="#d62728",
    )
    ax2.set_xlabel("Geometric schedule alpha")
    ax2.set_ylabel("Gap to global optimum")
    ax2.set_title("Gap to Global Optimum vs Geometric Schedule Alpha")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Geometric Schedule Alpha Study", y=0.98)
    fig.tight_layout()
    plt.savefig(output_dir / "final_energy_vs_geometric_schedule_alphas.png", dpi=160)
    plt.close(fig)


def plot_sa_vs_hill_summary(sa_results, hc_results, output_dir: Path) -> None:
    sa_best = [result.best_energy for result in sa_results]
    hc_best = [result.best_energy for result in hc_results]

    plt.figure(figsize=(8, 5))
    plt.boxplot([sa_best, hc_best], tick_labels=["Simulated annealing", "Hill climbing"])
    plt.ylabel("Final best energy")
    plt.title("SA vs Hill Climbing Across Repeated Runs")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "sa_vs_hill_climbing_boxplot.png", dpi=160)
    plt.close()


def estimate_global_minimum() -> dict[str, float]:
    xs = [(-3.0 + 6.0 * i / 200000) for i in range(200001)]
    best_x = xs[0]
    best_energy = rugged_landscape_energy(best_x)
    for x in xs[1:]:
        energy = rugged_landscape_energy(x)
        if energy < best_energy:
            best_energy = energy
            best_x = x
    return {"global_minimizer_x": best_x, "global_minimum_energy": best_energy}


def summarize_runs(results, label: str, global_minimum_energy: float) -> dict[str, float | str]:
    best_energies = [result.best_energy for result in results]
    final_energies = [result.final_energy for result in results]
    return {
        "algorithm": label,
        "mean_final_energy": statistics.fmean(final_energies),
        "std_final_energy": statistics.pstdev(final_energies) if len(final_energies) > 1 else 0.0,
        "mean_final_energy_gap_to_global_minimum": statistics.fmean(
            energy - global_minimum_energy for energy in final_energies
        ),
        "mean_best_energy_gap_to_global_minimum": statistics.fmean(
            energy - global_minimum_energy for energy in best_energies
        ),
        "mean_best_energy": statistics.fmean(best_energies),
        "std_best_energy": statistics.pstdev(best_energies) if len(best_energies) > 1 else 0.0,
        "mean_acceptance_rate": statistics.fmean(result.acceptance_rate for result in results),
        "mean_uphill_accepted": statistics.fmean(result.uphill_moves_accepted for result in results),
    }


def repeated_sa_sweep(
    param_name: str,
    values: list[float],
    *,
    repeats: int,
    steps: int,
    initial_state: float,
    base_seed: int,
    global_minimum_energy: float,
):
    rows = []
    for value_index, value in enumerate(values):
        results = []
        for repeat in range(repeats):
            kwargs = {
                "initial_temperature": 3.5,
                "cooling_rate": 0.995,
                "step_size": 0.6,
            }
            kwargs[param_name] = value
            result = run_sa(
                initial_temperature=kwargs["initial_temperature"],
                cooling_rate=kwargs["cooling_rate"],
                step_size=kwargs["step_size"],
                steps=steps,
                seed=base_seed + 100 * value_index + repeat,
                initial_state=initial_state,
            )
            results.append(result)
        rows.append(
            {
                param_name: value,
                "mean_best_energy": statistics.fmean(result.best_energy for result in results),
                "std_best_energy": statistics.pstdev(result.best_energy for result in results),
                "stderr_best_energy": (
                    statistics.pstdev(result.best_energy for result in results) / math.sqrt(len(results))
                    if len(results) > 1
                    else 0.0
                ),
                "mean_best_energy_gap_to_global_minimum": statistics.fmean(
                    result.best_energy - global_minimum_energy for result in results
                ),
                "stderr_best_energy_gap_to_global_minimum": (
                    statistics.pstdev(
                        result.best_energy - global_minimum_energy for result in results
                    )
                    / math.sqrt(len(results))
                    if len(results) > 1
                    else 0.0
                ),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark simulated annealing with extensive plots.")
    parser.add_argument("--steps", type=int, default=2500, help="Number of optimization steps.")
    parser.add_argument("--seed", type=int, default=123, help="Base random seed.")
    parser.add_argument("--initial-state", type=float, default=2.2, help="Initial state for the 1D landscape.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PAPER_DIR / "benchmarks" / "outputs",
        help="Directory for CSV and PNG outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sa_result = run_sa(
        initial_temperature=3.5,
        cooling_rate=0.995,
        step_size=0.6,
        steps=args.steps,
        seed=args.seed,
        initial_state=args.initial_state,
    )
    hc_result = run_hill_climb(
        step_size=0.6,
        steps=args.steps,
        seed=args.seed,
        initial_state=args.initial_state,
    )

    plot_temperature_trace(sa_result, output_dir)
    plot_energy_traces(sa_result, hc_result, output_dir)
    plot_single_run_sa_traces(sa_result, output_dir)
    plot_acceptance_metrics(sa_result, output_dir)
    plot_landscape_trajectory(sa_result, hc_result, output_dir)
    plot_energy_histograms_by_temperature(sa_result, output_dir)

    repeated_sa_results = [
        run_sa(
            initial_temperature=3.5,
            cooling_rate=0.995,
            step_size=0.6,
            steps=args.steps,
            seed=args.seed + repeat,
            initial_state=args.initial_state,
        )
        for repeat in range(24)
    ]
    repeated_hc_results = [
        run_hill_climb(
            step_size=0.6,
            steps=args.steps,
            seed=args.seed + repeat,
            initial_state=args.initial_state,
        )
        for repeat in range(24)
    ]

    global_minimum_info = estimate_global_minimum()
    plot_final_state_distribution(repeated_sa_results, output_dir)
    plot_sa_vs_hill_summary(repeated_sa_results, repeated_hc_results, output_dir)
    uphill_fraction_rows = compute_uphill_fraction_by_temperature_band(repeated_sa_results)
    plot_uphill_fraction_by_temperature_band(uphill_fraction_rows, output_dir)

    cooling_rows = repeated_sa_sweep(
        "cooling_rate",
        GEOMETRIC_ALPHAS,
        repeats=12,
        steps=args.steps,
        initial_state=args.initial_state,
        base_seed=args.seed + 1000,
        global_minimum_energy=global_minimum_info["global_minimum_energy"],
    )
    init_temp_rows = repeated_sa_sweep(
        "initial_temperature",
        [0.8, 1.2, 2.0, 3.5, 5.0, 7.0],
        repeats=12,
        steps=args.steps,
        initial_state=args.initial_state,
        base_seed=args.seed + 3000,
        global_minimum_energy=global_minimum_info["global_minimum_energy"],
    )
    step_size_rows = repeated_sa_sweep(
        "step_size",
        [0.15, 0.3, 0.45, 0.6, 0.9, 1.2],
        repeats=12,
        steps=args.steps,
        initial_state=args.initial_state,
        base_seed=args.seed + 5000,
        global_minimum_energy=global_minimum_info["global_minimum_energy"],
    )

    plot_parameter_sweep(
        cooling_rows,
        parameter_key="cooling_rate",
        title="Final Best Energy vs Cooling Rate",
        xlabel="Cooling rate",
        filename="final_best_energy_vs_cooling_rate.png",
        output_dir=output_dir,
    )
    plot_geometric_alpha_study(cooling_rows, output_dir)
    plot_parameter_sweep(
        init_temp_rows,
        parameter_key="initial_temperature",
        title="Final Best Energy vs Initial Temperature",
        xlabel="Initial temperature",
        filename="final_best_energy_vs_initial_temperature.png",
        output_dir=output_dir,
    )
    plot_parameter_sweep(
        step_size_rows,
        parameter_key="step_size",
        title="Final Best Energy vs Proposal Step Size",
        xlabel="Proposal step size",
        filename="final_best_energy_vs_step_size.png",
        output_dir=output_dir,
    )

    trajectory_rows = [
        {
            "step_index": step.step_index,
            "temperature": step.temperature,
            "state": step.state,
            "energy": step.energy,
            "best_energy": step.best_energy,
            "accepted": int(step.accepted),
            "uphill_move_accepted": int(step.uphill_move_accepted),
        }
        for step in sa_result.trajectory
    ]
    save_csv(output_dir / "sa_single_run_trajectory.csv", trajectory_rows)
    save_csv(output_dir / "cooling_rate_sweep.csv", cooling_rows)
    save_csv(output_dir / "initial_temperature_sweep.csv", init_temp_rows)
    save_csv(output_dir / "step_size_sweep.csv", step_size_rows)
    save_csv(output_dir / "uphill_fraction_by_temperature_band.csv", uphill_fraction_rows)
    save_csv(output_dir / "global_minimum_reference.csv", [global_minimum_info])
    summary_rows = [
        summarize_runs(
            repeated_sa_results,
            "simulated_annealing",
            global_minimum_energy=global_minimum_info["global_minimum_energy"],
        ),
        summarize_runs(
            repeated_hc_results,
            "hill_climbing",
            global_minimum_energy=global_minimum_info["global_minimum_energy"],
        ),
    ]
    save_csv(
        output_dir / "algorithm_summary.csv",
        summary_rows,
    )

    print("global_minimizer_x\tglobal_minimum_energy")
    print(f"{global_minimum_info['global_minimizer_x']:.6f}\t{global_minimum_info['global_minimum_energy']:.6f}")
    print()
    print(
        "algorithm\tmean_final_energy\tmean_final_energy_gap_to_global_minimum\t"
        "mean_best_energy\tmean_best_energy_gap_to_global_minimum\tstd_best_energy\t"
        "mean_acceptance_rate\tmean_uphill_accepted"
    )
    for row in summary_rows:
        print(
            f"{row['algorithm']}\t"
            f"{row['mean_final_energy']:.6f}\t"
            f"{row['mean_final_energy_gap_to_global_minimum']:.6f}\t"
            f"{row['mean_best_energy']:.6f}\t"
            f"{row['mean_best_energy_gap_to_global_minimum']:.6f}\t"
            f"{row['std_best_energy']:.6f}\t"
            f"{row['mean_acceptance_rate']:.6f}\t"
            f"{row['mean_uphill_accepted']:.6f}"
        )

    print()
    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
