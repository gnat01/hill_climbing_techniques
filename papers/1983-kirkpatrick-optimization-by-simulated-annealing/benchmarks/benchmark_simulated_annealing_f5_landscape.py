"""Benchmark simulated annealing on the custom discrete f5 landscape.

This benchmark studies the shifted block-rugged family

    f_5(x) = floor(2 (x-R)^2 / k) + floor(c * (((x-R) mod k)^2))

on x in {0, 1, ..., M-1}, with optional pointwise Gaussian perturbations.

If Gaussian noise is enabled, one realization is sampled once per experiment and
is then held fixed for every algorithm run so the comparison is fair.
"""

from __future__ import annotations

import argparse
import csv
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

from python.simulated_annealing import (  # noqa: E402
    SimulatedAnnealingOptimizer,
    geometric_schedule,
    greedy_hill_climb,
)


@dataclass(frozen=True)
class FixedLandscape:
    M: int
    k: int
    c: float
    R: int
    gaussian_noise: bool
    noise_mean: float
    noise_sd: float
    seed: int
    base_values: tuple[float, ...]
    noise_values: tuple[float, ...]
    total_values: tuple[float, ...]

    def energy(self, state: int) -> float:
        return self.total_values[state]

    def global_minimum_state(self) -> int:
        return min(range(self.M), key=lambda idx: self.total_values[idx])

    def global_minimum_energy(self) -> float:
        return min(self.total_values)


def f5_base_value(x: int, M: int, k: int, c: float, R: int) -> float:
    y = x - R
    return math.floor(2 * (y**2) / k) + math.floor(c * ((y % k) ** 2))


def build_fixed_landscape(
    *,
    M: int,
    k: int,
    c: float,
    R: int,
    gaussian_noise: bool,
    noise_mean: float,
    noise_sd: float,
    seed: int,
) -> FixedLandscape:
    base_values = [f5_base_value(x=x, M=M, k=k, c=c, R=R) for x in range(M)]
    rng = random.Random(seed)
    if gaussian_noise:
        noise_values = [rng.gauss(noise_mean, noise_sd) for _ in range(M)]
    else:
        noise_values = [0.0 for _ in range(M)]
    total_values = [base + noise for base, noise in zip(base_values, noise_values)]
    return FixedLandscape(
        M=M,
        k=k,
        c=c,
        R=R,
        gaussian_noise=gaussian_noise,
        noise_mean=noise_mean,
        noise_sd=noise_sd,
        seed=seed,
        base_values=tuple(base_values),
        noise_values=tuple(noise_values),
        total_values=tuple(total_values),
    )


def proposal_fn_factory(M: int):
    def propose(state: int, rng: random.Random) -> int:
        if state == 0:
            return 1
        if state == M - 1:
            return M - 2
        return state + rng.choice((-1, 1))

    return propose


def save_csv(path: Path, rows: list[dict[str, float | int | str | bool]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def rolling_mean(values: list[float], window: int) -> list[float]:
    result: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        chunk = values[start : index + 1]
        result.append(statistics.fmean(chunk))
    return result


def run_sa(*, landscape: FixedLandscape, initial_state: int, steps: int, seed: int, initial_temperature: float, alpha: float):
    optimizer = SimulatedAnnealingOptimizer(
        energy_fn=landscape.energy,
        proposal_fn=proposal_fn_factory(landscape.M),
        schedule_fn=geometric_schedule(initial_temperature=initial_temperature, cooling_rate=alpha),
        rng=random.Random(seed),
    )
    return optimizer.run(initial_state=initial_state, steps=steps)


def run_hill_climb(*, landscape: FixedLandscape, initial_state: int, steps: int, seed: int):
    return greedy_hill_climb(
        initial_state=initial_state,
        energy_fn=landscape.energy,
        proposal_fn=proposal_fn_factory(landscape.M),
        steps=steps,
        rng=random.Random(seed),
    )


def result_rows(name: str, result, landscape: FixedLandscape) -> list[dict[str, float | int | str | bool]]:
    rows = []
    for step in result.trajectory:
        rows.append(
            {
                "algorithm": name,
                "step_index": step.step_index,
                "state": step.state,
                "energy": step.energy,
                "best_state": step.best_state,
                "best_energy": step.best_energy,
                "temperature": step.temperature,
                "accepted": step.accepted,
                "delta_energy": step.delta_energy,
                "uphill_move_accepted": step.uphill_move_accepted,
                "global_minimum_state": landscape.global_minimum_state(),
                "global_minimum_energy": landscape.global_minimum_energy(),
            }
        )
    return rows


def plot_landscape(landscape: FixedLandscape, sa_result, hc_result, output_dir: Path) -> None:
    xs = list(range(landscape.M))

    plt.figure(figsize=(11, 5))
    plt.plot(xs, landscape.base_values, label="Base f5 landscape", linewidth=1.5, alpha=0.8)
    if landscape.gaussian_noise:
        plt.plot(xs, landscape.total_values, label="Noisy fixed landscape", linewidth=1.7, color="tab:red")
    else:
        plt.plot(xs, landscape.total_values, label="Total landscape", linewidth=1.7, color="tab:red")
    plt.axvline(landscape.R, linestyle=":", color="black", alpha=0.5, label=f"R={landscape.R}")

    sa_states = [step.state for step in sa_result.trajectory]
    sa_energies = [landscape.energy(x) for x in sa_states]
    hc_states = [step.state for step in hc_result.trajectory]
    hc_energies = [landscape.energy(x) for x in hc_states]
    stride_sa = max(1, len(sa_states) // 120)
    stride_hc = max(1, len(hc_states) // 120)
    plt.scatter(sa_states[::stride_sa], sa_energies[::stride_sa], s=16, alpha=0.65, label="SA trajectory")
    plt.scatter(hc_states[::stride_hc], hc_energies[::stride_hc], s=16, alpha=0.65, label="Greedy trajectory")

    plt.xlabel("State x")
    plt.ylabel("Energy")
    plt.title("Custom f5 Landscape with Greedy and SA Trajectories")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "f5_landscape_with_trajectories.png", dpi=160)
    plt.close()


def plot_state_and_energy_traces(sa_result, hc_result, output_dir: Path) -> None:
    steps = [step.step_index for step in sa_result.trajectory]
    sa_states = [step.state for step in sa_result.trajectory]
    hc_states = [step.state for step in hc_result.trajectory]
    sa_energy = [step.energy for step in sa_result.trajectory]
    hc_energy = [step.energy for step in hc_result.trajectory]
    sa_best = [step.best_energy for step in sa_result.trajectory]
    hc_best = [step.best_energy for step in hc_result.trajectory]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(steps, sa_states, label="SA state", linewidth=1.5)
    axes[0].plot(steps, hc_states, label="Greedy state", linewidth=1.2)
    axes[0].set_ylabel("State")
    axes[0].set_title("State vs Iteration")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, sa_energy, label="SA current energy", alpha=0.8)
    axes[1].plot(steps, sa_best, label="SA best energy", linewidth=1.8)
    axes[1].plot(steps, hc_energy, label="Greedy current energy", alpha=0.8)
    axes[1].plot(steps, hc_best, label="Greedy best energy", linewidth=1.8, linestyle="--")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Energy")
    axes[1].set_title("Current and Best Energy vs Iteration")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "f5_state_and_energy_traces.png", dpi=160)
    plt.close(fig)


def plot_temperature_and_acceptance(sa_result, output_dir: Path) -> None:
    steps = [step.step_index for step in sa_result.trajectory]
    temperatures = [step.temperature for step in sa_result.trajectory]
    acceptance = [1.0 if step.accepted else 0.0 for step in sa_result.trajectory]
    uphill = [1.0 if step.uphill_move_accepted else 0.0 for step in sa_result.trajectory]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(steps, temperatures, linewidth=1.5)
    axes[0].set_ylabel("Temperature")
    axes[0].set_title("Temperature vs Iteration")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, rolling_mean(acceptance, window=50), label="Rolling acceptance", linewidth=1.5)
    axes[1].plot(steps, rolling_mean(uphill, window=50), label="Rolling uphill acceptance", linewidth=1.5)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Rate")
    axes[1].set_title("Rolling Acceptance Metrics (window=50)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "f5_temperature_and_acceptance.png", dpi=160)
    plt.close(fig)


def plot_repeated_run_summary(rows: list[dict[str, float | str]], output_dir: Path) -> None:
    labels = [row["algorithm"] for row in rows]
    x_values = list(range(len(rows)))
    mean_best = [row["mean_best_energy"] for row in rows]
    stderr_best = [row["stderr_best_energy"] for row in rows]
    mean_gap = [row["mean_gap_to_global_minimum"] for row in rows]
    stderr_gap = [row["stderr_gap_to_global_minimum"] for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].errorbar(x_values, mean_best, yerr=stderr_best, fmt="o", capsize=5, linewidth=1.5)
    axes[0].set_ylabel("Mean best energy")
    axes[0].set_title("Repeated-Run Best Energy")
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(x_values, mean_gap, yerr=stderr_gap, fmt="o", capsize=5, linewidth=1.5, color="tab:red")
    axes[1].set_xticks(x_values, labels)
    axes[1].set_ylabel("Mean gap to global minimum")
    axes[1].set_title("Repeated-Run Gap to Global Minimum")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "f5_repeated_run_summary.png", dpi=160)
    plt.close(fig)


def summarize_repeated_runs(landscape: FixedLandscape, *, initial_state: int, steps: int, repeats: int, initial_temperature: float, alpha: float, base_seed: int):
    summaries = []
    for name, runner in [("greedy", run_hill_climb), ("simulated_annealing", run_sa)]:
        best_energies = []
        gaps = []
        for repeat in range(repeats):
            seed = base_seed + (10000 if name == "simulated_annealing" else 0) + repeat
            if name == "greedy":
                result = runner(landscape=landscape, initial_state=initial_state, steps=steps, seed=seed)
            else:
                result = runner(
                    landscape=landscape,
                    initial_state=initial_state,
                    steps=steps,
                    seed=seed,
                    initial_temperature=initial_temperature,
                    alpha=alpha,
                )
            best_energies.append(result.best_energy)
            gaps.append(result.best_energy - landscape.global_minimum_energy())

        summaries.append(
            {
                "algorithm": name,
                "mean_best_energy": statistics.fmean(best_energies),
                "stderr_best_energy": statistics.pstdev(best_energies) / math.sqrt(len(best_energies)) if len(best_energies) > 1 else 0.0,
                "mean_gap_to_global_minimum": statistics.fmean(gaps),
                "stderr_gap_to_global_minimum": statistics.pstdev(gaps) / math.sqrt(len(gaps)) if len(gaps) > 1 else 0.0,
                "repeats": repeats,
            }
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SA on the custom discrete f5 landscape.")
    parser.add_argument("--M", type=int, default=120, help="Number of discrete states x in {0, ..., M-1}.")
    parser.add_argument("--k", type=int, default=20, help="Block width parameter. Must divide M.")
    parser.add_argument("--c", type=float, default=0.4, help="Block-ruggedness scale.")
    parser.add_argument("--R", type=int, default=60, help="Landscape shift.")
    parser.add_argument("--steps", type=int, default=800, help="Number of iterations for greedy and SA.")
    parser.add_argument("--initial-state", type=int, default=10, help="Starting state for both algorithms.")
    parser.add_argument("--initial-temperature", type=float, default=40.0, help="Initial SA temperature.")
    parser.add_argument("--alpha", type=float, default=0.98, help="Geometric cooling factor for SA.")
    parser.add_argument("--repeats", type=int, default=40, help="Repeated runs per algorithm for the summary plot.")
    parser.add_argument("--seed", type=int, default=123, help="Base seed for the fixed landscape realization and repeated runs.")
    parser.add_argument("--gaussian-noise", action="store_true", help="Add one fixed Gaussian perturbation to the landscape.")
    parser.add_argument("--noise-mean", type=float, default=None, help="Mean of the Gaussian perturbation if enabled.")
    parser.add_argument("--noise-sd", type=float, default=None, help="Standard deviation of the Gaussian perturbation if enabled.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "f5_outputs",
        help="Directory for benchmark outputs.",
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
    if not 0 <= args.initial_state < args.M:
        raise ValueError("initial-state must lie in [0, M-1]")
    if args.steps < 0:
        raise ValueError("steps must be non-negative")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if args.initial_temperature <= 0.0:
        raise ValueError("initial-temperature must be positive")
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
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

    sa_result = run_sa(
        landscape=landscape,
        initial_state=args.initial_state,
        steps=args.steps,
        seed=args.seed + 1000,
        initial_temperature=args.initial_temperature,
        alpha=args.alpha,
    )
    hc_result = run_hill_climb(
        landscape=landscape,
        initial_state=args.initial_state,
        steps=args.steps,
        seed=args.seed + 2000,
    )

    plot_landscape(landscape, sa_result, hc_result, output_dir)
    plot_state_and_energy_traces(sa_result, hc_result, output_dir)
    plot_temperature_and_acceptance(sa_result, output_dir)

    landscape_rows = [
        {
            "x": x,
            "base_f_x": landscape.base_values[x],
            "noise": landscape.noise_values[x],
            "f_x": landscape.total_values[x],
            "global_minimum_state": landscape.global_minimum_state(),
            "global_minimum_energy": landscape.global_minimum_energy(),
        }
        for x in range(landscape.M)
    ]
    save_csv(output_dir / "f5_landscape.csv", landscape_rows)
    save_csv(output_dir / "f5_sa_trajectory.csv", result_rows("simulated_annealing", sa_result, landscape))
    save_csv(output_dir / "f5_greedy_trajectory.csv", result_rows("greedy", hc_result, landscape))

    repeated_summary = summarize_repeated_runs(
        landscape,
        initial_state=args.initial_state,
        steps=args.steps,
        repeats=args.repeats,
        initial_temperature=args.initial_temperature,
        alpha=args.alpha,
        base_seed=args.seed,
    )
    save_csv(output_dir / "f5_repeated_run_summary.csv", repeated_summary)
    plot_repeated_run_summary(repeated_summary, output_dir)

    config_rows = [
        {
            "M": args.M,
            "k": args.k,
            "c": args.c,
            "R": args.R,
            "steps": args.steps,
            "initial_state": args.initial_state,
            "initial_temperature": args.initial_temperature,
            "alpha": args.alpha,
            "repeats": args.repeats,
            "seed": args.seed,
            "gaussian_noise": args.gaussian_noise,
            "noise_mean": args.noise_mean,
            "noise_sd": args.noise_sd,
            "global_minimum_state": landscape.global_minimum_state(),
            "global_minimum_energy": landscape.global_minimum_energy(),
        }
    ]
    save_csv(output_dir / "f5_benchmark_config.csv", config_rows)

    print("algorithm\tbest_energy\tbest_state\tgap_to_global_minimum")
    print(
        "greedy"
        f"\t{hc_result.best_energy:.6f}\t{hc_result.best_state}\t"
        f"{hc_result.best_energy - landscape.global_minimum_energy():.6f}"
    )
    print(
        "simulated_annealing"
        f"\t{sa_result.best_energy:.6f}\t{sa_result.best_state}\t"
        f"{sa_result.best_energy - landscape.global_minimum_energy():.6f}"
    )
    print(f"\nGlobal minimum state: {landscape.global_minimum_state()}")
    print(f"Global minimum energy: {landscape.global_minimum_energy():.6f}")


if __name__ == "__main__":
    main()
