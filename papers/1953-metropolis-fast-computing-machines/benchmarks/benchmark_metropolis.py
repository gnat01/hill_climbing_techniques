"""Benchmark and visualize the Metropolis sampler on a double-well landscape."""

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

from python.metropolis import (  # noqa: E402
    MetropolisSampler,
    double_well_energy,
    random_walk_proposal,
)


DEFAULT_TEMPERATURES = [
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
]


def run_trial(temperature: float, seed: int, steps: int, burn_in: int) -> dict[str, float]:
    sampler = MetropolisSampler(
        energy_fn=double_well_energy,
        proposal_fn=random_walk_proposal(step_size=0.75),
        temperature=temperature,
        rng=random.Random(seed),
    )
    result = sampler.run(initial_state=0.0, steps=steps)
    post_burn_in = result.trajectory[burn_in:]
    energies = [step.energy for step in post_burn_in]
    positions = [step.state for step in post_burn_in]
    return {
        "temperature": temperature,
        "acceptance_rate": result.acceptance_rate,
        "mean_energy": statistics.fmean(energies),
        "mean_abs_position": statistics.fmean(abs(x) for x in positions),
    }


def run_suite(temperatures: list[float], steps: int, burn_in: int, base_seed: int) -> list[dict[str, float]]:
    rows = []
    for index, temperature in enumerate(temperatures):
        rows.append(
            run_trial(
                temperature=temperature,
                seed=base_seed + index,
                steps=steps,
                burn_in=burn_in,
            )
        )
    return rows


def print_table(rows: list[dict[str, float]]) -> None:
    print("temperature\tacceptance_rate\tmean_energy\tmean_abs_position")
    for metrics in rows:
        print(
            f"{metrics['temperature']:.2f}\t"
            f"{metrics['acceptance_rate']:.4f}\t"
            f"{metrics['mean_energy']:.4f}\t"
            f"{metrics['mean_abs_position']:.4f}"
        )


def write_csv(rows: list[dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["temperature", "acceptance_rate", "mean_energy", "mean_abs_position"],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(
    rows: list[dict[str, float]],
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    temperatures = [row["temperature"] for row in rows]
    values = [row[metric] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(temperatures, values, marker="o", linewidth=1.8)
    plt.xscale("log")
    plt.xlabel("Temperature")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark and visualize fixed-temperature Metropolis sampling."
    )
    parser.add_argument("--steps", type=int, default=5000, help="Number of Metropolis steps per temperature.")
    parser.add_argument("--burn-in", type=int, default=1000, help="Number of initial steps to discard.")
    parser.add_argument("--seed", type=int, default=100, help="Base seed for temperature sweeps.")
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
    if args.burn_in < 0 or args.burn_in >= args.steps:
        raise ValueError("burn-in must satisfy 0 <= burn_in < steps")

    rows = run_suite(
        temperatures=DEFAULT_TEMPERATURES,
        steps=args.steps,
        burn_in=args.burn_in,
        base_seed=args.seed,
    )
    print_table(rows)

    output_dir = args.output_dir
    write_csv(rows, output_dir / "metropolis_temperature_sweep.csv")
    plot_metric(
        rows,
        metric="acceptance_rate",
        title="Metropolis Acceptance Rate vs Temperature",
        ylabel="Acceptance rate",
        output_path=output_dir / "acceptance_rate_vs_temperature.png",
    )
    plot_metric(
        rows,
        metric="mean_energy",
        title="Metropolis Mean Energy vs Temperature",
        ylabel="Mean sampled energy",
        output_path=output_dir / "mean_energy_vs_temperature.png",
    )
    plot_metric(
        rows,
        metric="mean_abs_position",
        title="Metropolis Mean |x| vs Temperature",
        ylabel="Mean sampled |x|",
        output_path=output_dir / "mean_abs_position_vs_temperature.png",
    )

    print()
    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
