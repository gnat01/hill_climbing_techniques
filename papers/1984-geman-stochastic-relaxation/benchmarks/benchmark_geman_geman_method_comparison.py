"""Compare deterministic local descent, fixed-temperature Gibbs, and annealed Gibbs."""

from __future__ import annotations

import argparse
import csv
import os
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
import numpy as np

from python.geman_geman import (  # noqa: E402
    DeterministicImageRestorer,
    GibbsImageRestorer,
    all_examples,
    example_stripes,
    fixed_temperature_schedule,
    flip_noise,
    from_spin,
    geometric_schedule,
    pixel_accuracy,
)


def save_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def show_spin(ax, image: np.ndarray, title: str) -> None:
    ax.imshow(from_spin(image), cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def run_methods(truth: np.ndarray, noise: float, sweeps: int, seed_offset: int):
    noisy = flip_noise(truth, noise, np.random.default_rng(100 + seed_offset))
    deterministic = DeterministicImageRestorer(
        observation=noisy,
        eta=2.2,
        coupling=1.1,
        rng=np.random.default_rng(1000 + seed_offset),
    ).run(noisy, sweeps=sweeps, truth=truth)
    fixed_gibbs = GibbsImageRestorer(
        observation=noisy,
        eta=2.2,
        coupling=1.1,
        schedule_fn=fixed_temperature_schedule(1.2),
        rng=np.random.default_rng(2000 + seed_offset),
    ).run(noisy, sweeps=sweeps, truth=truth)
    annealed = GibbsImageRestorer(
        observation=noisy,
        eta=2.2,
        coupling=1.1,
        schedule_fn=geometric_schedule(2.8, 0.95),
        rng=np.random.default_rng(3000 + seed_offset),
    ).run(noisy, sweeps=sweeps, truth=truth)
    return noisy, deterministic, fixed_gibbs, annealed


def plot_montage(example_name: str, truth, noisy, deterministic, fixed_gibbs, annealed, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.6))
    show_spin(axes[0], truth, "Clean")
    show_spin(axes[1], noisy, "Noisy")
    show_spin(axes[2], deterministic.final_state, "Deterministic")
    show_spin(axes[3], fixed_gibbs.final_state, "Fixed-T Gibbs")
    show_spin(axes[4], annealed.final_state, "Annealed")
    fig.suptitle(f"{example_name}: method comparison")
    fig.tight_layout()
    fig.savefig(output_dir / f"{example_name}_method_comparison_montage.png", dpi=160)
    plt.close(fig)


def plot_traces(example_name: str, deterministic, fixed_gibbs, annealed, output_dir: Path) -> None:
    xs = [step.step_index for step in deterministic.trajectory]
    plt.figure(figsize=(9, 5.5))
    plt.plot(xs, [step.pixel_accuracy for step in deterministic.trajectory], label="Deterministic local descent")
    plt.plot(xs, [step.pixel_accuracy for step in fixed_gibbs.trajectory], label="Fixed-T Gibbs")
    plt.plot(xs, [step.pixel_accuracy for step in annealed.trajectory], label="Annealed Gibbs")
    plt.xlabel("Sweep")
    plt.ylabel("Pixel accuracy")
    plt.title(f"{example_name}: accuracy vs sweep")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{example_name}_accuracy_trace_comparison.png", dpi=160)
    plt.close()


def plot_summary(rows, output_dir: Path) -> None:
    examples = [row["example"] for row in rows]
    x = list(range(len(rows)))

    plt.figure(figsize=(10, 5.5))
    plt.plot(x, [row["noisy_accuracy"] for row in rows], marker="o", label="Noisy baseline")
    plt.plot(x, [row["deterministic_accuracy"] for row in rows], marker="o", label="Deterministic local descent")
    plt.plot(x, [row["fixed_gibbs_accuracy"] for row in rows], marker="o", label="Fixed-T Gibbs")
    plt.plot(x, [row["annealed_accuracy"] for row in rows], marker="o", label="Annealed Gibbs")
    plt.xticks(x, examples, rotation=15, ha="right")
    plt.ylabel("Pixel accuracy")
    plt.title("Method comparison across examples")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "method_comparison_summary.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Geman restoration methods.")
    parser.add_argument("--size", type=int, default=32, help="Image size for procedural examples.")
    parser.add_argument("--noise", type=float, default=0.2, help="Pixel-flip probability.")
    parser.add_argument("--sweeps", type=int, default=40, help="Number of sweeps for each method.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PAPER_DIR / "benchmarks" / "method_comparison_outputs",
        help="Directory for outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, (example_name, truth) in enumerate(all_examples(args.size).items()):
        noisy, deterministic, fixed_gibbs, annealed = run_methods(truth, args.noise, args.sweeps, index)
        plot_montage(example_name, truth, noisy, deterministic, fixed_gibbs, annealed, output_dir)
        plot_traces(example_name, deterministic, fixed_gibbs, annealed, output_dir)
        rows.append(
            {
                "example": example_name,
                "noisy_accuracy": pixel_accuracy(noisy, truth),
                "deterministic_accuracy": deterministic.final_accuracy,
                "fixed_gibbs_accuracy": fixed_gibbs.final_accuracy,
                "annealed_accuracy": annealed.final_accuracy,
            }
        )

    # Add the known limitation case separately as well.
    truth = example_stripes(args.size)
    noisy, deterministic, fixed_gibbs, annealed = run_methods(truth, args.noise, args.sweeps, 999)
    plot_montage("stripes_limitation", truth, noisy, deterministic, fixed_gibbs, annealed, output_dir)
    plot_traces("stripes_limitation", deterministic, fixed_gibbs, annealed, output_dir)
    rows.append(
        {
            "example": "stripes_limitation",
            "noisy_accuracy": pixel_accuracy(noisy, truth),
            "deterministic_accuracy": deterministic.final_accuracy,
            "fixed_gibbs_accuracy": fixed_gibbs.final_accuracy,
            "annealed_accuracy": annealed.final_accuracy,
        }
    )

    plot_summary(rows, output_dir)
    save_csv(output_dir / "method_comparison_summary.csv", rows)

    print("example\tnoisy_accuracy\tdeterministic_accuracy\tfixed_gibbs_accuracy\tannealed_accuracy")
    for row in rows:
        print(
            f"{row['example']}\t{row['noisy_accuracy']:.4f}\t{row['deterministic_accuracy']:.4f}\t"
            f"{row['fixed_gibbs_accuracy']:.4f}\t{row['annealed_accuracy']:.4f}"
        )
    print()
    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
