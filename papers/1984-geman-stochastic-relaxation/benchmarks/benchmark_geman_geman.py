"""Benchmark and visualize binary-image denoising for Geman-Geman."""

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


def plot_example_montage(example_name: str, truth, noisy, gibbs_result, annealed_result, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    show_spin(axes[0], truth, "Clean")
    show_spin(axes[1], noisy, "Noisy")
    show_spin(axes[2], gibbs_result.final_state, "Fixed-T Gibbs")
    show_spin(axes[3], annealed_result.final_state, "Annealed")
    fig.suptitle(f"{example_name}: denoising comparison")
    fig.tight_layout()
    fig.savefig(output_dir / f"{example_name}_montage.png", dpi=160)
    plt.close(fig)


def plot_trace_panels(example_name: str, gibbs_result, annealed_result, output_dir: Path) -> None:
    xs_g = [step.step_index for step in gibbs_result.trajectory]
    xs_a = [step.step_index for step in annealed_result.trajectory]

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(xs_g, [step.energy for step in gibbs_result.trajectory], label="Fixed-T Gibbs")
    axes[0].plot(xs_a, [step.energy for step in annealed_result.trajectory], label="Annealed")
    axes[0].set_ylabel("Posterior energy")
    axes[0].set_title(f"{example_name}: energy / accuracy / temperature")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(xs_g, [step.pixel_accuracy for step in gibbs_result.trajectory], label="Fixed-T Gibbs")
    axes[1].plot(xs_a, [step.pixel_accuracy for step in annealed_result.trajectory], label="Annealed")
    axes[1].set_ylabel("Pixel accuracy")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(xs_g, [step.temperature for step in gibbs_result.trajectory], label="Fixed-T Gibbs")
    axes[2].plot(xs_a, [step.temperature for step in annealed_result.trajectory], label="Annealed")
    axes[2].set_xlabel("Sweep")
    axes[2].set_ylabel("Temperature")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / f"{example_name}_traces.png", dpi=160)
    plt.close(fig)


def plot_summary(rows, output_dir: Path) -> None:
    examples = [row["example"] for row in rows]
    x = list(range(len(rows)))
    noisy_acc = [row["noisy_accuracy"] for row in rows]
    gibbs_acc = [row["fixed_gibbs_accuracy"] for row in rows]
    annealed_acc = [row["annealed_accuracy"] for row in rows]

    plt.figure(figsize=(10, 5))
    plt.plot(x, noisy_acc, marker="o", label="Noisy")
    plt.plot(x, gibbs_acc, marker="o", label="Fixed-T Gibbs")
    plt.plot(x, annealed_acc, marker="o", label="Annealed")
    plt.xticks(x, examples, rotation=15, ha="right")
    plt.ylabel("Pixel accuracy")
    plt.title("Denoising accuracy across examples")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_summary_across_examples.png", dpi=160)
    plt.close()


def plot_limitation_case(output_dir: Path, size: int, noise: float, sweeps: int) -> dict[str, float | str]:
    truth = example_stripes(size)
    noisy = flip_noise(truth, noise, np.random.default_rng(777))
    gibbs_result = GibbsImageRestorer(
        observation=noisy,
        eta=2.2,
        coupling=1.1,
        schedule_fn=fixed_temperature_schedule(1.2),
        rng=np.random.default_rng(778),
    ).run(noisy, sweeps=sweeps, truth=truth)
    annealed_result = GibbsImageRestorer(
        observation=noisy,
        eta=2.2,
        coupling=1.1,
        schedule_fn=geometric_schedule(2.8, 0.95),
        rng=np.random.default_rng(779),
    ).run(noisy, sweeps=sweeps, truth=truth)

    plot_example_montage("stripes_limitation", truth, noisy, gibbs_result, annealed_result, output_dir)
    plot_trace_panels("stripes_limitation", gibbs_result, annealed_result, output_dir)
    return {
        "example": "stripes_limitation",
        "noisy_accuracy": pixel_accuracy(noisy, truth),
        "fixed_gibbs_accuracy": gibbs_result.final_accuracy,
        "annealed_accuracy": annealed_result.final_accuracy,
    }


def plot_noise_level_study(example_name: str, truth: np.ndarray, output_dir: Path) -> list[dict[str, float | str]]:
    rows = []
    for noise_level in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
        noisy = flip_noise(truth, noise_level, np.random.default_rng(1000 + int(noise_level * 1000)))
        gibbs = GibbsImageRestorer(
            observation=noisy,
            eta=2.2,
            coupling=1.1,
            schedule_fn=fixed_temperature_schedule(1.2),
            rng=np.random.default_rng(2000 + int(noise_level * 1000)),
        ).run(noisy, sweeps=40, truth=truth)
        annealed = GibbsImageRestorer(
            observation=noisy,
            eta=2.2,
            coupling=1.1,
            schedule_fn=geometric_schedule(2.8, 0.95),
            rng=np.random.default_rng(3000 + int(noise_level * 1000)),
        ).run(noisy, sweeps=40, truth=truth)
        rows.append(
            {
                "example": example_name,
                "noise_level": noise_level,
                "noisy_accuracy": pixel_accuracy(noisy, truth),
                "fixed_gibbs_accuracy": annealed.final_accuracy if False else gibbs.final_accuracy,
                "annealed_accuracy": annealed.final_accuracy,
            }
        )

    plt.figure(figsize=(9, 5))
    plt.plot([row["noise_level"] for row in rows], [row["noisy_accuracy"] for row in rows], marker="o", label="Noisy")
    plt.plot(
        [row["noise_level"] for row in rows],
        [row["fixed_gibbs_accuracy"] for row in rows],
        marker="o",
        label="Fixed-T Gibbs",
    )
    plt.plot(
        [row["noise_level"] for row in rows],
        [row["annealed_accuracy"] for row in rows],
        marker="o",
        label="Annealed",
    )
    plt.xlabel("Flip probability")
    plt.ylabel("Pixel accuracy")
    plt.title(f"{example_name}: denoising accuracy vs noise level")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{example_name}_noise_level_study.png", dpi=160)
    plt.close()
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Geman-Geman image denoising.")
    parser.add_argument("--size", type=int, default=32, help="Image size for procedural examples.")
    parser.add_argument("--noise", type=float, default=0.2, help="Pixel-flip probability for the main examples.")
    parser.add_argument("--sweeps", type=int, default=40, help="Number of Gibbs sweeps.")
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

    rows = []
    examples = all_examples(args.size)
    for index, (example_name, truth) in enumerate(examples.items()):
        noisy = flip_noise(truth, args.noise, np.random.default_rng(100 + index))
        gibbs_result = GibbsImageRestorer(
            observation=noisy,
            eta=2.2,
            coupling=1.1,
            schedule_fn=fixed_temperature_schedule(1.2),
            rng=np.random.default_rng(1000 + index),
        ).run(noisy, sweeps=args.sweeps, truth=truth)
        annealed_result = GibbsImageRestorer(
            observation=noisy,
            eta=2.2,
            coupling=1.1,
            schedule_fn=geometric_schedule(2.8, 0.95),
            rng=np.random.default_rng(2000 + index),
        ).run(noisy, sweeps=args.sweeps, truth=truth)

        plot_example_montage(example_name, truth, noisy, gibbs_result, annealed_result, output_dir)
        plot_trace_panels(example_name, gibbs_result, annealed_result, output_dir)

        rows.append(
            {
                "example": example_name,
                "noisy_accuracy": pixel_accuracy(noisy, truth),
                "fixed_gibbs_accuracy": gibbs_result.final_accuracy,
                "annealed_accuracy": annealed_result.final_accuracy,
                "fixed_gibbs_final_energy": gibbs_result.final_energy,
                "annealed_final_energy": annealed_result.final_energy,
            }
        )

    plot_summary(rows, output_dir)
    save_csv(output_dir / "example_summary.csv", rows)

    noise_rows = plot_noise_level_study("square", examples["square"], output_dir)
    save_csv(output_dir / "square_noise_level_study.csv", noise_rows)
    limitation_row = plot_limitation_case(output_dir, args.size, args.noise, args.sweeps)
    save_csv(output_dir / "limitation_case_summary.csv", [limitation_row])

    print("example\tnoisy_accuracy\tfixed_gibbs_accuracy\tannealed_accuracy")
    for row in rows:
        print(
            f"{row['example']}\t{row['noisy_accuracy']:.4f}\t"
            f"{row['fixed_gibbs_accuracy']:.4f}\t{row['annealed_accuracy']:.4f}"
        )
    print()
    print(
        f"{limitation_row['example']}\t{limitation_row['noisy_accuracy']:.4f}\t"
        f"{limitation_row['fixed_gibbs_accuracy']:.4f}\t{limitation_row['annealed_accuracy']:.4f}"
    )
    print()
    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
