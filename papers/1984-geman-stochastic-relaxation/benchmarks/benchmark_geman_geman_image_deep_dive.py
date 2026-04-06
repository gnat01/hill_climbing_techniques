"""Deep-dive parameter study for bespoke-image Geman-Geman denoising."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
from PIL import Image

from python.geman_geman import (  # noqa: E402
    GibbsImageRestorer,
    fixed_temperature_schedule,
    flip_noise,
    geometric_schedule,
    otsu_threshold_from_grayscale,
    pixel_accuracy,
    to_spin,
)


def save_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_grayscale(path: Path, max_dim: int | None) -> np.ndarray:
    with Image.open(path) as image:
        gray = image.convert("L")
        width, height = gray.size
        if max_dim is not None and max(width, height) > max_dim:
            scale = max_dim / max(width, height)
            resized = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            gray = gray.resize(resized, Image.Resampling.LANCZOS)
        return np.array(gray, dtype=np.uint8)


def grayscale_to_spin(grayscale: np.ndarray, threshold: int, invert: bool) -> np.ndarray:
    binary01 = (grayscale >= threshold).astype(np.uint8)
    if invert:
        binary01 = 1 - binary01
    return to_spin(binary01)


def stderr(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.pstdev(values) / math.sqrt(len(values))


def run_once(
    *,
    truth: np.ndarray,
    noise: float,
    sweeps: int,
    eta: float,
    coupling: float,
    fixed_temperature: float,
    anneal_initial_temperature: float,
    anneal_alpha: float,
    seed_offset: int,
) -> dict[str, float]:
    noisy = flip_noise(truth, noise, np.random.default_rng(1000 + seed_offset))
    noisy_accuracy = pixel_accuracy(noisy, truth)

    fixed_result = GibbsImageRestorer(
        observation=noisy,
        eta=eta,
        coupling=coupling,
        schedule_fn=fixed_temperature_schedule(fixed_temperature),
        rng=np.random.default_rng(2000 + seed_offset),
    ).run(noisy, sweeps=sweeps, truth=truth)

    annealed_result = GibbsImageRestorer(
        observation=noisy,
        eta=eta,
        coupling=coupling,
        schedule_fn=geometric_schedule(anneal_initial_temperature, anneal_alpha),
        rng=np.random.default_rng(3000 + seed_offset),
    ).run(noisy, sweeps=sweeps, truth=truth)

    return {
        "noisy_accuracy": noisy_accuracy,
        "fixed_gibbs_accuracy": fixed_result.final_accuracy,
        "annealed_accuracy": annealed_result.final_accuracy,
    }


def summarize_repeats(results: list[dict[str, float]]) -> dict[str, float]:
    return {
        "mean_noisy_accuracy": statistics.fmean(item["noisy_accuracy"] for item in results),
        "mean_fixed_gibbs_accuracy": statistics.fmean(item["fixed_gibbs_accuracy"] for item in results),
        "stderr_fixed_gibbs_accuracy": stderr([item["fixed_gibbs_accuracy"] for item in results]),
        "mean_annealed_accuracy": statistics.fmean(item["annealed_accuracy"] for item in results),
        "stderr_annealed_accuracy": stderr([item["annealed_accuracy"] for item in results]),
    }


def plot_accuracy_vs_threshold(rows: list[dict[str, float | int]], output_dir: Path, stem: str) -> None:
    thresholds = [row["threshold"] for row in rows]
    noisy = [row["mean_noisy_accuracy"] for row in rows]
    fixed = [row["mean_fixed_gibbs_accuracy"] for row in rows]
    fixed_err = [row["stderr_fixed_gibbs_accuracy"] for row in rows]
    annealed = [row["mean_annealed_accuracy"] for row in rows]
    annealed_err = [row["stderr_annealed_accuracy"] for row in rows]

    plt.figure(figsize=(9, 5.5))
    plt.plot(thresholds, noisy, marker="o", label="Noisy baseline")
    plt.errorbar(thresholds, fixed, yerr=fixed_err, fmt="o-", capsize=4, label="Fixed-T Gibbs")
    plt.errorbar(thresholds, annealed, yerr=annealed_err, fmt="o-", capsize=4, label="Annealed")
    plt.xlabel("Binarization threshold")
    plt.ylabel("Pixel accuracy")
    plt.title(f"{stem}: accuracy vs threshold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}_accuracy_vs_threshold.png", dpi=160)
    plt.close()


def plot_accuracy_vs_sweeps(rows: list[dict[str, float | int]], output_dir: Path, stem: str) -> None:
    sweeps = [row["sweeps"] for row in rows]
    noisy = [row["mean_noisy_accuracy"] for row in rows]
    fixed = [row["mean_fixed_gibbs_accuracy"] for row in rows]
    fixed_err = [row["stderr_fixed_gibbs_accuracy"] for row in rows]
    annealed = [row["mean_annealed_accuracy"] for row in rows]
    annealed_err = [row["stderr_annealed_accuracy"] for row in rows]

    plt.figure(figsize=(9, 5.5))
    plt.plot(sweeps, noisy, marker="o", label="Noisy baseline")
    plt.errorbar(sweeps, fixed, yerr=fixed_err, fmt="o-", capsize=4, label="Fixed-T Gibbs")
    plt.errorbar(sweeps, annealed, yerr=annealed_err, fmt="o-", capsize=4, label="Annealed")
    plt.xlabel("Number of sweeps")
    plt.ylabel("Pixel accuracy")
    plt.title(f"{stem}: accuracy vs sweeps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}_accuracy_vs_sweeps.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep-dive study for bespoke-image Geman-Geman denoising.")
    parser.add_argument("--input-image", type=Path, required=True, help="Path to the source image.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for deep-dive outputs.")
    parser.add_argument("--max-dim", type=int, default=128, help="Optional maximum resized image dimension.")
    parser.add_argument("--invert", action="store_true", help="Invert the binarized image.")
    parser.add_argument("--noise", type=float, default=0.2, help="Pixel-flip probability.")
    parser.add_argument("--eta", type=float, default=2.2, help="Data-fidelity weight.")
    parser.add_argument("--coupling", type=float, default=1.1, help="Smoothness weight.")
    parser.add_argument("--fixed-temperature", type=float, default=1.2, help="Fixed Gibbs temperature.")
    parser.add_argument("--anneal-initial-temperature", type=float, default=2.8, help="Initial annealing temperature.")
    parser.add_argument("--anneal-alpha", type=float, default=0.95, help="Annealing alpha.")
    parser.add_argument("--repeats", type=int, default=4, help="Repeated runs per setting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats <= 1:
        raise ValueError("repeats must be greater than 1")

    stem = args.input_image.stem
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    grayscale = load_grayscale(args.input_image, args.max_dim)
    otsu_threshold = otsu_threshold_from_grayscale(grayscale)
    threshold_values = sorted(set(max(0, min(255, t)) for t in [
        otsu_threshold - 40,
        otsu_threshold - 20,
        otsu_threshold,
        otsu_threshold + 20,
        otsu_threshold + 40,
    ]))
    sweep_values = [5, 10, 20, 40, 80, 120]

    threshold_rows: list[dict[str, float | int]] = []
    for threshold in threshold_values:
        truth = grayscale_to_spin(grayscale, threshold, args.invert)
        repeat_results = [
            run_once(
                truth=truth,
                noise=args.noise,
                sweeps=40,
                eta=args.eta,
                coupling=args.coupling,
                fixed_temperature=args.fixed_temperature,
                anneal_initial_temperature=args.anneal_initial_temperature,
                anneal_alpha=args.anneal_alpha,
                seed_offset=repeat,
            )
            for repeat in range(args.repeats)
        ]
        summary = summarize_repeats(repeat_results)
        threshold_rows.append({"threshold": threshold, **summary})

    base_truth = grayscale_to_spin(grayscale, otsu_threshold, args.invert)
    sweep_rows: list[dict[str, float | int]] = []
    for sweeps in sweep_values:
        repeat_results = [
            run_once(
                truth=base_truth,
                noise=args.noise,
                sweeps=sweeps,
                eta=args.eta,
                coupling=args.coupling,
                fixed_temperature=args.fixed_temperature,
                anneal_initial_temperature=args.anneal_initial_temperature,
                anneal_alpha=args.anneal_alpha,
                seed_offset=500 + repeat,
            )
            for repeat in range(args.repeats)
        ]
        summary = summarize_repeats(repeat_results)
        sweep_rows.append({"sweeps": sweeps, **summary})

    save_csv(output_dir / f"{stem}_threshold_study.csv", threshold_rows)
    save_csv(output_dir / f"{stem}_sweeps_study.csv", sweep_rows)
    save_json(
        output_dir / f"{stem}_deep_dive_config.json",
        {
            "input_image": str(args.input_image),
            "output_dir": str(output_dir),
            "max_dim": args.max_dim,
            "invert": args.invert,
            "noise": args.noise,
            "eta": args.eta,
            "coupling": args.coupling,
            "fixed_temperature": args.fixed_temperature,
            "anneal_initial_temperature": args.anneal_initial_temperature,
            "anneal_alpha": args.anneal_alpha,
            "repeats": args.repeats,
            "otsu_threshold": int(otsu_threshold),
            "threshold_values": threshold_values,
            "sweep_values": sweep_values,
            "derived_width": int(grayscale.shape[1]),
            "derived_height": int(grayscale.shape[0]),
        },
    )

    plot_accuracy_vs_threshold(threshold_rows, output_dir, stem)
    plot_accuracy_vs_sweeps(sweep_rows, output_dir, stem)

    print("Threshold study")
    print("threshold\tmean_noisy_accuracy\tmean_fixed_gibbs_accuracy\tmean_annealed_accuracy")
    for row in threshold_rows:
        print(
            f"{row['threshold']}\t{row['mean_noisy_accuracy']:.4f}\t"
            f"{row['mean_fixed_gibbs_accuracy']:.4f}\t{row['mean_annealed_accuracy']:.4f}"
        )
    print()
    print("Sweeps study")
    print("sweeps\tmean_noisy_accuracy\tmean_fixed_gibbs_accuracy\tmean_annealed_accuracy")
    for row in sweep_rows:
        print(
            f"{row['sweeps']}\t{row['mean_noisy_accuracy']:.4f}\t"
            f"{row['mean_fixed_gibbs_accuracy']:.4f}\t{row['mean_annealed_accuracy']:.4f}"
        )
    print()
    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
