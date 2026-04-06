"""Run Geman-Geman denoising on a bespoke input image.

Pipeline:
- load image
- auto-derive size
- optionally downscale
- binarize
- add synthetic flip noise
- restore with fixed-temperature and annealed Gibbs
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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
    from_spin,
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


def load_and_binarize_image(path: Path, max_dim: int | None, threshold: int | None, invert: bool) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    with Image.open(path) as image:
        gray = image.convert("L")
        original_width, original_height = gray.size
        if max_dim is not None and max(original_width, original_height) > max_dim:
            scale = max_dim / max(original_width, original_height)
            resized_size = (max(1, int(round(original_width * scale))), max(1, int(round(original_height * scale))))
            gray = gray.resize(resized_size, Image.Resampling.LANCZOS)
        grayscale = np.array(gray, dtype=np.uint8)

    chosen_threshold = otsu_threshold_from_grayscale(grayscale) if threshold is None else threshold
    binary01 = (grayscale >= chosen_threshold).astype(np.uint8)
    if invert:
        binary01 = 1 - binary01
    return grayscale, to_spin(binary01), {
        "threshold": int(chosen_threshold),
        "width": int(binary01.shape[1]),
        "height": int(binary01.shape[0]),
    }


def show_binary(ax, image: np.ndarray, title: str) -> None:
    ax.imshow(from_spin(image), cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def show_grayscale(ax, image: np.ndarray, title: str) -> None:
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_main_montage(grayscale, clean_spin, noisy_spin, fixed_result, annealed_result, output_dir: Path, stem: str) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.8))
    show_grayscale(axes[0], grayscale, "Original grayscale")
    show_binary(axes[1], clean_spin, "Binarized clean")
    show_binary(axes[2], noisy_spin, "Noisy")
    show_binary(axes[3], fixed_result.final_state, "Fixed-T Gibbs")
    show_binary(axes[4], annealed_result.final_state, "Annealed")
    fig.suptitle(f"{stem}: image denoising pipeline")
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}_montage.png", dpi=160)
    plt.close(fig)


def plot_traces(fixed_result, annealed_result, output_dir: Path, stem: str) -> None:
    xs_f = [step.step_index for step in fixed_result.trajectory]
    xs_a = [step.step_index for step in annealed_result.trajectory]

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(xs_f, [step.energy for step in fixed_result.trajectory], label="Fixed-T Gibbs")
    axes[0].plot(xs_a, [step.energy for step in annealed_result.trajectory], label="Annealed")
    axes[0].set_ylabel("Posterior energy")
    axes[0].set_title(f"{stem}: restoration traces")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(xs_f, [step.pixel_accuracy for step in fixed_result.trajectory], label="Fixed-T Gibbs")
    axes[1].plot(xs_a, [step.pixel_accuracy for step in annealed_result.trajectory], label="Annealed")
    axes[1].set_ylabel("Pixel accuracy")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(xs_f, [step.temperature for step in fixed_result.trajectory], label="Fixed-T Gibbs")
    axes[2].plot(xs_a, [step.temperature for step in annealed_result.trajectory], label="Annealed")
    axes[2].set_xlabel("Sweep")
    axes[2].set_ylabel("Temperature")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}_traces.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Geman-Geman denoising on a bespoke image.")
    parser.add_argument("--input-image", type=Path, required=True, help="Path to the source image.")
    parser.add_argument("--noise", type=float, default=0.2, help="Pixel-flip probability after binarization.")
    parser.add_argument("--sweeps", type=int, default=40, help="Number of Gibbs sweeps.")
    parser.add_argument("--max-dim", type=int, default=128, help="Optional maximum image dimension after resizing.")
    parser.add_argument("--threshold", type=int, default=None, help="Manual grayscale threshold; defaults to Otsu.")
    parser.add_argument("--invert", action="store_true", help="Invert the binarized image.")
    parser.add_argument("--fixed-temperature", type=float, default=1.2, help="Fixed Gibbs temperature.")
    parser.add_argument("--anneal-initial-temperature", type=float, default=2.8, help="Initial annealing temperature.")
    parser.add_argument("--anneal-alpha", type=float, default=0.95, help="Geometric decay factor across sweeps.")
    parser.add_argument("--eta", type=float, default=2.2, help="Data-fidelity weight.")
    parser.add_argument("--coupling", type=float, default=1.1, help="Smoothness weight.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PAPER_DIR / "benchmarks" / "image_outputs",
        help="Directory for output images and CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stem = args.input_image.stem
    output_dir = args.output_dir / stem
    output_dir.mkdir(parents=True, exist_ok=True)

    grayscale, clean_spin, meta = load_and_binarize_image(
        args.input_image,
        max_dim=args.max_dim,
        threshold=args.threshold,
        invert=args.invert,
    )
    noisy_spin = flip_noise(clean_spin, args.noise, np.random.default_rng(123))

    fixed_result = GibbsImageRestorer(
        observation=noisy_spin,
        eta=args.eta,
        coupling=args.coupling,
        schedule_fn=fixed_temperature_schedule(args.fixed_temperature),
        rng=np.random.default_rng(1001),
    ).run(noisy_spin, sweeps=args.sweeps, truth=clean_spin)

    annealed_result = GibbsImageRestorer(
        observation=noisy_spin,
        eta=args.eta,
        coupling=args.coupling,
        schedule_fn=geometric_schedule(args.anneal_initial_temperature, args.anneal_alpha),
        rng=np.random.default_rng(2001),
    ).run(noisy_spin, sweeps=args.sweeps, truth=clean_spin)

    plot_main_montage(grayscale, clean_spin, noisy_spin, fixed_result, annealed_result, output_dir, stem)
    plot_traces(fixed_result, annealed_result, output_dir, stem)

    rows = [
        {
            "input_image": str(args.input_image),
            "width": meta["width"],
            "height": meta["height"],
            "threshold": meta["threshold"],
            "noise": args.noise,
            "sweeps": args.sweeps,
            "noisy_accuracy": pixel_accuracy(noisy_spin, clean_spin),
            "fixed_gibbs_accuracy": fixed_result.final_accuracy,
            "annealed_accuracy": annealed_result.final_accuracy,
        }
    ]
    save_csv(output_dir / f"{stem}_summary.csv", rows)
    save_json(
        output_dir / f"{stem}_config.json",
        {
            "input_image": str(args.input_image),
            "max_dim": args.max_dim,
            "threshold": args.threshold,
            "invert": args.invert,
            "noise": args.noise,
            "sweeps": args.sweeps,
            "fixed_temperature": args.fixed_temperature,
            "anneal_initial_temperature": args.anneal_initial_temperature,
            "anneal_alpha": args.anneal_alpha,
            "eta": args.eta,
            "coupling": args.coupling,
            "derived_width": meta["width"],
            "derived_height": meta["height"],
            "derived_threshold": meta["threshold"],
        },
    )

    print("input_image\twidth\theight\tthreshold\tnoise\tsweeps\tnoisy_accuracy\tfixed_gibbs_accuracy\tannealed_accuracy")
    row = rows[0]
    print(
        f"{row['input_image']}\t{row['width']}\t{row['height']}\t{row['threshold']}\t"
        f"{row['noise']:.3f}\t{row['sweeps']}\t{row['noisy_accuracy']:.4f}\t"
        f"{row['fixed_gibbs_accuracy']:.4f}\t{row['annealed_accuracy']:.4f}"
    )
    print()
    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
