"""Plot discrete 1D landscape families for discussion.

Supported families:

1. base:
       f(x) = floor(2x/k) + floor(c * x * (M - 1 - x) / M)

2. f3:
       y = x - R
       f_3(x) = floor(2 y^2 / k) - floor(c y^4 / M^2)

3. f5:
       y = x - R
       f_5(x) = floor(2 y^2 / k) + floor(c ((y mod k)^2))

with:
- x in {0, 1, ..., M-1}
- k dividing M
- c controlling perturbation scale
- R controlling the landscape shift
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from pathlib import Path

MPL_DIR = Path(__file__).resolve().parent / ".mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parent / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def landscape_value_base(x: int, M: int, k: int, c: float, R: int) -> int:
    return math.floor(2 * x / k) + math.floor(c * x * (M - 1 - x) / M)


def landscape_value_f3(x: int, M: int, k: int, c: float, R: int) -> int:
    y = x - R
    return math.floor(2 * (y**2) / k) - math.floor(c * (y**4) / (M**2))


def landscape_value_f5(x: int, M: int, k: int, c: float, R: int) -> int:
    y = x - R
    return math.floor(2 * (y**2) / k) + math.floor(c * ((y % k) ** 2))


def family_title(family: str) -> str:
    if family == "base":
        return "Base staircase-concave family"
    if family == "f3":
        return "Shifted quartic-competing family f3"
    if family == "f5":
        return "Shifted block-rugged family f5"
    raise ValueError(f"Unsupported family: {family}")


def family_formula(family: str) -> str:
    if family == "base":
        return "floor(2x/k) + floor(c x (M-1-x) / M)"
    if family == "f3":
        return "floor(2(x-R)^2/k) - floor(c (x-R)^4 / M^2)"
    if family == "f5":
        return "floor(2(x-R)^2/k) + floor(c (((x-R) mod k)^2))"
    raise ValueError(f"Unsupported family: {family}")


def landscape_value(x: int, M: int, k: int, c: float, R: int, family: str) -> int:
    if family == "base":
        return landscape_value_base(x, M=M, k=k, c=c, R=R)
    if family == "f3":
        return landscape_value_f3(x, M=M, k=k, c=c, R=R)
    if family == "f5":
        return landscape_value_f5(x, M=M, k=k, c=c, R=R)
    raise ValueError(f"Unsupported family: {family}")


def generate_landscape(
    M: int,
    k: int,
    c: float,
    R: int,
    family: str,
    gaussian_noise: bool,
    noise_mean: float,
    noise_sd: float,
) -> tuple[list[int], list[float]]:
    base_values = [landscape_value(x, M=M, k=k, c=c, R=R, family=family) for x in range(M)]
    if not gaussian_noise:
        return base_values, [0.0] * M

    noise_values = [random.gauss(noise_mean, noise_sd) for _ in range(M)]
    noisy_values = [value + noise for value, noise in zip(base_values, noise_values)]
    return noisy_values, noise_values


def save_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def slugify_float(value: float) -> str:
    return str(value).replace(".", "p")


def plot_single_landscape(
    M: int,
    k: int,
    c: float,
    R: int,
    family: str,
    output_dir: Path,
    gaussian_noise: bool,
    noise_mean: float,
    noise_sd: float,
) -> None:
    xs = list(range(M))
    ys, noise_values = generate_landscape(
        M,
        k,
        c,
        R,
        family,
        gaussian_noise=gaussian_noise,
        noise_mean=noise_mean,
        noise_sd=noise_sd,
    )

    plt.figure(figsize=(10, 4.5))
    plt.step(xs, ys, where="mid", linewidth=1.8)
    plt.scatter(xs, ys, s=18)
    for boundary in range(k, M, k):
        plt.axvline(boundary - 0.5, color="gray", linestyle="--", alpha=0.25)
    if family != "base":
        plt.axvline(R, color="tab:red", linestyle=":", alpha=0.6, label=f"R={R}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    title = f"{family_title(family)}: M={M}, k={k}, c={c}, R={R}"
    if gaussian_noise:
        title += f", noise=N({noise_mean}, {noise_sd})"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if family != "base":
        plt.legend()
    plt.tight_layout()
    noise_suffix = ""
    if gaussian_noise:
        noise_suffix = f"_noise_mu{slugify_float(noise_mean)}_sd{slugify_float(noise_sd)}"
    filename = f"{family}_landscape_M{M}_k{k}_c{slugify_float(c)}_R{R}{noise_suffix}.png"
    plt.savefig(output_dir / filename, dpi=160)
    plt.close()

    rows = [
        {
            "x": x,
            "f_x": y,
            "base_f_x": y - noise,
            "noise": noise,
            "family": family,
            "formula": family_formula(family),
            "M": M,
            "k": k,
            "c": c,
            "R": R,
            "gaussian_noise": gaussian_noise,
            "noise_mean": noise_mean if gaussian_noise else 0.0,
            "noise_sd": noise_sd if gaussian_noise else 0.0,
        }
        for x, y, noise in zip(xs, ys, noise_values)
    ]
    csv_name = f"{family}_landscape_M{M}_k{k}_c{slugify_float(c)}_R{R}{noise_suffix}.csv"
    save_csv(output_dir / csv_name, rows)


def plot_grid(
    M: int,
    ks: list[int],
    cs: list[float],
    R: int,
    family: str,
    output_dir: Path,
    gaussian_noise: bool,
    noise_mean: float,
    noise_sd: float,
) -> None:
    fig, axes = plt.subplots(len(cs), len(ks), figsize=(4.2 * len(ks), 2.8 * len(cs)), sharex=True)
    if len(cs) == 1 and len(ks) == 1:
        axes = [[axes]]
    elif len(cs) == 1:
        axes = [axes]
    elif len(ks) == 1:
        axes = [[ax] for ax in axes]

    for row_index, c in enumerate(cs):
        for col_index, k in enumerate(ks):
            ax = axes[row_index][col_index]
            xs = list(range(M))
            ys, _ = generate_landscape(
                M,
                k,
                c,
                R,
                family,
                gaussian_noise=gaussian_noise,
                noise_mean=noise_mean,
                noise_sd=noise_sd,
            )
            ax.step(xs, ys, where="mid", linewidth=1.4)
            ax.set_title(f"k={k}, c={c}")
            for boundary in range(k, M, k):
                ax.axvline(boundary - 0.5, color="gray", linestyle="--", alpha=0.2)
            if family != "base":
                ax.axvline(R, color="tab:red", linestyle=":", alpha=0.5)
            ax.grid(True, alpha=0.2)
            if row_index == len(cs) - 1:
                ax.set_xlabel("x")
            if col_index == 0:
                ax.set_ylabel("f(x)")

    title = f"{family_title(family)} grid for M={M}, R={R}"
    if gaussian_noise:
        title += f", noise=N({noise_mean}, {noise_sd})"
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    noise_suffix = ""
    if gaussian_noise:
        noise_suffix = f"_noise_mu{slugify_float(noise_mean)}_sd{slugify_float(noise_sd)}"
    fig.savefig(output_dir / f"{family}_landscape_grid_M{M}_R{R}{noise_suffix}.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the proposed 1D landscape families.")
    parser.add_argument("--M", type=int, default=120, help="Domain size. x runs from 0 to M-1.")
    parser.add_argument(
        "--family",
        type=str,
        default="base",
        choices=["base", "f3", "f5"],
        help="Which landscape family to plot.",
    )
    parser.add_argument(
        "--ks",
        type=str,
        default="10,20,30",
        help="Comma-separated k values. Each must divide M.",
    )
    parser.add_argument(
        "--cs",
        type=str,
        default="0.1,0.2,0.4,0.8",
        help="Comma-separated c values.",
    )
    parser.add_argument(
        "--R",
        type=int,
        default=60,
        help="Shift parameter for the shifted families.",
    )
    parser.add_argument(
        "--gaussian-noise",
        action="store_true",
        help="Add independent Gaussian noise at each x value. If set, --noise-mean and --noise-sd are required.",
    )
    parser.add_argument(
        "--noise-mean",
        type=float,
        default=None,
        help="Mean of the Gaussian noise added pointwise when --gaussian-noise is enabled.",
    )
    parser.add_argument(
        "--noise-sd",
        type=float,
        default=None,
        help="Standard deviation of the Gaussian noise added pointwise when --gaussian-noise is enabled.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed used when Gaussian noise is enabled.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for PNG and CSV outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    M = args.M
    family = args.family
    ks = [int(item.strip()) for item in args.ks.split(",") if item.strip()]
    cs = [float(item.strip()) for item in args.cs.split(",") if item.strip()]
    R = args.R
    gaussian_noise = args.gaussian_noise
    noise_mean = args.noise_mean
    noise_sd = args.noise_sd

    if M <= 1:
        raise ValueError("M must be greater than 1")
    if not 0 <= R < M:
        raise ValueError(f"R={R} must lie in [0, M-1]")
    for k in ks:
        if k <= 0 or M % k != 0:
            raise ValueError(f"k={k} must be positive and divide M={M}")
    if gaussian_noise and family != "f5":
        raise ValueError("--gaussian-noise is currently supported only for --family f5")
    if gaussian_noise and (noise_mean is None or noise_sd is None):
        raise ValueError("--noise-mean and --noise-sd are required when --gaussian-noise is set")
    if not gaussian_noise:
        noise_mean = 0.0
        noise_sd = 0.0
    if noise_sd < 0:
        raise ValueError("--noise-sd must be non-negative")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    for c in cs:
        for k in ks:
            plot_single_landscape(
                M,
                k,
                c,
                R,
                family,
                output_dir,
                gaussian_noise=gaussian_noise,
                noise_mean=noise_mean,
                noise_sd=noise_sd,
            )

    plot_grid(
        M,
        ks,
        cs,
        R,
        family,
        output_dir,
        gaussian_noise=gaussian_noise,
        noise_mean=noise_mean,
        noise_sd=noise_sd,
    )
    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
