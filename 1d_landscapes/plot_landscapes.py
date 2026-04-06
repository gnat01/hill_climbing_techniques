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


def generate_landscape(M: int, k: int, c: float, R: int, family: str) -> list[int]:
    return [landscape_value(x, M=M, k=k, c=c, R=R, family=family) for x in range(M)]


def save_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def slugify_float(value: float) -> str:
    return str(value).replace(".", "p")


def plot_single_landscape(M: int, k: int, c: float, R: int, family: str, output_dir: Path) -> None:
    xs = list(range(M))
    ys = generate_landscape(M, k, c, R, family)

    plt.figure(figsize=(10, 4.5))
    plt.step(xs, ys, where="mid", linewidth=1.8)
    plt.scatter(xs, ys, s=18)
    for boundary in range(k, M, k):
        plt.axvline(boundary - 0.5, color="gray", linestyle="--", alpha=0.25)
    if family != "base":
        plt.axvline(R, color="tab:red", linestyle=":", alpha=0.6, label=f"R={R}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(f"{family_title(family)}: M={M}, k={k}, c={c}, R={R}")
    plt.grid(True, alpha=0.3)
    if family != "base":
        plt.legend()
    plt.tight_layout()
    filename = f"{family}_landscape_M{M}_k{k}_c{slugify_float(c)}_R{R}.png"
    plt.savefig(output_dir / filename, dpi=160)
    plt.close()

    rows = [
        {
            "x": x,
            "f_x": y,
            "family": family,
            "formula": family_formula(family),
            "M": M,
            "k": k,
            "c": c,
            "R": R,
        }
        for x, y in zip(xs, ys)
    ]
    csv_name = f"{family}_landscape_M{M}_k{k}_c{slugify_float(c)}_R{R}.csv"
    save_csv(output_dir / csv_name, rows)


def plot_grid(M: int, ks: list[int], cs: list[float], R: int, family: str, output_dir: Path) -> None:
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
            ys = generate_landscape(M, k, c, R, family)
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

    fig.suptitle(f"{family_title(family)} grid for M={M}, R={R}", y=0.995)
    fig.tight_layout()
    fig.savefig(output_dir / f"{family}_landscape_grid_M{M}_R{R}.png", dpi=160)
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

    if M <= 1:
        raise ValueError("M must be greater than 1")
    if not 0 <= R < M:
        raise ValueError(f"R={R} must lie in [0, M-1]")
    for k in ks:
        if k <= 0 or M % k != 0:
            raise ValueError(f"k={k} must be positive and divide M={M}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for c in cs:
        for k in ks:
            plot_single_landscape(M, k, c, R, family, output_dir)

    plot_grid(M, ks, cs, R, family, output_dir)
    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
