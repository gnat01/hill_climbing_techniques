"""Binary image denoising via Gibbs restoration and annealed stochastic relaxation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable, Sequence

import numpy as np


Array = np.ndarray


def to_spin(image: Array) -> Array:
    """Convert a 0/1 image to -1/+1 spin representation."""
    return np.where(image > 0, 1, -1).astype(np.int8)


def from_spin(image: Array) -> Array:
    """Convert -1/+1 image to 0/1 representation."""
    return ((image > 0).astype(np.uint8)).copy()


def otsu_threshold_from_grayscale(grayscale: Array) -> int:
    """Compute an Otsu threshold for a uint8 grayscale image."""
    hist = np.bincount(grayscale.ravel(), minlength=256).astype(np.float64)
    total = grayscale.size
    sum_total = float(np.dot(np.arange(256), hist))
    sum_background = 0.0
    weight_background = 0.0
    best_threshold = 127
    best_between = -1.0

    for threshold in range(256):
        weight_background += hist[threshold]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += threshold * hist[threshold]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if between > best_between:
            best_between = between
            best_threshold = threshold
    return int(best_threshold)


def flip_noise(image: Array, flip_probability: float, rng: np.random.Generator) -> Array:
    if not 0.0 <= flip_probability <= 1.0:
        raise ValueError("flip_probability must lie in [0, 1]")
    mask = rng.random(image.shape) < flip_probability
    noisy = image.copy()
    noisy[mask] *= -1
    return noisy


def geometric_schedule(initial_temperature: float, alpha: float) -> Callable[[int], float]:
    if initial_temperature <= 0.0:
        raise ValueError("initial_temperature must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")

    def schedule(step_index: int) -> float:
        return max(initial_temperature * (alpha ** step_index), 1.0e-12)

    return schedule


def local_field(state: Array, observation: Array, row: int, col: int, eta: float, coupling: float) -> float:
    height, width = state.shape
    neighbor_sum = 0
    if row > 0:
        neighbor_sum += int(state[row - 1, col])
    if row + 1 < height:
        neighbor_sum += int(state[row + 1, col])
    if col > 0:
        neighbor_sum += int(state[row, col - 1])
    if col + 1 < width:
        neighbor_sum += int(state[row, col + 1])
    return eta * int(observation[row, col]) + coupling * neighbor_sum


def conditional_probability_positive(field: float, temperature: float) -> float:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    z = 2.0 * field / temperature
    if z >= 0.0:
        exp_neg_z = math.exp(-z)
        return 1.0 / (1.0 + exp_neg_z)
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def posterior_energy(state: Array, observation: Array, eta: float, coupling: float) -> float:
    fidelity = -eta * float(np.sum(state * observation))
    smoothness = 0.0
    smoothness -= coupling * float(np.sum(state[1:, :] * state[:-1, :]))
    smoothness -= coupling * float(np.sum(state[:, 1:] * state[:, :-1]))
    return fidelity + smoothness


def pixel_accuracy(state: Array, truth: Array) -> float:
    return float(np.mean(state == truth))


@dataclass(frozen=True)
class RestorationStep:
    step_index: int
    temperature: float
    energy: float
    pixel_accuracy: float
    changed_pixels: int


@dataclass(frozen=True)
class RestorationResult:
    initial_state: Array
    final_state: Array
    final_energy: float
    final_accuracy: float
    trajectory: Sequence[RestorationStep]


class GibbsImageRestorer:
    def __init__(
        self,
        observation: Array,
        eta: float,
        coupling: float,
        schedule_fn: Callable[[int], float],
        rng: np.random.Generator | None = None,
    ) -> None:
        self.observation = observation.astype(np.int8)
        self.eta = eta
        self.coupling = coupling
        self.schedule_fn = schedule_fn
        self.rng = rng if rng is not None else np.random.default_rng()

    def run(self, initial_state: Array, sweeps: int, truth: Array | None = None) -> RestorationResult:
        if sweeps < 0:
            raise ValueError("sweeps must be non-negative")
        state = initial_state.astype(np.int8).copy()
        trajectory: list[RestorationStep] = []
        height, width = state.shape

        for sweep_index in range(sweeps):
            temperature = self.schedule_fn(sweep_index)
            changed_pixels = 0
            flat_indices = self.rng.permutation(height * width)
            for flat_idx in flat_indices:
                row = int(flat_idx // width)
                col = int(flat_idx % width)
                field = local_field(state, self.observation, row, col, self.eta, self.coupling)
                p_positive = conditional_probability_positive(field, temperature)
                new_value = 1 if self.rng.random() < p_positive else -1
                if new_value != int(state[row, col]):
                    changed_pixels += 1
                state[row, col] = new_value

            accuracy = pixel_accuracy(state, truth) if truth is not None else float("nan")
            trajectory.append(
                RestorationStep(
                    step_index=sweep_index,
                    temperature=temperature,
                    energy=posterior_energy(state, self.observation, self.eta, self.coupling),
                    pixel_accuracy=accuracy,
                    changed_pixels=changed_pixels,
                )
            )

        final_accuracy = pixel_accuracy(state, truth) if truth is not None else float("nan")
        return RestorationResult(
            initial_state=initial_state.copy(),
            final_state=state.copy(),
            final_energy=posterior_energy(state, self.observation, self.eta, self.coupling),
            final_accuracy=final_accuracy,
            trajectory=tuple(trajectory),
        )


class DeterministicImageRestorer:
    """Iterated conditional modes style deterministic local descent."""

    def __init__(
        self,
        observation: Array,
        eta: float,
        coupling: float,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.observation = observation.astype(np.int8)
        self.eta = eta
        self.coupling = coupling
        self.rng = rng if rng is not None else np.random.default_rng()

    def run(self, initial_state: Array, sweeps: int, truth: Array | None = None) -> RestorationResult:
        if sweeps < 0:
            raise ValueError("sweeps must be non-negative")
        state = initial_state.astype(np.int8).copy()
        trajectory: list[RestorationStep] = []
        height, width = state.shape

        for sweep_index in range(sweeps):
            changed_pixels = 0
            flat_indices = self.rng.permutation(height * width)
            for flat_idx in flat_indices:
                row = int(flat_idx // width)
                col = int(flat_idx % width)
                field = local_field(state, self.observation, row, col, self.eta, self.coupling)
                if field > 0:
                    new_value = 1
                elif field < 0:
                    new_value = -1
                else:
                    new_value = int(state[row, col])
                if new_value != int(state[row, col]):
                    changed_pixels += 1
                state[row, col] = new_value

            accuracy = pixel_accuracy(state, truth) if truth is not None else float("nan")
            trajectory.append(
                RestorationStep(
                    step_index=sweep_index,
                    temperature=0.0,
                    energy=posterior_energy(state, self.observation, self.eta, self.coupling),
                    pixel_accuracy=accuracy,
                    changed_pixels=changed_pixels,
                )
            )

        final_accuracy = pixel_accuracy(state, truth) if truth is not None else float("nan")
        return RestorationResult(
            initial_state=initial_state.copy(),
            final_state=state.copy(),
            final_energy=posterior_energy(state, self.observation, self.eta, self.coupling),
            final_accuracy=final_accuracy,
            trajectory=tuple(trajectory),
        )


def fixed_temperature_schedule(temperature: float) -> Callable[[int], float]:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    return lambda _: temperature


def example_square(size: int = 32) -> Array:
    image = np.zeros((size, size), dtype=np.uint8)
    image[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 1
    return to_spin(image)


def example_x(size: int = 32) -> Array:
    image = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        image[i, i] = 1
        image[i, size - 1 - i] = 1
    return to_spin(image)


def example_frame(size: int = 32) -> Array:
    image = np.zeros((size, size), dtype=np.uint8)
    thickness = 3
    margin = 5
    image[margin : size - margin, margin : margin + thickness] = 1
    image[margin : size - margin, size - margin - thickness : size - margin] = 1
    image[margin : margin + thickness, margin : size - margin] = 1
    image[size - margin - thickness : size - margin, margin : size - margin] = 1
    return to_spin(image)


def example_stripes(size: int = 32) -> Array:
    image = np.zeros((size, size), dtype=np.uint8)
    image[:, ::4] = 1
    image[:, 1::4] = 1
    return to_spin(image)


def example_plus(size: int = 32) -> Array:
    image = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    half_width = 3
    image[center - half_width : center + half_width + 1, size // 4 : 3 * size // 4] = 1
    image[size // 4 : 3 * size // 4, center - half_width : center + half_width + 1] = 1
    return to_spin(image)


def example_disk(size: int = 32) -> Array:
    y, x = np.ogrid[:size, :size]
    center = size / 2 - 0.5
    radius = size / 4
    image = (((x - center) ** 2 + (y - center) ** 2) <= radius * radius).astype(np.uint8)
    return to_spin(image)


def all_examples(size: int = 32) -> dict[str, Array]:
    return {
        "square": example_square(size),
        "x_shape": example_x(size),
        "frame": example_frame(size),
        "plus": example_plus(size),
        "disk": example_disk(size),
    }
