"""Simulated annealing for Euclidean TSP."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import random
from typing import Callable, List, Sequence


Point = tuple[float, float]
Tour = tuple[int, ...]


def euclidean_distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_distance_matrix(points: Sequence[Point]) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(euclidean_distance(points[i], points[j]) for j in range(len(points)))
        for i in range(len(points))
    )


def canonical_tour(tour: Sequence[int]) -> Tour:
    if not tour:
        raise ValueError("tour must be non-empty")
    n = len(tour)
    start = min(range(n), key=lambda i: tour[i])
    rotated = tuple(tour[(start + i) % n] for i in range(n))
    reversed_rotated = tuple(rotated[0:1] + rotated[:0:-1])
    return min(rotated, reversed_rotated)


def route_length(tour: Sequence[int], distance_matrix: Sequence[Sequence[float]]) -> float:
    total = 0.0
    n = len(tour)
    for i in range(n):
        total += distance_matrix[tour[i]][tour[(i + 1) % n]]
    return total


def swap_move(tour: Sequence[int], i: int, j: int) -> Tour:
    if i == j:
        return tuple(tour)
    new_tour = list(tour)
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return tuple(new_tour)


def insert_move(tour: Sequence[int], i: int, j: int) -> Tour:
    if i == j:
        return tuple(tour)
    new_tour = list(tour)
    city = new_tour.pop(i)
    new_tour.insert(j, city)
    return tuple(new_tour)


def two_opt_move(tour: Sequence[int], i: int, j: int) -> Tour:
    if i >= j:
        raise ValueError("require i < j")
    return tuple(tour[:i] + tuple(reversed(tour[i : j + 1])) + tour[j + 1 :])


def random_swap_proposal(tour: Tour, rng: random.Random) -> Tour:
    i, j = sorted(rng.sample(range(len(tour)), 2))
    return swap_move(tour, i, j)


def random_insert_proposal(tour: Tour, rng: random.Random) -> Tour:
    i, j = rng.sample(range(len(tour)), 2)
    return insert_move(tour, i, j)


def random_two_opt_proposal(tour: Tour, rng: random.Random) -> Tour:
    i, j = sorted(rng.sample(range(len(tour)), 2))
    if i == 0 and j == len(tour) - 1:
        i = 1
    return two_opt_move(tour, i, j)


def acceptance_probability(delta: float, temperature: float) -> float:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if delta <= 0.0:
        return 1.0
    return math.exp(-delta / temperature)


def geometric_schedule(initial_temperature: float, alpha: float) -> Callable[[int], float]:
    if initial_temperature <= 0.0:
        raise ValueError("initial_temperature must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")

    def schedule(step_index: int) -> float:
        return max(initial_temperature * (alpha ** step_index), 1.0e-12)

    return schedule


@dataclass(frozen=True)
class TspAnnealingStep:
    step_index: int
    temperature: float
    tour: Tour
    route_length: float
    best_tour: Tour
    best_route_length: float
    accepted: bool
    proposed_route_length: float
    delta: float
    uphill_move_accepted: bool


@dataclass(frozen=True)
class TspAnnealingResult:
    initial_tour: Tour
    final_tour: Tour
    final_route_length: float
    best_tour: Tour
    best_route_length: float
    acceptance_rate: float
    accepted_moves: int
    uphill_moves_accepted: int
    trajectory: Sequence[TspAnnealingStep]


class TspSimulatedAnnealing:
    def __init__(
        self,
        distance_matrix: Sequence[Sequence[float]],
        proposal_fn: Callable[[Tour, random.Random], Tour],
        schedule_fn: Callable[[int], float],
        rng: random.Random | None = None,
    ) -> None:
        self.distance_matrix = distance_matrix
        self.proposal_fn = proposal_fn
        self.schedule_fn = schedule_fn
        self.rng = rng if rng is not None else random.Random()

    def run(self, initial_tour: Tour, steps: int) -> TspAnnealingResult:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        tour = tuple(initial_tour)
        current_length = route_length(tour, self.distance_matrix)
        best_tour = tour
        best_length = current_length
        accepted_moves = 0
        uphill_moves_accepted = 0
        trajectory: List[TspAnnealingStep] = []

        for step_index in range(steps):
            temperature = self.schedule_fn(step_index)
            proposed_tour = self.proposal_fn(tour, self.rng)
            proposed_length = route_length(proposed_tour, self.distance_matrix)
            delta = proposed_length - current_length
            accept_prob = acceptance_probability(delta, temperature)
            accepted = self.rng.random() < accept_prob

            if accepted:
                tour = proposed_tour
                current_length = proposed_length
                accepted_moves += 1
                if delta > 0.0:
                    uphill_moves_accepted += 1

            if current_length < best_length:
                best_length = current_length
                best_tour = tour

            trajectory.append(
                TspAnnealingStep(
                    step_index=step_index,
                    temperature=temperature,
                    tour=tour,
                    route_length=current_length,
                    best_tour=best_tour,
                    best_route_length=best_length,
                    accepted=accepted,
                    proposed_route_length=proposed_length,
                    delta=delta,
                    uphill_move_accepted=accepted and delta > 0.0,
                )
            )

        return TspAnnealingResult(
            initial_tour=tuple(initial_tour),
            final_tour=tour,
            final_route_length=current_length,
            best_tour=best_tour,
            best_route_length=best_length,
            acceptance_rate=accepted_moves / steps if steps else 0.0,
            accepted_moves=accepted_moves,
            uphill_moves_accepted=uphill_moves_accepted,
            trajectory=tuple(trajectory),
        )


def nearest_neighbor_tour(distance_matrix: Sequence[Sequence[float]], start: int = 0) -> Tour:
    n = len(distance_matrix)
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    current = start
    while unvisited:
        current = min(unvisited, key=lambda city: distance_matrix[current][city])
        tour.append(current)
        unvisited.remove(current)
    return tuple(tour)


def exact_tsp_optimum(distance_matrix: Sequence[Sequence[float]]) -> tuple[Tour, float]:
    n = len(distance_matrix)
    if n > 10:
        raise ValueError("exact search is restricted to n <= 10")
    cities = tuple(range(n))
    best_tour = cities
    best_length = float("inf")
    for perm in itertools.permutations(cities[1:]):
        tour = (0,) + perm
        length = route_length(tour, distance_matrix)
        if length < best_length:
            best_length = length
            best_tour = tour
    return canonical_tour(best_tour), best_length


def small_benchmark_points() -> tuple[Point, ...]:
    return (
        (0.1, 0.2),
        (1.0, 0.1),
        (2.0, 0.4),
        (2.7, 1.5),
        (2.0, 2.7),
        (0.9, 2.9),
        (-0.1, 2.0),
        (-0.4, 1.0),
    )


def medium_benchmark_points() -> tuple[Point, ...]:
    return (
        (0.1, 0.2),
        (0.8, -0.1),
        (1.8, 0.1),
        (2.7, 0.4),
        (3.1, 1.1),
        (3.0, 2.0),
        (2.3, 2.8),
        (1.4, 3.1),
        (0.4, 2.8),
        (-0.2, 2.0),
        (-0.4, 1.1),
        (1.2, 1.5),
    )
