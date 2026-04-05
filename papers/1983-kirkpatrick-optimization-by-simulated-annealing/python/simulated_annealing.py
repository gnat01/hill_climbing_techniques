"""Simulated annealing optimizer for minimization problems."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable, Generic, List, Optional, Sequence, TypeVar


State = TypeVar("State")


def acceptance_probability(delta_energy: float, temperature: float) -> float:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if delta_energy <= 0.0:
        return 1.0
    return math.exp(-delta_energy / temperature)


def geometric_schedule(initial_temperature: float, cooling_rate: float) -> Callable[[int], float]:
    if initial_temperature <= 0.0:
        raise ValueError("initial_temperature must be positive")
    if not 0.0 < cooling_rate < 1.0:
        raise ValueError("cooling_rate must lie in (0, 1)")

    def schedule(step_index: int) -> float:
        return max(initial_temperature * (cooling_rate ** step_index), 1.0e-12)

    return schedule


def linear_schedule(
    initial_temperature: float,
    final_temperature: float,
    total_steps: int,
) -> Callable[[int], float]:
    if initial_temperature <= 0.0 or final_temperature <= 0.0:
        raise ValueError("temperatures must be positive")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if final_temperature > initial_temperature:
        raise ValueError("final_temperature must not exceed initial_temperature")

    def schedule(step_index: int) -> float:
        fraction = min(max(step_index / total_steps, 0.0), 1.0)
        return initial_temperature + fraction * (final_temperature - initial_temperature)

    return schedule


def logarithmic_schedule(initial_temperature: float) -> Callable[[int], float]:
    if initial_temperature <= 0.0:
        raise ValueError("initial_temperature must be positive")

    def schedule(step_index: int) -> float:
        return initial_temperature / math.log(step_index + math.e)

    return schedule


@dataclass(frozen=True)
class SimulatedAnnealingStep(Generic[State]):
    step_index: int
    temperature: float
    state: State
    energy: float
    best_state: State
    best_energy: float
    proposed_state: State
    proposed_energy: float
    delta_energy: float
    accepted: bool
    acceptance_probability: float
    uphill_move_accepted: bool


@dataclass(frozen=True)
class SimulatedAnnealingResult(Generic[State]):
    initial_state: State
    final_state: State
    final_energy: float
    best_state: State
    best_energy: float
    accepted_moves: int
    uphill_moves_accepted: int
    total_steps: int
    acceptance_rate: float
    trajectory: Sequence[SimulatedAnnealingStep[State]]


class SimulatedAnnealingOptimizer(Generic[State]):
    def __init__(
        self,
        energy_fn: Callable[[State], float],
        proposal_fn: Callable[[State, random.Random], State],
        schedule_fn: Callable[[int], float],
        rng: Optional[random.Random] = None,
    ) -> None:
        self.energy_fn = energy_fn
        self.proposal_fn = proposal_fn
        self.schedule_fn = schedule_fn
        self.rng = rng if rng is not None else random.Random()

    def run(self, initial_state: State, steps: int) -> SimulatedAnnealingResult[State]:
        if steps < 0:
            raise ValueError("steps must be non-negative")

        state = initial_state
        energy = self.energy_fn(state)
        best_state = state
        best_energy = energy
        accepted_moves = 0
        uphill_moves_accepted = 0
        trajectory: List[SimulatedAnnealingStep[State]] = []

        for step_index in range(steps):
            temperature = self.schedule_fn(step_index)
            if temperature <= 0.0:
                raise ValueError("schedule produced a non-positive temperature")

            proposed_state = self.proposal_fn(state, self.rng)
            proposed_energy = self.energy_fn(proposed_state)
            delta_energy = proposed_energy - energy
            accept_prob = acceptance_probability(delta_energy, temperature)
            accepted = self.rng.random() < accept_prob

            if accepted:
                state = proposed_state
                energy = proposed_energy
                accepted_moves += 1
                if delta_energy > 0.0:
                    uphill_moves_accepted += 1

            if energy < best_energy:
                best_energy = energy
                best_state = state

            trajectory.append(
                SimulatedAnnealingStep(
                    step_index=step_index,
                    temperature=temperature,
                    state=state,
                    energy=energy,
                    best_state=best_state,
                    best_energy=best_energy,
                    proposed_state=proposed_state,
                    proposed_energy=proposed_energy,
                    delta_energy=delta_energy,
                    accepted=accepted,
                    acceptance_probability=accept_prob,
                    uphill_move_accepted=accepted and delta_energy > 0.0,
                )
            )

        return SimulatedAnnealingResult(
            initial_state=initial_state,
            final_state=state,
            final_energy=energy,
            best_state=best_state,
            best_energy=best_energy,
            accepted_moves=accepted_moves,
            uphill_moves_accepted=uphill_moves_accepted,
            total_steps=steps,
            acceptance_rate=accepted_moves / steps if steps else 0.0,
            trajectory=tuple(trajectory),
        )


def greedy_hill_climb(
    initial_state: State,
    energy_fn: Callable[[State], float],
    proposal_fn: Callable[[State, random.Random], State],
    steps: int,
    rng: Optional[random.Random] = None,
) -> SimulatedAnnealingResult[State]:
    local_rng = rng if rng is not None else random.Random()
    always_cold = lambda _: 1.0e-12
    optimizer = SimulatedAnnealingOptimizer(
        energy_fn=energy_fn,
        proposal_fn=proposal_fn,
        schedule_fn=always_cold,
        rng=local_rng,
    )
    return optimizer.run(initial_state=initial_state, steps=steps)


def rugged_landscape_energy(x: float) -> float:
    """One-dimensional rugged landscape with multiple local minima."""
    return 0.16 * ((x + 1.7) ** 2) * ((x - 1.0) ** 2) + 0.35 * x + 0.05 * math.sin(8.0 * x)


def random_walk_proposal(step_size: float) -> Callable[[float, random.Random], float]:
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")

    def propose(state: float, rng: random.Random) -> float:
        proposed = state + rng.uniform(-step_size, step_size)
        return min(3.0, max(-3.0, proposed))

    return propose
