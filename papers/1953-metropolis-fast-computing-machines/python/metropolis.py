"""Metropolis sampler for symmetric proposal kernels.

This implementation is intentionally small and reusable. It captures the 1953
acceptance rule directly and records enough diagnostics to support later
simulated annealing and optimization-oriented extensions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable, Generic, List, Optional, Sequence, TypeVar


State = TypeVar("State")


@dataclass(frozen=True)
class MetropolisStep(Generic[State]):
    step_index: int
    state: State
    energy: float
    accepted: bool
    proposed_state: State
    proposed_energy: float
    delta_energy: float
    acceptance_probability: float


@dataclass(frozen=True)
class MetropolisRunResult(Generic[State]):
    initial_state: State
    final_state: State
    final_energy: float
    temperature: float
    accepted_moves: int
    total_steps: int
    acceptance_rate: float
    trajectory: Sequence[MetropolisStep[State]]


def acceptance_probability(delta_energy: float, temperature: float) -> float:
    """Return the Metropolis acceptance probability for a symmetric proposal."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if delta_energy <= 0.0:
        return 1.0
    return math.exp(-delta_energy / temperature)


class MetropolisSampler(Generic[State]):
    """Metropolis sampler with explicit proposal and energy functions."""

    def __init__(
        self,
        energy_fn: Callable[[State], float],
        proposal_fn: Callable[[State, random.Random], State],
        temperature: float,
        rng: Optional[random.Random] = None,
    ) -> None:
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.energy_fn = energy_fn
        self.proposal_fn = proposal_fn
        self.temperature = temperature
        self.rng = rng if rng is not None else random.Random()

    def step(self, state: State) -> MetropolisStep[State]:
        current_energy = self.energy_fn(state)
        proposed_state = self.proposal_fn(state, self.rng)
        proposed_energy = self.energy_fn(proposed_state)
        delta_energy = proposed_energy - current_energy
        accept_prob = acceptance_probability(delta_energy, self.temperature)
        accepted = self.rng.random() < accept_prob
        next_state = proposed_state if accepted else state
        next_energy = proposed_energy if accepted else current_energy
        return MetropolisStep(
            step_index=-1,
            state=next_state,
            energy=next_energy,
            accepted=accepted,
            proposed_state=proposed_state,
            proposed_energy=proposed_energy,
            delta_energy=delta_energy,
            acceptance_probability=accept_prob,
        )

    def run(self, initial_state: State, steps: int) -> MetropolisRunResult[State]:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        trajectory: List[MetropolisStep[State]] = []
        state = initial_state
        accepted_moves = 0

        for step_index in range(steps):
            raw_step = self.step(state)
            step = MetropolisStep(
                step_index=step_index,
                state=raw_step.state,
                energy=raw_step.energy,
                accepted=raw_step.accepted,
                proposed_state=raw_step.proposed_state,
                proposed_energy=raw_step.proposed_energy,
                delta_energy=raw_step.delta_energy,
                acceptance_probability=raw_step.acceptance_probability,
            )
            trajectory.append(step)
            state = step.state
            if step.accepted:
                accepted_moves += 1

        final_energy = self.energy_fn(state)
        acceptance_rate = accepted_moves / steps if steps else 0.0
        return MetropolisRunResult(
            initial_state=initial_state,
            final_state=state,
            final_energy=final_energy,
            temperature=self.temperature,
            accepted_moves=accepted_moves,
            total_steps=steps,
            acceptance_rate=acceptance_rate,
            trajectory=tuple(trajectory),
        )


def random_walk_proposal(step_size: float) -> Callable[[float, random.Random], float]:
    """Return a symmetric random-walk proposal on the real line."""
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")

    def propose(state: float, rng: random.Random) -> float:
        return state + rng.uniform(-step_size, step_size)

    return propose


def double_well_energy(x: float) -> float:
    """Simple quartic double-well landscape with minima near x = +/-1."""
    return (x * x - 1.0) ** 2
