import unittest
import math
from pathlib import Path
import random
import sys


PAPER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_DIR))

from python.metropolis import (  # noqa: E402
    MetropolisSampler,
    acceptance_probability,
)


class AcceptanceProbabilityTests(unittest.TestCase):
    def test_improving_move_is_always_accepted(self) -> None:
        self.assertEqual(acceptance_probability(-2.0, 1.0), 1.0)
        self.assertEqual(acceptance_probability(0.0, 1.0), 1.0)

    def test_worsening_move_follows_metropolis_law(self) -> None:
        observed = acceptance_probability(2.0, 4.0)
        expected = math.exp(-0.5)
        self.assertAlmostEqual(observed, expected)

    def test_temperature_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            acceptance_probability(1.0, 0.0)


class SamplerBehaviorTests(unittest.TestCase):
    def test_run_is_reproducible_with_fixed_seed(self) -> None:
        energy = lambda x: float(x * x)
        proposal = lambda x, rng: x + rng.choice([-1, 1])

        sampler_a = MetropolisSampler(
            energy_fn=energy,
            proposal_fn=proposal,
            temperature=1.5,
            rng=random.Random(7),
        )
        sampler_b = MetropolisSampler(
            energy_fn=energy,
            proposal_fn=proposal,
            temperature=1.5,
            rng=random.Random(7),
        )

        result_a = sampler_a.run(initial_state=0, steps=25)
        result_b = sampler_b.run(initial_state=0, steps=25)
        self.assertEqual(result_a, result_b)

    def test_low_temperature_biases_toward_low_energy_state(self) -> None:
        states = [0, 1]
        energy_map = {0: 0.0, 1: 2.0}

        def energy(state: int) -> float:
            return energy_map[state]

        def proposal(state: int, rng: random.Random) -> int:
            return states[1 - state]

        cold_sampler = MetropolisSampler(
            energy_fn=energy,
            proposal_fn=proposal,
            temperature=0.25,
            rng=random.Random(11),
        )
        result = cold_sampler.run(initial_state=0, steps=400)
        visits_to_high_energy = sum(1 for step in result.trajectory if step.state == 1)
        self.assertLess(visits_to_high_energy, 10)

    def test_high_temperature_accepts_more_worsening_moves(self) -> None:
        energy = lambda x: float(x)
        proposal = lambda x, rng: x + 1

        cold_sampler = MetropolisSampler(
            energy_fn=energy,
            proposal_fn=proposal,
            temperature=0.5,
            rng=random.Random(3),
        )
        hot_sampler = MetropolisSampler(
            energy_fn=energy,
            proposal_fn=proposal,
            temperature=5.0,
            rng=random.Random(3),
        )

        cold_result = cold_sampler.run(initial_state=0, steps=50)
        hot_result = hot_sampler.run(initial_state=0, steps=50)
        self.assertLess(cold_result.accepted_moves, hot_result.accepted_moves)


if __name__ == "__main__":
    unittest.main()
