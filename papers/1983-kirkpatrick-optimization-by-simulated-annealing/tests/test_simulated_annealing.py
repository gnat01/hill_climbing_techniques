import math
import random
import sys
from pathlib import Path
import unittest


PAPER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_DIR))

from python.simulated_annealing import (  # noqa: E402
    SimulatedAnnealingOptimizer,
    acceptance_probability,
    geometric_schedule,
)


class AcceptanceTests(unittest.TestCase):
    def test_improving_move_acceptance_is_one(self) -> None:
        self.assertEqual(acceptance_probability(-1.0, 1.0), 1.0)

    def test_worsening_move_acceptance_matches_boltzmann_rule(self) -> None:
        self.assertAlmostEqual(acceptance_probability(3.0, 2.0), math.exp(-1.5))


class ScheduleTests(unittest.TestCase):
    def test_geometric_schedule_decreases_monotonically(self) -> None:
        schedule = geometric_schedule(initial_temperature=10.0, cooling_rate=0.9)
        values = [schedule(i) for i in range(5)]
        self.assertTrue(all(values[i + 1] < values[i] for i in range(len(values) - 1)))


class OptimizerTests(unittest.TestCase):
    def test_run_is_reproducible(self) -> None:
        energy = lambda x: float(x * x)
        proposal = lambda x, rng: x + rng.choice([-1, 1])
        schedule = geometric_schedule(initial_temperature=5.0, cooling_rate=0.95)

        run_a = SimulatedAnnealingOptimizer(
            energy_fn=energy,
            proposal_fn=proposal,
            schedule_fn=schedule,
            rng=random.Random(17),
        ).run(initial_state=0, steps=30)
        run_b = SimulatedAnnealingOptimizer(
            energy_fn=energy,
            proposal_fn=proposal,
            schedule_fn=schedule,
            rng=random.Random(17),
        ).run(initial_state=0, steps=30)

        self.assertEqual(run_a, run_b)

    def test_best_energy_is_monotone_nonincreasing(self) -> None:
        energy = lambda x: float((x - 2) ** 2)
        proposal = lambda x, rng: x + rng.choice([-1, 1])
        schedule = geometric_schedule(initial_temperature=4.0, cooling_rate=0.97)

        result = SimulatedAnnealingOptimizer(
            energy_fn=energy,
            proposal_fn=proposal,
            schedule_fn=schedule,
            rng=random.Random(5),
        ).run(initial_state=0, steps=40)

        best_energies = [step.best_energy for step in result.trajectory]
        self.assertTrue(all(best_energies[i + 1] <= best_energies[i] for i in range(len(best_energies) - 1)))

    def test_hotter_schedule_accepts_more_uphill_moves(self) -> None:
        energy = lambda x: float(x)
        proposal = lambda x, rng: x + 1

        cold_schedule = geometric_schedule(initial_temperature=0.5, cooling_rate=0.9)
        hot_schedule = geometric_schedule(initial_temperature=5.0, cooling_rate=0.99)

        cold_result = SimulatedAnnealingOptimizer(
            energy_fn=energy,
            proposal_fn=proposal,
            schedule_fn=cold_schedule,
            rng=random.Random(3),
        ).run(initial_state=0, steps=50)
        hot_result = SimulatedAnnealingOptimizer(
            energy_fn=energy,
            proposal_fn=proposal,
            schedule_fn=hot_schedule,
            rng=random.Random(3),
        ).run(initial_state=0, steps=50)

        self.assertLess(cold_result.uphill_moves_accepted, hot_result.uphill_moves_accepted)


if __name__ == "__main__":
    unittest.main()
