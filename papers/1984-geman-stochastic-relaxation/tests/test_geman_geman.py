import sys
from pathlib import Path
import unittest

import numpy as np


PAPER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_DIR))

from python.geman_geman import (  # noqa: E402
    GibbsImageRestorer,
    conditional_probability_positive,
    example_square,
    fixed_temperature_schedule,
    flip_noise,
    geometric_schedule,
    local_field,
    pixel_accuracy,
)


class ProbabilityTests(unittest.TestCase):
    def test_conditional_probability_is_half_at_zero_field(self) -> None:
        self.assertAlmostEqual(conditional_probability_positive(0.0, 1.0), 0.5)

    def test_positive_field_increases_positive_probability(self) -> None:
        self.assertGreater(conditional_probability_positive(2.0, 1.0), 0.5)


class RestorationTests(unittest.TestCase):
    def test_noise_is_reproducible(self) -> None:
        image = example_square(16)
        noisy_a = flip_noise(image, 0.25, np.random.default_rng(7))
        noisy_b = flip_noise(image, 0.25, np.random.default_rng(7))
        self.assertTrue(np.array_equal(noisy_a, noisy_b))

    def test_local_field_matches_simple_case(self) -> None:
        state = np.ones((3, 3), dtype=np.int8)
        observation = np.ones((3, 3), dtype=np.int8)
        field = local_field(state, observation, 1, 1, eta=2.0, coupling=1.5)
        self.assertAlmostEqual(field, 2.0 + 1.5 * 4.0)

    def test_restoration_improves_accuracy_on_simple_example(self) -> None:
        truth = example_square(16)
        noisy = flip_noise(truth, 0.2, np.random.default_rng(3))
        before = pixel_accuracy(noisy, truth)
        restorer = GibbsImageRestorer(
            observation=noisy,
            eta=2.2,
            coupling=1.1,
            schedule_fn=geometric_schedule(2.5, 0.98),
            rng=np.random.default_rng(5),
        )
        result = restorer.run(initial_state=noisy, sweeps=30, truth=truth)
        self.assertGreaterEqual(result.final_accuracy, before)

    def test_fixed_seed_run_is_reproducible(self) -> None:
        truth = example_square(16)
        noisy = flip_noise(truth, 0.25, np.random.default_rng(10))
        restorer_a = GibbsImageRestorer(noisy, 2.0, 1.2, fixed_temperature_schedule(1.5), np.random.default_rng(11))
        restorer_b = GibbsImageRestorer(noisy, 2.0, 1.2, fixed_temperature_schedule(1.5), np.random.default_rng(11))
        result_a = restorer_a.run(noisy, sweeps=10, truth=truth)
        result_b = restorer_b.run(noisy, sweeps=10, truth=truth)
        self.assertEqual(result_a.trajectory, result_b.trajectory)
        self.assertTrue(np.array_equal(result_a.final_state, result_b.final_state))


if __name__ == "__main__":
    unittest.main()
