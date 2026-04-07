import unittest
from pathlib import Path
import random
import sys


PAPER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_DIR))

from python.tsp_annealing import (  # noqa: E402
    TspSimulatedAnnealing,
    build_distance_matrix,
    canonical_tour,
    exact_tsp_optimum,
    geometric_schedule,
    nearest_neighbor_tour,
    random_two_opt_proposal,
    route_length,
    small_benchmark_points,
    two_opt_move,
)


class TourUtilityTests(unittest.TestCase):
    def test_canonical_tour_is_rotation_invariant(self) -> None:
        tour_a = (3, 4, 0, 1, 2)
        tour_b = (0, 1, 2, 3, 4)
        self.assertEqual(canonical_tour(tour_a), canonical_tour(tour_b))

    def test_two_opt_move_preserves_permutation(self) -> None:
        tour = (0, 1, 2, 3, 4, 5)
        moved = two_opt_move(tour, 1, 4)
        self.assertEqual(sorted(moved), list(range(6)))


class AnnealingTests(unittest.TestCase):
    def test_run_is_reproducible(self) -> None:
        points = small_benchmark_points()
        distance_matrix = build_distance_matrix(points)
        initial_tour = nearest_neighbor_tour(distance_matrix)
        schedule = geometric_schedule(4.0, 0.995)

        result_a = TspSimulatedAnnealing(
            distance_matrix=distance_matrix,
            proposal_fn=random_two_opt_proposal,
            schedule_fn=schedule,
            rng=random.Random(9),
        ).run(initial_tour, 200)
        result_b = TspSimulatedAnnealing(
            distance_matrix=distance_matrix,
            proposal_fn=random_two_opt_proposal,
            schedule_fn=schedule,
            rng=random.Random(9),
        ).run(initial_tour, 200)

        self.assertEqual(result_a, result_b)

    def test_best_route_length_is_monotone(self) -> None:
        points = small_benchmark_points()
        distance_matrix = build_distance_matrix(points)
        initial_tour = nearest_neighbor_tour(distance_matrix)
        result = TspSimulatedAnnealing(
            distance_matrix=distance_matrix,
            proposal_fn=random_two_opt_proposal,
            schedule_fn=geometric_schedule(4.0, 0.995),
            rng=random.Random(3),
        ).run(initial_tour, 200)

        best_lengths = [step.best_route_length for step in result.trajectory]
        self.assertTrue(all(best_lengths[i + 1] <= best_lengths[i] for i in range(len(best_lengths) - 1)))

    def test_small_instance_reaches_exact_optimum(self) -> None:
        points = small_benchmark_points()
        distance_matrix = build_distance_matrix(points)
        initial_tour = nearest_neighbor_tour(distance_matrix)
        _, optimum = exact_tsp_optimum(distance_matrix)

        result = TspSimulatedAnnealing(
            distance_matrix=distance_matrix,
            proposal_fn=random_two_opt_proposal,
            schedule_fn=geometric_schedule(5.0, 0.998),
            rng=random.Random(11),
        ).run(initial_tour, 1200)

        self.assertLessEqual(result.best_route_length - optimum, 1.0e-9)

    def test_route_length_matches_recomputed_best_tour(self) -> None:
        points = small_benchmark_points()
        distance_matrix = build_distance_matrix(points)
        initial_tour = nearest_neighbor_tour(distance_matrix)
        result = TspSimulatedAnnealing(
            distance_matrix=distance_matrix,
            proposal_fn=random_two_opt_proposal,
            schedule_fn=geometric_schedule(4.0, 0.995),
            rng=random.Random(7),
        ).run(initial_tour, 100)
        self.assertAlmostEqual(result.best_route_length, route_length(result.best_tour, distance_matrix))


if __name__ == "__main__":
    unittest.main()
