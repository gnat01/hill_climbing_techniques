# Implementation Plan

## Paper Objective

Implement simulated annealing for the symmetric Euclidean traveling salesman problem, emphasizing permutation-state search, local tour moves, and efficient route-length updates.

## Algorithmic Core

Let a tour be a permutation $\sigma$ of cities with cyclic objective

$$
f(\sigma) = \sum_{i=1}^{n} d(\sigma_i, \sigma_{i+1}).
$$

At each step, propose a local perturbation of the tour, compute the route-length change

$$
\Delta = f(\sigma') - f(\sigma),
$$

and accept according to the Metropolis rule at the current temperature.

## State Representation

- a tour as a tuple of city indices
- a Euclidean point set as the TSP instance

## Move Mechanisms

- swap move
- insertion move
- segment reversal / 2-opt-style move

The benchmark will primarily use segment reversal, since it is historically and practically meaningful for TSP.

## Acceptance Rule

- always accept if $\Delta \le 0$
- otherwise accept with probability $e^{-\Delta / T}$

## Key Invariants

- tours remain valid permutations
- route length is rotation invariant
- delta evaluation matches full route recomputation
- best-so-far route length is monotone non-increasing

## Python Implementation Plan

- TSP utilities for distance matrices, route length, and move application
- simulated annealing driver specialized to tours
- benchmark instance generator and optional exact search for small instances
- visualizations for tours, route length traces, temperature, acceptance, and neighborhood comparisons

## R Implementation Plan

- a compact Euclidean TSP annealing implementation mirroring the Python semantics
- route-length and move helpers for didactic experiments

## Test Plan

- route validity tests
- move correctness tests
- delta-versus-full-cost checks
- exact-optimum gap checks on a small benchmark instance
- reproducibility with fixed seeds

## Benchmark Plan

- one small Euclidean instance with exact optimum for reference
- one medium instance for richer route visualizations
- repeated-run comparisons across move types
- plots for:
  - route length vs iteration
  - best route length vs iteration
  - temperature vs iteration
  - acceptance metrics vs iteration
  - final route overlays
  - final gap to optimum across move types
  - route snapshots through the annealing run

## Deferred Extensions

- TSPLIB parsing
- mixed neighborhood schedules
- asymmetric TSP
- post-annealing 2-opt polishing
