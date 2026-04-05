# Implementation Plan

## Paper Objective

Implement the Metropolis acceptance mechanism as a reusable primitive for both equilibrium sampling and optimization-oriented local search.

## Algorithmic Core

For a current state $x$ and proposed state $x'$, define

$$
\Delta E = E(x') - E(x).
$$

Accept the move with probability

$$
\alpha(x \to x') = \min(1, e^{-\Delta E / T}),
$$

assuming a symmetric proposal kernel and units with Boltzmann constant absorbed into temperature.

## State Representation

- generic Python state type for reusable kernels
- scalar-state examples for continuous toy landscapes
- small discrete-state examples for exact sanity checks

## Move or Proposal Mechanism

- user-supplied proposal function in Python
- user-supplied proposal function in R
- benchmark examples with symmetric random-walk proposals

## Acceptance Rule

- always accept if $\Delta E \le 0$
- otherwise accept with probability $e^{-\Delta E / T}$

## Key Invariants

- acceptance probability is exactly `1.0` for non-worsening moves
- acceptance probability is in `(0, 1)` for worsening moves at positive temperature
- deterministic reproducibility under a fixed random seed
- lower temperature decreases worsening-move acceptance for fixed $\Delta E > 0$

## Python Implementation Plan

- reusable `MetropolisSampler` class
- `MetropolisStep` and `MetropolisRunResult` data objects for diagnostics
- helper acceptance-probability function
- simple benchmark landscape based on a double-well energy

## R Implementation Plan

- a clear `metropolis_sampler` function with trace outputs
- exact same acceptance semantics as Python
- small example runner for exploratory diagnostics

## Test Plan

- direct tests of acceptance probability
- fixed-seed reproducibility test
- stationarity-oriented sanity check on a tiny finite state space
- optimization-style check that low temperature favors lower-energy states

## Benchmark Plan

- run the sampler on a double-well landscape
- compare occupancy, mean energy, and acceptance rate across temperatures
- print a compact tabular summary suitable for regression checking

## Deferred Extensions

- asymmetric proposals and Metropolis-Hastings
- adaptive proposal scales
- simulated annealing schedules
- combinatorial optimization example such as TSP tour perturbations
