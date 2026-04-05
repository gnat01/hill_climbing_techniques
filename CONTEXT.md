# Context

This repository is meant to become a durable working context for statistical-physics-plus-optimization problems. It is not just a reading list and not just a code dump. The objective is to build a reusable research and implementation corpus around energy landscapes, local search, stochastic search, and adaptive metaheuristics.

## Scope

The core themes are:

- statistical mechanics viewpoints on optimization
- energy-based modeling and Gibbs distributions
- Markov-chain Monte Carlo foundations relevant to search
- hill climbing and local search
- probabilistic acceptance methods such as Metropolis and simulated annealing
- memory-based and adaptive methods such as tabu search
- restart, perturbation, and neighborhood-changing methods such as GRASP, ILS, and VNS
- evolutionary and population-based search where it materially connects to local search

## Working Principles

- Historical progression matters. Earlier papers should be implemented first because later methods often reuse their primitives.
- Every paper should have a technically serious summary, not a superficial overview.
- Every implementation should make the algorithmic object explicit: state, neighborhood or proposal kernel, acceptance or selection rule, memory state if any, and stopping criteria.
- Python and R are both first-class when justified. Python is the default for reusable engineering; R is especially useful for diagnostics, statistical comparison, and quick experimental visualization.
- Code must be PR-ready, tested, benchmarkable, and reusable as future reference material.

## What Counts As Good Context

For this repository to be useful later, each paper folder should answer the following questions quickly:

1. What exact optimization or sampling problem is the paper solving?
2. What is the state representation?
3. What are the allowed local moves, proposals, or population operators?
4. What rule governs evolution: greedy descent, Metropolis acceptance, tabu admissibility, Pareto survival, and so on?
5. What theoretical claims matter: detailed balance, convergence, asymptotic concentration, complexity, approximation, or empirical dominance?
6. What code should exist now, and what extensions are natural later?

## Repository Shape

At a minimum, each paper directory should contain:

- `summary.md`: graduate-seminar-level technical summary with equations where needed
- `codes.md`: implementation opportunities in Python and R, including natural extensions
- `implementation.md`: the concrete implementation plan for this repo
- `python/`: Python source for the paper’s main algorithms or primitives
- `r/`: R source for equivalent or diagnostic implementations when justified
- `tests/`: paper-specific tests
- `benchmarks/`: reproducible benchmark or experiment entry points

Not every paper will have the same volume of code, but the structure should stay consistent unless there is a strong reason to deviate.

## Cross-Cutting Themes

Several ideas recur across papers and should be tracked explicitly as the repository grows:

- objective as energy: $E(x)$ versus optimization cost $f(x)$
- local geometry: neighborhood systems, proposal kernels, and move operators
- basin escape: uphill acceptance, memory, perturbation, or changing neighborhoods
- state augmentation: adding temperature, memory, archives, or population state changes the dynamics fundamentally
- Markov versus non-Markov search: many methods are Markov only after expanding the state description
- intensification versus diversification
- computational locality: efficient delta evaluation is often the difference between a clean idea and a practical algorithm

## Historical Coding Order

The intended implementation order is:

1. Metropolis (1953)
2. Simulated annealing foundations and optimization reinterpretation
3. Structured annealing and energy-based restoration
4. Deterministic local-search strengthening such as Lin-Kernighan
5. Tabu search and adaptive memory
6. Multi-start and perturbation frameworks such as GRASP, VNS, and ILS
7. Evolutionary methods and hybrids

This order should be followed unless there is a strong dependency-driven reason not to.
