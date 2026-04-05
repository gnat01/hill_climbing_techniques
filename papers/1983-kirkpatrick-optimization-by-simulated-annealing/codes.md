# Codes in Python and R

## What Can Be Coded

### Python

- A generic simulated annealing engine with pluggable neighborhood and schedule.
- Benchmarks on TSP, graph partitioning, and binary string landscapes.
- Cooling schedules: geometric, logarithmic, linear, and reheating variants.
- Instrumentation for acceptance rates, best-so-far curves, and temperature-energy plots.

### R

- A mirror implementation for pedagogy and statistical comparison.
- Parameter sweeps over initial temperature, cooling factor, and neighborhood size.
- Publication-grade plots comparing SA to greedy descent and random restart hill climbing.

## Extensions

- Adaptive annealing using target acceptance rates.
- Parallel tempering or replica exchange for multimodal landscapes.
- Hybrid SA with local descent after each accepted move.
- Constraint handling through penalty annealing:

$$
f_\lambda(x) = f(x) + \lambda(T) \, c(x).
$$

- Large-neighborhood SA and problem-specific move sets for routing and scheduling.
