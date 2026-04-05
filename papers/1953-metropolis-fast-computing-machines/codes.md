# Codes in Python and R

## What Can Be Coded

The paper naturally supports a reusable Metropolis sampler and a small optimization-oriented variant.

### Python

- A generic Metropolis sampler for discrete or continuous state spaces with user-supplied `energy(x)` and `proposal(x)`.
- A combinatorial optimization demo where $E(x)$ is replaced by an objective such as TSP tour length.
- Diagnostics for acceptance rate, energy trace, and empirical occupation frequencies.
- Visualizations of the effect of temperature on exploration and basin crossing.

### R

- The same kernel can be implemented idiomatically with functions for energy and proposals.
- R is especially suitable for quick diagnostics: trace plots, histograms of visited energies, and comparison across temperatures.
- For discrete optimization examples, base R plus `ggplot2` is sufficient.

## Extensions

- Generalize from symmetric proposals to Metropolis-Hastings:

$$
\alpha(x \to x') = \min\left(1, \frac{\pi(x') q(x',x)}{\pi(x) q(x,x')}\right).
$$

- Add adaptive proposal scales for continuous spaces.
- Add a cooling schedule to turn the sampler into simulated annealing.
- Compare pure Metropolis against greedy hill climbing and threshold accepting on the same landscape.
- Build teaching demos on double-well potentials and small TSP instances to show barrier crossing.
