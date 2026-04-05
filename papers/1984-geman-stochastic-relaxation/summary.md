# Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images

## Citation

Geman, S., Geman, D. (1984). *Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images*.

## Technical Summary

Geman and Geman supply one of the deepest theoretical bridges between probabilistic modeling, local updates, and annealing-based optimization. The paper studies image restoration through a Bayesian formulation. Let $x$ denote the latent image and $y$ the observed corrupted image. The goal is to maximize the posterior

$$
p(x \mid y) \propto p(y \mid x) p(x),
$$

or equivalently minimize the posterior energy

$$
U(x \mid y) = -\log p(y \mid x) - \log p(x) + \text{const}.
$$

The prior is represented as a Gibbs random field on a lattice, typically with clique potentials encouraging local spatial coherence. This gives the posterior the Gibbs form

$$
p(x \mid y) = \frac{1}{Z} e^{-U(x \mid y)}.
$$

The computational problem is enormous because direct optimization over all images is combinatorial. The paper therefore uses local conditional updates derived from the Gibbs field structure. Each pixel is updated using its full conditional distribution given the rest. This is the Gibbs sampler in a form tailored to Markov random fields.

The major optimization result is the annealing theorem: under a sufficiently slow cooling schedule, a stochastic relaxation process can converge to the set of global minima of the energy function. In broad terms, the process uses local probabilistic updates at temperature $T_t$, so the transition probabilities are based on the tempered posterior

$$
\pi_{T_t}(x) \propto e^{-U(x \mid y)/T_t}.
$$

As temperature decreases, probability mass concentrates on lower-energy configurations. The significance is not merely empirical. The paper gives one of the earliest rigorous convergence arguments connecting annealing schedules to global optimization in a discrete high-dimensional setting.

For local search theory, the paper matters in three ways. First, it formalizes an energy landscape induced by probabilistic modeling rather than arbitrary objective design. Second, it demonstrates that single-site local moves can be globally meaningful when embedded in an appropriate stochastic process. Third, it sharpens the distinction between local dynamics and global objective behavior: the update is entirely local, but the stationary law encodes a global optimization criterion.

The image model also highlights a structural advantage of local search methods. The energy decomposes over cliques:

$$
U(x) = \sum_{c \in \mathcal{C}} V_c(x_c),
$$

so the change in energy from a local move can often be computed incrementally from nearby terms. This locality is exactly what makes annealing and other hill-climbing variants computationally viable on large discrete systems.

From the modern perspective, the paper is a prototype for MRF optimization, MAP inference, and energy-based learning. For this repo’s purpose, its place is more specific: it gives a technically serious account of annealing on structured combinatorial spaces and ties local stochastic search to rigorous asymptotic global-optimum guarantees.

## Seminar Notes

- Bayesian restoration becomes energy minimization.
- Gibbs fields provide the local factorization needed for efficient single-site updates.
- Slow annealing gives an asymptotic route from local stochastic moves to global minimization.
