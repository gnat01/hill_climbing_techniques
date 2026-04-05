# Equation of State Calculations by Fast Computing Machines

## Citation

Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., Teller, E. (1953). *Equation of State Calculations by Fast Computing Machines*.

## Technical Summary

This paper is not an optimization paper in the modern sense, but it supplies the acceptance mechanism that later becomes the core of simulated annealing and, more broadly, probabilistic hill climbing with uphill moves. The setting is statistical mechanics. The objective is to estimate equilibrium expectations of a many-particle system without explicitly evaluating the full partition function

$$
Z = \int e^{-E(x)/(kT)} \, dx,
$$

which is intractable in high dimension. The central insight is that one can construct a Markov chain whose stationary distribution is the Boltzmann distribution

$$
\pi(x) \propto e^{-E(x)/(kT)},
$$

and then approximate thermodynamic averages by time averages along the chain.

The conceptual move is subtle and historically decisive. Instead of trying to compute equilibrium quantities by summing over all configurations, the paper proposes to generate a dependent sequence of configurations whose long-run visitation frequencies are already weighted correctly. In modern terms, exact integration over state space is replaced by sampling from a carefully designed transition kernel.

The paper proposes a local perturbation scheme. Starting from a configuration $x$, one proposes a nearby configuration $x'$. If the energy decreases, the move is accepted. If the energy increases by $\Delta E = E(x') - E(x) > 0$, the move is accepted with probability

$$
\alpha(x \to x') = e^{-\Delta E/(kT)}.
$$

In modern notation, this is the Metropolis kernel for symmetric proposals. The move rule induces a reversible Markov chain satisfying detailed balance:

$$
\pi(x) P(x,x') = \pi(x') P(x',x).
$$

This is the technical hinge. Once detailed balance holds and the chain is irreducible over the relevant state space, the Boltzmann law becomes stationary, so simulation can replace direct integration. The authors use this to estimate the equation of state for hard-sphere systems by sampling configurations instead of enumerating them.

The symmetric-proposal assumption is important. If $q(x,x') = q(x',x)$, then the proposal asymmetry cancels and the acceptance law depends only on the energy increment. This is why the original Metropolis rule is so elegant. Later Metropolis-Hastings generalizations restore correctness for asymmetric proposals by explicitly compensating for the proposal ratio.

From the standpoint of hill climbing, the paper makes two foundational moves. First, it legitimizes non-improving moves, but only with a temperature-controlled probability. Second, it turns local search into a stochastic dynamical system rather than a purely greedy descent. The latter is what makes barrier crossing possible: the chain can escape basins separated by energy increases that a strict hill climber would never traverse.

The paper is technically important because it demonstrates how the acceptance law follows from the target distribution rather than from heuristic intuition alone. For symmetric neighborhood proposals $q(x,x') = q(x',x)$, acceptance proportional to $\min(1, e^{-\Delta E/(kT)})$ is exactly what is needed to preserve $\pi$. This is the probabilistic skeleton later reused in combinatorial optimization: replace physical energy by objective value, reinterpret temperature as a search control parameter, and let the chain drift toward low-cost states.

One useful way to view the method is as a disciplined weakening of greedy descent. A deterministic descent method imposes the hard constraint $\Delta E \le 0$. Metropolis relaxes that constraint into a probabilistic one. Small uphill moves remain reasonably likely at moderate temperature, while large uphill moves are exponentially suppressed. This means the algorithm preserves a systematic low-energy bias without collapsing into a trap-prone monotone search.

The paper does not study cooling schedules, convergence to global optima, or finite-time optimization guarantees. Those are later developments. Its contribution is more primitive and more durable: it provides an equilibrium sampling mechanism whose local acceptance structure is computationally cheap and whose statistical justification is clean.

It is also worth separating three roles that later literature sometimes blends together:

- sampling role: draw from a Boltzmann distribution at fixed temperature
- physical role: estimate thermodynamic observables
- optimization role: use the same acceptance law as a local-search primitive for escaping local minima

The 1953 paper is fundamentally about the first two. The third is the later reinterpretation that makes the paper central to optimization history.

For a graduate-seminar reading, one should view this paper as the bridge from deterministic local search to stochastic local search. A steepest-descent method uses only the sign of $\Delta E$ and therefore gets trapped in local minima. Metropolis adds a tunable likelihood of accepting uphill moves. At high temperature, the chain behaves diffusively and explores broadly. At low temperature, it becomes increasingly greedy. Simulated annealing is then the idea of changing $T$ over time so that the chain initially mixes across basins and eventually concentrates on low-energy minima. That later reinterpretation is historically contingent, but it is mathematically latent already in this paper.

## Seminar Notes

- Core object: a Markov chain over local perturbations.
- Core law: accept all improving moves and accept worsening moves with probability $e^{-\Delta E/(kT)}$.
- Core significance for hill climbing: this is the first principled mechanism for escaping local minima while preserving a target low-energy bias.
