# Optimization by Simulated Annealing

## Citation

Kirkpatrick, S., Gelatt, C. D., Vecchi, M. P. (1983). *Optimization by Simulated Annealing*.

## Technical Summary

This paper is the canonical transplantation of the Metropolis mechanism from statistical physics into combinatorial optimization. The central analogy is direct: a configuration of a physical system corresponds to a candidate solution; energy corresponds to objective value; thermal equilibrium at temperature $T$ corresponds to sampling with bias toward low-cost configurations; annealing corresponds to gradually lowering $T$ so that the system freezes into a low-energy state.

The algorithmic loop is simple. Given a current solution $x_t$, propose a local perturbation $x'$. Let

$$
\Delta = f(x') - f(x_t)
$$

for a minimization problem. If $\Delta \le 0$, accept the move. Otherwise accept with probability

$$
\Pr(\text{accept}) = e^{-\Delta/T_t}.
$$

Then reduce temperature according to a cooling schedule $T_0 > T_1 > \cdots$. The paper emphasizes that the schedule must be slow enough to permit near-equilibration at each temperature, but in practice finite-time schedules are used as heuristics rather than exact equilibrium procedures.

The paper’s technical force lies in its reinterpretation of local search. Classical hill climbing descends until no improving neighbor exists. Simulated annealing replaces monotone descent by a nonstationary Markov process. High temperatures flatten the acceptance landscape, increasing mobility and enabling transitions across objective barriers. Low temperatures restore greediness, focusing the search in a promising basin. In rugged landscapes, this helps separate transient trapping from genuine high-quality minima.

The authors illustrate the method on archetypal hard optimization problems, notably layout and partitioning problems, and discuss VLSI-style applications. The contribution is not just the rule itself, which is inherited from Metropolis, but the optimization interpretation: one need not sample the equilibrium distribution accurately forever; rather, one should drive the distribution through a sequence of temperatures so that mass shifts toward better and better configurations.

From a technical perspective, the method can be interpreted through Gibbs measures

$$
\pi_T(x) = \frac{e^{-f(x)/T}}{Z(T)}.
$$

As $T \to 0$, the measure concentrates on global minimizers. The optimization question becomes whether an inhomogeneous Markov chain can track this concentration process sufficiently well. The 1983 paper is heuristic and empirical in emphasis; later work supplies asymptotic convergence conditions, typically involving logarithmically slow cooling. But even in this early form, the paper correctly identifies the practical tradeoff: search power comes from allowing objective increases, yet computational efficiency requires cooling far faster than equilibrium statistical mechanics would prescribe.

The paper is influential because it offers a universal design pattern rather than a problem-specific heuristic. To instantiate simulated annealing one must specify only three ingredients: a representation, a neighborhood operator, and a cooling schedule. This makes it portable across graph partitioning, routing, placement, scheduling, and other NP-hard problems.

The paper also reveals a now-standard insight about metaheuristics: the move operator and the acceptance rule interact strongly. Small neighborhoods produce fine local adaptation but slow barrier crossing. Large neighborhoods increase exploration but may disrupt local improvement structure. Simulated annealing is therefore not just a temperature schedule attached to arbitrary moves; it is a search process whose quality depends on the geometry induced by the neighborhood system.

## Seminar Notes

- Physics-to-optimization translation: $E(x)$ becomes $f(x)$.
- Nonzero temperature gives a probabilistic escape from local minima.
- Cooling transforms a stationary local sampler into a global optimization heuristic.
