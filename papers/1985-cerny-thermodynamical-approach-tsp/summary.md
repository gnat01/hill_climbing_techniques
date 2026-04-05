# Thermodynamical Approach to the Traveling Salesman Problem

## Citation

Cerny, V. (1985). *Thermodynamical Approach to the Traveling Salesman Problem: An Efficient Simulation Algorithm*.

## Technical Summary

Cerny’s paper is one of the early demonstrations that simulated annealing can be specialized effectively to a canonical NP-hard combinatorial problem, the traveling salesman problem (TSP). The contribution is not merely to restate the simulated annealing idea, but to operationalize it in a concrete permutation space with carefully chosen local moves and computationally efficient cost updates.

Let a tour be represented by a permutation $\sigma$ over cities, with objective

$$
f(\sigma) = \sum_{i=1}^{n} d(\sigma_i, \sigma_{i+1}),
$$

where indices wrap around cyclically. The search space is discrete and highly multimodal. Greedy local search with weak neighborhoods gets trapped quickly. Cerny’s approach is to define a neighborhood over permutations and accept proposed modifications using a Boltzmann factor

$$
\Pr(\text{accept}) =
\begin{cases}
1 & \Delta \le 0, \\
e^{-\Delta/T} & \Delta > 0.
\end{cases}
$$

The crucial technical point is that on permutations, the neighborhood design determines both computational cost and search power. A local exchange or segment reversal can have a cost difference $\Delta$ computed from only a few broken and reconnected edges rather than from full tour recomputation. This makes high-volume stochastic exploration feasible.

The paper shows how annealing functions as a basin-escape device in TSP. At high temperature, poor rearrangements are often accepted, allowing macro-scale reconfiguration of tour structure. As temperature decreases, the process transitions toward local refinement. The thermodynamic analogy is useful but subordinate to the algorithmic insight: the local move distribution and the cooling schedule together induce a search through permutation space that is neither purely random nor prematurely greedy.

An implicit lesson of the paper is that annealing’s effectiveness depends on move granularity. If moves are too small, the chain diffuses slowly and may need impractically many accepted uphill transitions to leave a bad basin. If moves are too large, most proposals become poor and acceptance collapses. The TSP setting makes this tension explicit because one can compare exchange-style moves, insertion moves, and segment reversals.

The paper also helped establish a template still used today for optimization experiments: define a state representation, define local moves with incremental objective updates, choose an initial temperature based on typical uphill costs, run for a fixed number of iterations per temperature, then cool geometrically or quasi-geometrically until freezing. This experimental protocol is more important historically than any single theoretical statement in the paper.

In seminar terms, Cerny is best read as the combinatorial engineering counterpart to the general simulated annealing paper. Kirkpatrick et al. provide the universal idea; Cerny shows how it becomes a practical optimization method on permutations. The paper’s importance for a hill-climbing corpus is that it makes explicit how a local-search method becomes problem effective only after representation and neighborhood are handled with equal care.

## Seminar Notes

- State space: tours as permutations.
- Technical hinge: incremental edge-based computation of $\Delta$ under local moves.
- Historical role: early convincing evidence that annealing could tackle hard combinatorial optimization.
