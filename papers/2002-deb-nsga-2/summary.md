# A Fast and Elitist Multi-Objective Genetic Algorithm: NSGA-II

## Citation

Deb, K., Pratap, A., Agarwal, S., Meyarivan, T. (2002). *A Fast and Elitist Multi-Objective Genetic Algorithm: NSGA-II*.

## Technical Summary

NSGA-II is one of the most influential papers in evolutionary optimization because it makes multiobjective search computationally practical and algorithmically clean. The setting is optimization of a vector objective

$$
F(x) = (f_1(x), \dots, f_M(x)),
$$

where solutions are compared by Pareto dominance rather than a single scalar objective. A solution $x$ dominates $y$ if it is no worse in all objectives and strictly better in at least one.

The original NSGA had conceptual promise but computational and elitism deficiencies. NSGA-II resolves these by introducing three key mechanisms: fast nondominated sorting, elitist survival using parent-offspring merging, and crowding distance to maintain spread along the Pareto front.

Given a combined population $R_t = P_t \cup Q_t$ of size $2N$, the algorithm sorts solutions into nondomination fronts $F_1, F_2, \dots$. The next parent population is filled front by front until adding the next front would exceed capacity. Within the partially admitted front, solutions are ranked by crowding distance, which estimates local sparsity in objective space. Solutions with larger crowding distance are preferred to preserve diversity.

Crowding distance for objective $m$ is based on normalized neighboring gaps in sorted objective order, and aggregate crowding is the sum across objectives. Binary tournament selection then prefers lower nondomination rank and, under ties, larger crowding distance.

The paper’s importance to a hill-climbing corpus is twofold. First, it shows how population-based search can be extended from scalar optimization to frontier approximation without scalarization. Second, it becomes a natural anchor for later hybrid multiobjective local-search methods. Once a Pareto archive exists, local search can be used to improve individual archive members or explore neighborhoods around sparsely represented front regions.

From a technical perspective, NSGA-II is also a lesson in algorithm engineering. The fast sorting routine reduces the computational burden enough to make the framework usable. Elitism prevents loss of discovered nondominated solutions. Crowding distance replaces an explicit niching parameter with a more robust geometry-aware diversity proxy. Together these changes converted an appealing concept into a standard baseline.

## Seminar Notes

- Optimization target is a Pareto set, not a single optimum.
- Rank plus crowding gives a practical survival and selection rule.
- NSGA-II is a cornerstone for hybrid evolutionary-local multiobjective search.
