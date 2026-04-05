# Variable Neighborhood Search

## Citation

Mladenovic, N., Hansen, P. (1997). *Variable Neighborhood Search*.

## Technical Summary

Variable Neighborhood Search (VNS) is built on a sharp empirical and conceptual claim: a local minimum with respect to one neighborhood need not be a local minimum with respect to another. This sounds elementary, but it has major algorithmic consequences. Instead of trying to escape a basin through memory or stochastic acceptance alone, VNS changes the neighborhood system itself.

Let $\mathcal{N}_1, \mathcal{N}_2, \ldots, \mathcal{N}_k$ be a sequence of neighborhoods of increasing scale or structural difference. Basic VNS alternates between a shaking phase and a local descent phase. Starting from an incumbent solution $x$, it samples a point $x'$ from $\mathcal{N}_i(x)$, then applies local search to reach a local optimum $x''$. If $f(x'') < f(x)$, the incumbent is replaced and the search returns to the smallest neighborhood. Otherwise it moves to the next larger neighborhood.

In pseudomathematical terms:

$$
x \leftarrow \text{LocalSearch}(\text{Shake}(x, \mathcal{N}_i)),
$$

with neighborhood index reset on improvement and incremented otherwise.

The technical beauty of VNS is that it operationalizes escape by structured perturbation. Unlike simulated annealing, worsening moves are not accepted according to a temperature law. Unlike tabu search, history does not explicitly constrain move admissibility. Instead, the search attacks local optimality itself by changing the definition of locality.

This is particularly powerful on combinatorial problems where neighborhoods have clear inclusion relationships. For routing problems, one may progress from small exchanges to large segment moves. For subset problems, one may move from single-bit flips to multi-bit swaps. The method therefore exploits a geometric fact about search spaces: ruggedness is neighborhood relative.

VNS also has a clean complexity-performance profile. Large neighborhoods need not be searched exhaustively. The shaking step can sample from them stochastically, and the expensive optimization burden is pushed onto the local search routine. This makes VNS a framework rather than a single algorithm; its effectiveness depends on the design and ordering of neighborhoods and on the efficiency of the embedded local search.

The paper’s enduring contribution is to show that changing the neighborhood system is itself a metaheuristic principle. Earlier local search traditions largely fixed the neighborhood and changed only the control law. VNS changes both the geometry and the dynamics. That shift has influenced countless hybrid heuristics.

## Seminar Notes

- Local optimality is neighborhood dependent.
- VNS escapes basins by moving across a family of neighborhoods.
- The framework is simple but structurally very expressive.
