# An Analysis of the Behavior of a Class of Genetic Adaptive Systems

## Citation

De Jong, K. A. (1975). *An Analysis of the Behavior of a Class of Genetic Adaptive Systems*.

## Technical Summary

De Jong’s dissertation is one of the earliest systematic empirical and conceptual studies of genetic algorithms (GAs). Although GAs are not hill climbers in the narrow sense, they belong in this corpus because they offer a contrasting global-search logic against which local-search methods can be compared and later hybridized.

The thesis studies populations of encoded candidate solutions evolving under selection, crossover, and mutation. If $P_t$ denotes the population at generation $t$, the search dynamic is not a trajectory over single solutions but over empirical distributions of schemata and strings. The key question is not whether one move improves one incumbent, but how variation and selection jointly shift sampling mass toward fit regions while preserving enough diversity for continued exploration.

De Jong’s work is especially important for parameterization. The thesis studies population size, crossover rate, mutation rate, and replacement behavior across benchmark functions. This is historically significant because early evolutionary computation could easily have remained a collection of biological metaphors. De Jong helped turn it into an experimentally disciplined optimization methodology.

The underlying representation is typically binary. Fitness-proportionate or related selection biases reproductive opportunity toward high-fitness strings. Crossover recombines building blocks, while mutation injects local perturbations. In a minimalist model, the expected sampling probability of a schema $H$ can be discussed through the schema theorem heuristic:

$$
E[m(H,t+1)] \gtrsim m(H,t)\frac{\bar{f}(H)}{\bar{f}} (1 - p_c \cdot \text{disruption})(1-p_m)^{o(H)},
$$

where $m(H,t)$ is the number of strings matching schema $H$ at generation $t$.

Even if later theory refined or criticized some of these heuristics, the thesis established a durable design perspective: optimization can be achieved through biased recombinative sampling rather than explicit local improvement. This is the main contrast with hill climbing. Hill climbing exploits a neighborhood relation in solution space. GAs exploit a population-level sampling process over representations.

For this repository, the main value of De Jong is comparative and hybrid. Many successful optimization methods later combine population-based global exploration with local improvement kernels. To understand memetic algorithms, hybrid GAs, and evolutionary local search, one needs this earlier population-search viewpoint.

## Seminar Notes

- Search occurs over populations, not a single incumbent.
- Parameter analysis is one of the thesis’s most durable contributions.
- The thesis provides an important foil to local-search thinking.
