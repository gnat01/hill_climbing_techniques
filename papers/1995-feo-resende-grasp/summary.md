# Greedy Randomized Adaptive Search Procedures

## Citation

Feo, T. A., Resende, M. G. C. (1995). *Greedy Randomized Adaptive Search Procedures*.

## Technical Summary

GRASP is a multi-start metaheuristic built from two repeatedly executed phases: randomized greedy construction and local search. Its importance lies in the decomposition of search into a constructive diversification phase and a descent-based intensification phase. This architecture is especially valuable when pure hill climbing depends strongly on the starting solution.

Each GRASP iteration begins with a partial solution. At each construction step, candidate elements are scored greedily, but instead of selecting the single best candidate, the algorithm forms a restricted candidate list (RCL) of high-quality options and selects randomly from that list. If $g(c)$ denotes the marginal greedy score of candidate $c$, one common rule is to include in the RCL all candidates satisfying

$$
g(c) \le g_{\min} + \alpha (g_{\max} - g_{\min}),
$$

for minimization, with $\alpha \in [0,1]$. The parameter $\alpha$ controls the greediness-randomness tradeoff. After construction, a local search phase descends from the constructed solution until a local optimum is reached.

The technical significance of GRASP is that it explicitly isolates two tasks that naive hill climbing conflates: generating a promising basin and exploiting that basin. The construction phase provides structured diversity. The local search phase provides deterministic or stochastic improvement. Because each iteration is independent except for optional enhancements, the method is naturally parallelizable and easy to analyze experimentally.

The “adaptive” aspect refers to the way greedy scores are recomputed as the partial solution evolves. This matters because the marginal value of an element depends on previously selected elements. The construction phase therefore performs a myopic but state-dependent rollout rather than static sorting.

GRASP is often underestimated because its outer loop is simple. But that simplicity is precisely its strength. Unlike elaborate memory-based methods, GRASP offers a clean baseline architecture in which most problem-specific sophistication can be concentrated in the greedy function and the local search neighborhood. It is thus one of the most reusable frameworks in combinatorial optimization.

From a hill-climbing viewpoint, GRASP is a principled answer to start-point dependence. If hill climbing reaches only a local optimum relative to its initial basin, then one should control the distribution of initial basins. Pure random restart samples basins indiscriminately. GRASP biases the starts toward good structural regions while preserving enough randomness to cover multiple basins.

## Seminar Notes

- Two-phase structure: biased randomized construction followed by local search.
- The RCL controls the exploration-exploitation tradeoff.
- GRASP is one of the clearest restart-based generalizations of hill climbing.
