# An Effective Heuristic Algorithm for the Traveling-Salesman Problem

## Citation

Lin, S., Kernighan, B. W. (1973). *An Effective Heuristic Algorithm for the Traveling-Salesman Problem*.

## Technical Summary

Lin-Kernighan (LK) is one of the most influential local-search heuristics ever proposed. While it is problem specific, its conceptual impact is broad: it shows how variable-depth local search can drastically outperform fixed-neighborhood descent. For the TSP, simple $k$-opt methods improve tours by removing $k$ edges and reconnecting the fragments. The challenge is that large $k$ yields enormous neighborhoods. Lin and Kernighan avoid exhaustive enumeration by choosing a sequence of edge exchanges adaptively.

The algorithm begins with a current tour and constructs a gain sequence. At each step it selects an edge to remove and an edge to add so that the cumulative gain remains promising. If the partial exchange sequence can be closed into a feasible tour with positive total gain, the move is accepted. Otherwise the sequence may be extended, subject to feasibility and gain constraints.

In modern shorthand, the algorithm searches a variable-depth neighborhood:

$$
\Delta = \sum_{i=1}^{r} \bigl( d(x_i) - d(y_i) \bigr),
$$

where removed edges contribute positive gain and inserted edges consume gain. The search continues while partial gains justify deeper exploration.

The critical innovation is not merely variable $k$, but sequential pruning. Rather than enumerate all $k$-opt moves, LK constructs only gain-promising exchange sequences. This creates a neighborhood that is simultaneously rich and computationally tractable. The method is therefore a landmark in the design of large neighborhoods with embedded search.

From a hill-climbing perspective, LK is a lesson in neighborhood engineering. The algorithm remains fundamentally local: it transforms one tour into another via edge exchanges. Yet its locality is dynamic and context dependent. The neighborhood explored from a solution depends on promising partial moves discovered online. This is much more sophisticated than fixed 2-opt descent and explains the method’s enduring practical dominance.

LK also changes the semantics of “improvement step.” A single accepted move can encode a composite sequence of local edits that would be invisible to shallow descent. In basin terms, variable-depth search can jump between tours that fixed-depth neighborhoods would treat as separated by non-improving intermediates. That makes LK a powerful escape mechanism without needing stochastic acceptance or explicit memory.

## Seminar Notes

- Variable-depth edge exchanges define a huge but selectively explored neighborhood.
- Positive partial gains are used to prune the search.
- LK is a canonical example of advanced deterministic local search.
