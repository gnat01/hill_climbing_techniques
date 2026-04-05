# Tabu Search, Part II

## Citation

Glover, F. (1990). *Tabu Search, Part II*.

## Technical Summary

Part II extends the short-term memory ideas of Part I into a broader adaptive-memory framework. The paper deepens the search philosophy by distinguishing intensification and diversification and by introducing longer-horizon memory structures. If Part I explains how to avoid local cycling, Part II explains how to use accumulated history to bias the search toward either promising regions or underexplored regions.

Short-term tabu memory alone prevents immediate backtracking, but it does not tell the algorithm where to search next after escaping a basin. It is essentially a local anti-cycling device. Part II asks for more: how can memory help rank candidate moves even when none is explicitly tabu, and how can search history guide the method toward either better regions or neglected regions?

Glover’s answer is to collect additional statistics, such as frequencies with which solution components occur, and then use them strategically. Intensification favors features frequently observed in high-quality solutions, while diversification encourages rarely used attributes. In abstract form, one can reinterpret the evaluation as an augmented score

$$
\tilde{f}(x) = f(x) + \lambda D(x) - \mu I(x),
$$

where $D(x)$ penalizes overused patterns and $I(x)$ rewards elite-associated patterns. The exact formula is problem dependent; the principle is not.

This is a major conceptual upgrade over ordinary local search. The objective function is no longer the only ranking device. Search history becomes an endogenous source of surrogate structure. In effect, tabu search learns a soft model of what regions have been overexploited and what components correlate with good solutions.

The admissible set therefore has a richer role in Part II than in Part I. In Part I, admissibility is mostly a yes-no filter: tabu or not tabu, possibly with aspiration. In Part II, the search still uses that filter, but once admissible candidates are identified, memory may further bias how they are scored and selected. So there are really two memory roles:

- exclusion memory, which determines whether a move enters the admissible set at all
- evaluative memory, which changes the priority ordering inside the admissible set

That distinction is useful because it explains why tabu search develops from a clever anti-cycling trick into a general metaheuristic architecture.

Part II also broadens the operational playbook: path relinking ideas are foreshadowed, strategic oscillation is discussed more explicitly, and memory is cast as a layered control system rather than a single prohibition list. Strategic oscillation is especially important in constrained optimization. Instead of remaining strictly feasible, the search can cross the feasibility boundary and oscillate between feasible and infeasible regions, using penalties or control logic to exploit the structure near the boundary. This is another place where admissibility and aspiration need context. A method that never permits boundary crossing may be too conservative; a method that crosses freely may become erratic. Tabu-style memory and aspiration-style overrides provide a disciplined way to decide when prohibitions should hold and when they should yield.

The paper also clarifies that diversification should not mean random restart in a naive sense. Restarting from scratch discards learned information. Diversification in tabu search is informed displacement. One seeks solutions that differ materially in key attributes while still leveraging memory about what kinds of structures may be promising. This is a more nuanced escape mechanism than either pure randomness or rigid descent.

The non-Markov character becomes even stronger here. In Part I, history affects move admissibility through short-term tabu status. In Part II, history also affects scoring through long-term frequencies, elite-solution records, and strategic control modes. Thus even if the current solution $x_t$ is fixed, the next move can differ because the search may currently be in an intensification phase, a diversification phase, or a strategic-oscillation phase, all determined by prior trajectory information. Again, the process can be made Markov only by expanding the state to include the relevant memory structures and control-mode variables.

From the perspective of hill-climbing techniques, Part II is where tabu search becomes a full metaheuristic. The search is no longer just “local search plus prohibition.” It becomes a memory-driven architecture in which move admissibility, evaluation, restarting, and region selection all depend on accumulated experience. This paper therefore marks a transition from local-search mechanics to adaptive search design.

## Seminar Notes

- Short-term memory prevents cycling; long-term memory shapes exploration.
- Intensification and diversification are complementary control modes.
- Memory affects both admissibility and ranking.
- The search is non-Markov in the raw solution state because control depends on accumulated trajectory information.
- The paper turns tabu search into a general adaptive-memory metaheuristic.
