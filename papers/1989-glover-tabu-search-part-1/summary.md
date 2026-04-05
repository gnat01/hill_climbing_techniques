# Tabu Search, Part I

## Citation

Glover, F. (1989). *Tabu Search, Part I*.

## Technical Summary

Glover’s first tabu search paper is foundational because it replaces the memoryless logic of ordinary hill climbing with adaptive memory. The method still uses local moves, but the control law is no longer determined solely by the current neighborhood objective values. Instead, selected attributes of recent moves or solutions are declared tabu for a number of iterations, preventing immediate reversal and thereby discouraging cycling and shallow entrapment.

Let $x_t$ be the current solution and $\mathcal{N}(x_t)$ a neighborhood. In an ordinary local search, one evaluates all neighbors in $\mathcal{N}(x_t)$ and chooses the best improving one, or the best one overall under some descent policy. Tabu search inserts an additional filtering layer between “all neighbors” and “neighbors we are allowed to choose.” This filtered collection is the admissible set. If the current neighborhood is $\mathcal{N}(x_t)$ and the active tabu restrictions at time $t$ are encoded by a memory state $M_t$, then a more explicit description is

$$
\mathcal{A}(x_t, M_t) = \{x' \in \mathcal{N}(x_t) : x' \text{ is not tabu under } M_t \text{ or is released by aspiration}\}.
$$

So the admissible set is simply the subset of neighbors currently allowed to compete for selection. It is needed because tabu search does not forbid revisits by deleting them from the neighborhood definition itself; it forbids them dynamically through memory.

Formally, if a move $m$ has attribute set $A(m)$ and some attributes are tabu-active, then $m$ may be forbidden even when it improves locally. This is the first major philosophical break from ordinary hill climbing: local objective value alone no longer determines whether a move may be taken. Search history intervenes.

The method can be written schematically as

$$
x_{t+1} = \arg\min_{x' \in \mathcal{A}(x_t)} f(x'),
$$

where $\mathcal{A}(x_t)$ is the set of admissible neighbors after applying tabu restrictions and aspiration overrides. The key point is that $\mathcal{A}(x_t)$ depends on search history, so the process is non-Markovian in the solution variable alone.

This non-Markov point is worth stating carefully. A process is Markov in the state variable $x_t$ if the distribution of the next state depends only on the current state and not on the earlier trajectory. Tabu search violates this if one uses only the current solution as the state description. Two identical current solutions $x_t = x'_t$ may yield different next moves if they were reached by different histories, because their tabu lists can be different. For example, the move that would return a recently deleted edge in TSP may be forbidden in one trajectory but allowed in another, even though the current tour is the same. The evolution therefore is not Markov in $x_t$ alone. It becomes Markov only if one augments the state to include the memory contents, for example $(x_t, M_t)$.

Part I concentrates on short-term memory and strategic use of tabu restrictions. The most basic mechanism forbids reversing a recent move for tenure $\tau$. On permutation problems, one may label moves by swapped elements or deleted-added edges. On subset problems, one may label by entering and leaving variables. This attribute-based encoding is technically important because it generalizes across problem classes without requiring exact solution-level memory. Exact solution memory is often too expensive and too brittle; attribute memory is cheaper and better aligned with the actual mechanism of cycling, which often happens through move reversals or repeated structural motifs rather than literal full-state repetition.

Glover’s deeper contribution is conceptual: local search should not merely avoid revisiting exact states; it should guide future search using structured memory. Cycles are not the only failure mode of hill climbing. A search can oscillate among similar configurations within the same basin while never explicitly repeating a full state. Tabu attributes act as a low-cost repulsion field that pushes the search out of such regions. In effect, memory deforms the local landscape seen by the search. A move that looks best under the raw objective may be declared temporarily unavailable because it is strategically myopic.

The paper also introduces aspiration as a controlled exception mechanism. Aspiration is needed because a tabu rule is intentionally blunt. If the search forbids every move carrying a recently used attribute, it may accidentally block a move that is genuinely exceptional. The classic example is a move that produces a new global best solution. Declaring such a move inadmissible just because it matches a tabu attribute would mean that the memory device is now obstructing the very optimization goal it was supposed to serve.

Aspiration therefore acts as an override rule:

$$
x' \text{ tabu } \land \text{Aspire}(x', H_t) \implies x' \in \mathcal{A}(x_t, M_t),
$$

where $H_t$ denotes whatever history statistics the aspiration test uses. The most common aspiration criterion is best-so-far improvement, but the general idea is broader: a tabu move may be admitted if it is sufficiently attractive relative to search history. In practical terms, aspiration restores selectivity. Tabu restrictions encourage diversification, while aspiration prevents diversification from becoming dogmatic.

The method’s importance for hill climbing is profound. It demonstrates that local search can escape local minima without stochasticity. Simulated annealing does so by random acceptance of worsening moves. Tabu search does so deterministically by changing admissibility through memory. This gives tabu search a different operating character: less thermodynamic, more strategic.

## Seminar Notes

- Adaptive memory is the defining innovation.
- Tabu restrictions are usually attribute based, not full-state based.
- The admissible set is the dynamically filtered neighborhood after tabu and aspiration are applied.
- The process is non-Markov in $x_t$ alone because identical current solutions can have different allowed moves under different memory states.
- Aspiration prevents memory from becoming self-defeating.
