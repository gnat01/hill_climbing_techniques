# The Reactive Tabu Search

## Citation

Battiti, R., Tecchiolli, G. (1994). *The Reactive Tabu Search*.

## Technical Summary

Reactive tabu search is important because it addresses one of the main weaknesses of early tabu search: the need to hand-tune tabu tenure. A tenure that is too small fails to prevent cycles; a tenure that is too large suppresses useful moves and slows intensification. Battiti and Tecchiolli convert tenure from a fixed parameter into a dynamically adapted control variable driven by search behavior.

The central observation is diagnostic. If the search repeatedly revisits previous configurations or exhibits short recurrence cycles, then the current memory horizon is too weak. Conversely, if recurrence is rare and the search appears overconstrained, the tenure may be reduced. The algorithm therefore monitors repetition patterns and updates tabu tenure online.

This turns tabu search into a feedback-controlled local search. Let $\tau_t$ denote tabu tenure. Rather than fixing $\tau_t = \tau$, the algorithm adjusts it according to observed recurrence characteristics. While the paper’s rules are implementation specific, the generic logic is

$$
\tau_{t+1} =
\begin{cases}
\tau_t + \delta_+ & \text{if recurrence is too frequent}, \\
\max(\tau_{\min}, \tau_t - \delta_-) & \text{otherwise}.
\end{cases}
$$

The exact mechanism may include multiplicative updates and escape phases, but the principle is consistent: search parameters should respond to search pathology.

The paper also introduces a stronger notion of search memory diagnosis. It is not enough to know whether a move is currently tabu; one should monitor the empirical dynamics of the trajectory itself. This is a methodological step toward self-tuning metaheuristics. Rather than calibrating by offline parameter sweeps, the algorithm infers from its own trajectory whether it is trapped, overcycling, or exploring effectively.

Reactive tabu search is especially relevant to hill climbing because it makes a local-search method adaptive at runtime without using global model learning. The control signal is endogenous and cheap: recurrence intervals, repetitions, and neighborhood behavior. In modern language, it is a lightweight online controller over an underlying combinatorial search process.

Conceptually, the paper also strengthens the connection between local search and dynamical systems. A search trajectory is no longer just a sequence of candidate solutions; it is an object that can be measured, diagnosed, and controlled. Once that viewpoint is adopted, tenure becomes only one adjustable variable among many. Candidate-list size, aspiration threshold, perturbation strength, and evaluation bias can all be adapted in the same spirit.

## Seminar Notes

- Fixed tabu tenure is replaced by online adaptation.
- Recurrence statistics serve as a feedback signal.
- The paper is a major early example of self-tuning local search.
