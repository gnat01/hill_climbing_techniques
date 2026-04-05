# Paper Implementation Template

Use this structure for every paper unless there is a concrete reason to deviate.

## Required Files

- `summary.md`
- `codes.md`
- `implementation.md`
- `python/`
- `r/`
- `tests/`
- `benchmarks/`

## `implementation.md`

This file should contain:

1. Paper objective
2. Algorithmic core
3. State representation
4. Move or proposal mechanism
5. Acceptance, selection, or survival rule
6. Key invariants or theoretical properties worth preserving in code
7. Python implementation plan
8. R implementation plan
9. Test plan
10. Benchmark plan
11. Extensions deferred for later

## Python Expectations

- Prefer a small reusable library over a single script.
- Make stochastic components reproducible through explicit seeds or injected RNG objects.
- Expose diagnostics needed for analysis, not just final answers.
- Avoid unnecessary dependencies in early implementations.

## R Expectations

- Prioritize clarity, diagnostics, and experiment reproducibility.
- Mirror the conceptual structure of the Python implementation where reasonable.
- Use plotting-oriented outputs where they materially help interpretation.

## Tests

Tests should check the most important algorithmic invariants, for example:

- move acceptance semantics
- probability or score calculations
- state-update correctness
- reproducibility under fixed seeds
- expected qualitative behavior on a small benchmark problem

## Benchmarks

Each paper should include at least one benchmark or experiment entry point that:

- runs without hidden dependencies
- uses a documented seed
- emits interpretable metrics
- is small enough for regression checking

## Documentation Standard

The summary should explain the mathematics. The implementation should explain the engineering choices. The tests and benchmarks should make the algorithm trustworthy.
