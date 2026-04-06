# How To Run

All commands below assume you are in the repo root:

```bash
cd /Users/gn/work/learn/python/hill_climbing_techniques
```

## Tests

Run:

```bash
python -m unittest papers/1983-kirkpatrick-optimization-by-simulated-annealing/tests/test_simulated_annealing.py
```

## Main Benchmark

Script:

- [benchmarks/benchmark_simulated_annealing.py](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1983-kirkpatrick-optimization-by-simulated-annealing/benchmarks/benchmark_simulated_annealing.py)

Run with defaults:

```bash
python papers/1983-kirkpatrick-optimization-by-simulated-annealing/benchmarks/benchmark_simulated_annealing.py
```

What it does:

- runs simulated annealing on the rugged 1D landscape
- compares against greedy hill climbing
- produces schedule studies, alpha studies, log-schedule comparisons, CSV summaries, and many plots

## Flags

- `--steps`
  Number of optimization steps.
- `--seed`
  Base random seed.
- `--initial-state`
  Initial state on the 1D toy landscape.
- `--output-dir`
  Directory for CSV and PNG outputs.

Example:

```bash
python papers/1983-kirkpatrick-optimization-by-simulated-annealing/benchmarks/benchmark_simulated_annealing.py \
  --steps 4000 \
  --seed 321 \
  --initial-state 2.2 \
  --output-dir papers/1983-kirkpatrick-optimization-by-simulated-annealing/benchmarks/outputs
```

## Output Location

Default:

- [benchmarks/outputs](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1983-kirkpatrick-optimization-by-simulated-annealing/benchmarks/outputs)

Typical generated files include:

- `algorithm_summary.csv`
- `cooling_rate_sweep.csv`
- `initial_temperature_sweep.csv`
- `step_size_sweep.csv`
- `logarithmic_schedule_sweep.csv`
- `temperature_vs_iteration.png`
- `sa_best_energy_vs_iteration.png`
- `final_energy_vs_geometric_schedule_alphas.png`
- `temperature_vs_iteration_all_geometric_alphas.png`
- `best_energy_vs_iteration_all_geometric_alphas.png`
- `geometric_vs_logarithmic_temperature_curves.png`
- `geometric_vs_logarithmic_best_energy_traces.png`
- `logarithmic_schedule_summary.png`

## Notes

- the geometric alpha study is the main practical schedule comparison
- the logarithmic schedule section is included for completeness and theory context
