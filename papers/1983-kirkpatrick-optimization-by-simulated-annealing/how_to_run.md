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

## Custom f5 Landscape Benchmark

Script:

- [benchmarks/benchmark_simulated_annealing_f5_landscape.py](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1983-kirkpatrick-optimization-by-simulated-annealing/benchmarks/benchmark_simulated_annealing_f5_landscape.py)

Run with defaults:

```bash
python papers/1983-kirkpatrick-optimization-by-simulated-annealing/benchmarks/benchmark_simulated_annealing_f5_landscape.py
```

Run with a fixed Gaussian-perturbed landscape:

```bash
python papers/1983-kirkpatrick-optimization-by-simulated-annealing/benchmarks/benchmark_simulated_annealing_f5_landscape.py \
  --gaussian-noise \
  --noise-mean 30 \
  --noise-sd 10
```

What it does:

- builds the custom discrete `f5` landscape
- optionally adds one fixed Gaussian realization across the entire landscape
- compares greedy hill climbing and simulated annealing on that same frozen landscape
- writes the landscape CSV, both trajectories, repeated-run summaries, and comparison plots

Flags:

- `--M`
  Number of states in `{0, ..., M-1}`.
- `--k`
  Block-width parameter. Must divide `M`.
- `--c`
  Ruggedness scale in the `f5` family.
- `--R`
  Shift parameter.
- `--steps`
  Number of iterations for greedy and SA.
- `--initial-state`
  Shared starting state for both algorithms.
- `--initial-temperature`
  Initial SA temperature.
- `--alpha`
  Geometric cooling factor.
- `--repeats`
  Number of repeated runs per algorithm in the summary plot.
- `--seed`
  Base seed used for the fixed landscape realization and repeated runs.
- `--gaussian-noise`
  Enable a pointwise Gaussian perturbation on the landscape.
- `--noise-mean`
  Required when `--gaussian-noise` is set.
- `--noise-sd`
  Required when `--gaussian-noise` is set.
- `--output-dir`
  Directory for CSV and PNG outputs.

Example:

```bash
python papers/1983-kirkpatrick-optimization-by-simulated-annealing/benchmarks/benchmark_simulated_annealing_f5_landscape.py \
  --M 120 \
  --k 20 \
  --c 0.4 \
  --R 60 \
  --steps 800 \
  --initial-state 10 \
  --initial-temperature 40 \
  --alpha 0.98 \
  --repeats 40 \
  --seed 123 \
  --gaussian-noise \
  --noise-mean 30 \
  --noise-sd 10
```

Default output location:

- [benchmarks/f5_outputs](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1983-kirkpatrick-optimization-by-simulated-annealing/benchmarks/f5_outputs)

## Multi-Start f5 Benchmark

Script:

- [benchmarks/benchmark_simulated_annealing_f5_multi_start.py](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1983-kirkpatrick-optimization-by-simulated-annealing/benchmarks/benchmark_simulated_annealing_f5_multi_start.py)

What it does:

- freezes one `f5` landscape realization
- selects many distinct initial states on that same landscape
- compares greedy hill climbing and simulated annealing across those starts
- reports mean best energy, mean gap to global minimum, and global-minimum hit rate

Run:

```bash
python papers/1983-kirkpatrick-optimization-by-simulated-annealing/benchmarks/benchmark_simulated_annealing_f5_multi_start.py \
  --num-starts 100 \
  --gaussian-noise \
  --noise-mean 30 \
  --noise-sd 10
```

Flags:

- `--M`
- `--k`
- `--c`
- `--R`
- `--steps`
- `--initial-temperature`
- `--alpha`
- `--num-starts`
- `--seed`
- `--gaussian-noise`
- `--noise-mean`
- `--noise-sd`
- `--output-dir`

Default output location:

- [benchmarks/f5_multi_start_outputs](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1983-kirkpatrick-optimization-by-simulated-annealing/benchmarks/f5_multi_start_outputs)

## Empirical SA Basin Benchmark

Script:

- [benchmarks/benchmark_simulated_annealing_empirical_basin.py](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1983-kirkpatrick-optimization-by-simulated-annealing/benchmarks/benchmark_simulated_annealing_empirical_basin.py)

What it does:

- freezes one `f5` landscape realization
- runs simulated annealing repeatedly from many initial states
- estimates the empirical basin of attraction of SA
- reports hit probability and mean gap to optimum as functions of the initial state

Run:

```bash
python papers/1983-kirkpatrick-optimization-by-simulated-annealing/benchmarks/benchmark_simulated_annealing_empirical_basin.py \
  --M 120 \
  --k 20 \
  --c 0.4 \
  --R 60 \
  --steps 8000 \
  --initial-temperature 4000 \
  --alpha 0.98 \
  --runs-per-start 30 \
  --start-step 1 \
  --success-epsilon 0.0 \
  --seed 123 \
  --gaussian-noise \
  --noise-mean 40 \
  --noise-sd 10
```

Flags:

- `--M`
- `--k`
- `--c`
- `--R`
- `--steps`
- `--initial-temperature`
- `--alpha`
- `--runs-per-start`
  Clamped into `[10, 150]`.
- `--start-step`
- `--success-epsilon`
- `--seed`
- `--gaussian-noise`
- `--noise-mean`
- `--noise-sd`
- `--output-dir`

Default output location:

- [benchmarks/f5_empirical_basin_outputs](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1983-kirkpatrick-optimization-by-simulated-annealing/benchmarks/f5_empirical_basin_outputs)
