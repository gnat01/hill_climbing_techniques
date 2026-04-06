# How To Run

All commands below assume you are in the repo root:

```bash
cd /Users/gn/work/learn/python/hill_climbing_techniques
```

## Tests

Run:

```bash
python -m unittest papers/1953-metropolis-fast-computing-machines/tests/test_metropolis.py
```

## Main Benchmark

Script:

- [benchmarks/benchmark_metropolis.py](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1953-metropolis-fast-computing-machines/benchmarks/benchmark_metropolis.py)

Run with defaults:

```bash
python papers/1953-metropolis-fast-computing-machines/benchmarks/benchmark_metropolis.py
```

What it does:

- runs fixed-temperature Metropolis sampling on the double-well energy
- prints a temperature sweep table
- writes a CSV and PNG plots

## Flags

- `--steps`
  Number of Metropolis steps per temperature.
- `--burn-in`
  Number of initial steps discarded before computing summary statistics.
- `--seed`
  Base seed for the temperature sweep.
- `--output-dir`
  Directory for CSV and PNG outputs.

Example:

```bash
python papers/1953-metropolis-fast-computing-machines/benchmarks/benchmark_metropolis.py \
  --steps 10000 \
  --burn-in 2000 \
  --seed 321 \
  --output-dir papers/1953-metropolis-fast-computing-machines/benchmarks/outputs
```

## Output Location

Default:

- [benchmarks/outputs](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1953-metropolis-fast-computing-machines/benchmarks/outputs)

Typical generated files:

- `metropolis_temperature_sweep.csv`
- `acceptance_rate_vs_temperature.png`
- `mean_energy_vs_temperature.png`
- `mean_abs_position_vs_temperature.png`
