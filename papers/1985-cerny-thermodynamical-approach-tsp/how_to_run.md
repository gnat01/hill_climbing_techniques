# How To Run

All commands below assume you are in the repo root:

```bash
cd /Users/gn/work/learn/python/hill_climbing_techniques
```

## Tests

Run:

```bash
python -m unittest papers/1985-cerny-thermodynamical-approach-tsp/tests/test_tsp_annealing.py
```

## Original Benchmark (`V0`)

Script:

- [benchmarks/benchmark_tsp_annealing.py](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1985-cerny-thermodynamical-approach-tsp/benchmarks/benchmark_tsp_annealing.py)

Run:

```bash
python papers/1985-cerny-thermodynamical-approach-tsp/benchmarks/benchmark_tsp_annealing.py
```

Flags:

- `--steps`
  Annealing steps for the medium instance.
- `--seed`
  Base random seed.
- `--output-dir`
  Directory for CSV and PNG outputs.

Example:

```bash
python papers/1985-cerny-thermodynamical-approach-tsp/benchmarks/benchmark_tsp_annealing.py \
  --steps 6000 \
  --seed 321 \
  --output-dir papers/1985-cerny-thermodynamical-approach-tsp/benchmarks/outputs
```

## Tightened Benchmark (`V1`)

Script:

- [benchmarks/benchmark_tsp_annealing_v1.py](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1985-cerny-thermodynamical-approach-tsp/benchmarks/benchmark_tsp_annealing_v1.py)

This is the stronger benchmark and should usually be preferred.

Run:

```bash
python papers/1985-cerny-thermodynamical-approach-tsp/benchmarks/benchmark_tsp_annealing_v1.py
```

Flags:

- `--steps`
  Annealing steps.
- `--seed`
  Base random seed.
- `--repeats`
  Repeated runs per move type for summary statistics.
- `--output-dir`
  Directory for CSV and PNG outputs.

Example:

```bash
python papers/1985-cerny-thermodynamical-approach-tsp/benchmarks/benchmark_tsp_annealing_v1.py \
  --steps 800 \
  --seed 321 \
  --repeats 100 \
  --output-dir papers/1985-cerny-thermodynamical-approach-tsp/benchmarks/outputs_v1
```

## Output Locations

Defaults:

- `V0`:
  [benchmarks/outputs](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1985-cerny-thermodynamical-approach-tsp/benchmarks/outputs)
- `V1`:
  [benchmarks/outputs_v1](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1985-cerny-thermodynamical-approach-tsp/benchmarks/outputs_v1)

## Notes

- `V0` is the original first working benchmark
- `V1` keeps `V0` intact but uses a deliberately bad initial tour and repeated-run statistics as the primary result
- if you want the more defensible paper-style result, use `V1`
