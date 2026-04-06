# f5 Benchmark Notes

This note records how to think about the custom discrete `f5` landscape benchmarks in the Kirkpatrick paper directory.

## Landscape

The custom family used in the benchmark is:

$$
f_5(x) = \left\lfloor \frac{2(x-R)^2}{k} \right\rfloor + \left\lfloor c\,((x-R)\bmod k)^2 \right\rfloor
$$

with optional pointwise Gaussian perturbation added once to the landscape:

$$
\widetilde f_5(x) = f_5(x) + \varepsilon_x, \qquad \varepsilon_x \sim \mathcal N(\mu, \sigma^2).
$$

When Gaussian noise is enabled in the benchmark, the realization is sampled once and then held fixed for the entire experiment. This is deliberate: greedy hill climbing and simulated annealing must be compared on the same realized landscape, not on different noisy draws.

## Why `mean_best_energy - mean_gap` Is Constant

In the multi-start summary, the reported gap is

$$
\text{gap} = \text{best energy reached} - \text{global minimum energy of the frozen landscape}.
$$

So for each algorithm,

$$
\text{mean best energy} - \text{mean gap}
$$

is just the global minimum energy of that fixed sampled landscape. That is why the offset is constant across algorithms in the summary table.

This does not mean the objective itself has zero mean or that the gap is "equal to the noise mean". It only reflects how the gap statistic is defined.

## Why Greedy Barely Changes When `steps` Increases

On a fixed landscape, greedy hill climbing is effectively deterministic once the start state is fixed. After it falls into a local minimum, additional iterations do not help. So increasing `--steps` does not materially change greedy once the local basin has already trapped it.

## Why SA Improves with Higher Temperature and Longer Runs

For this `f5` family, the traps are real enough that weak SA settings are not very impressive. Larger values of:

- `--initial-temperature`
- `--steps`

give SA more opportunity to:

- accept transient uphill moves,
- cross barriers early,
- explore outside the starting basin,
- and then cool into a better basin later.

This is exactly the mechanism Kirkpatrick-style simulated annealing is supposed to exploit.

## Useful Commands

Repo root:

```bash
cd /Users/gn/work/learn/python/hill_climbing_techniques
```

### Single-run `f5` benchmark

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

### 100-start benchmark, moderate SA

```bash
python papers/1983-kirkpatrick-optimization-by-simulated-annealing/benchmarks/benchmark_simulated_annealing_f5_multi_start.py \
  --M 120 \
  --k 20 \
  --c 0.4 \
  --R 60 \
  --steps 800 \
  --initial-temperature 40 \
  --alpha 0.98 \
  --num-starts 100 \
  --seed 123 \
  --gaussian-noise \
  --noise-mean 30 \
  --noise-sd 10
```

### 100-start benchmark, stronger SA

This is a more revealing setting on the noisy `f5` landscape:

```bash
python papers/1983-kirkpatrick-optimization-by-simulated-annealing/benchmarks/benchmark_simulated_annealing_f5_multi_start.py \
  --M 120 \
  --k 20 \
  --c 0.4 \
  --R 60 \
  --steps 8000 \
  --initial-temperature 4000 \
  --alpha 0.98 \
  --num-starts 100 \
  --seed 123 \
  --gaussian-noise \
  --noise-mean 40 \
  --noise-sd 10
```

In that stronger run, simulated annealing improves materially over greedy:

- greedy mean gap stays large because it remains trapped in local basins,
- SA mean gap drops significantly,
- SA global-minimum hit rate rises as the schedule becomes hot enough and long enough to cross barriers.

## What to Look At

For the single-run benchmark:

- trajectory on the landscape
- state vs iteration
- current and best energy vs iteration
- temperature and rolling acceptance metrics

For the multi-start benchmark:

- best energy vs initial state
- gap to global minimum vs initial state
- SA advantage vs initial state
- multi-start summary plot

These together give both:

- the mechanistic story of how SA moves,
- and the broader empirical story of when SA helps across many starting conditions.
