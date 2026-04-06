# 1D Landscapes

This directory is a sandbox for plotting and discussing discrete 1D objective functions before deciding whether they are worth integrating into the Kirkpatrick simulated-annealing section.

Supported families:

$$
f(x) = \left\lfloor \frac{2x}{k} \right\rfloor + \left\lfloor \frac{c\,x(M-1-x)}{M} \right\rfloor
$$

$$
f_3(x) = \left\lfloor \frac{2(x-R)^2}{k} \right\rfloor - \left\lfloor \frac{c\,(x-R)^4}{M^2} \right\rfloor
$$

$$
f_5(x) = \left\lfloor \frac{2(x-R)^2}{k} \right\rfloor + \left\lfloor c\,((x-R) \bmod k)^2 \right\rfloor
$$

with:

- $x \in \{0,1,\dots,M-1\}$
- $k \mid M$
- $c > 0$
- $R \in \{0,1,\dots,M-1\}$

## Run

From repo root:

```bash
cd /Users/gn/work/learn/python/hill_climbing_techniques
python 1d_landscapes/plot_landscapes.py
```

Optional flags:

- `--M`
- `--family`
- `--ks`
- `--cs`
- `--R`
- `--gaussian-noise`
- `--noise-mean`
- `--noise-sd`
- `--seed`
- `--output-dir`

Example:

```bash
python 1d_landscapes/plot_landscapes.py \
  --M 120 \
  --family f3 \
  --ks 10,20,30 \
  --cs 0.1,0.2,0.4,0.8 \
  --R 60
```

Example with noisy `f5`:

```bash
python 1d_landscapes/plot_landscapes.py \
  --family f5 \
  --M 120 \
  --ks 20 \
  --cs 0.4 \
  --R 60 \
  --gaussian-noise \
  --noise-mean 30 \
  --noise-sd 10 \
  --seed 123
```

Outputs go by default to:

- [outputs](/Users/gn/work/learn/python/hill_climbing_techniques/1d_landscapes/outputs)
