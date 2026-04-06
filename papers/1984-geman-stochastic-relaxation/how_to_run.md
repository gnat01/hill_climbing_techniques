# How To Run

All commands below assume you are in the repo root:

```bash
cd /Users/gn/work/learn/python/hill_climbing_techniques
```

## Tests

Run:

```bash
python -m unittest papers/1984-geman-stochastic-relaxation/tests/test_geman_geman.py
```

## Procedural Benchmark

Script:

- [benchmarks/benchmark_geman_geman.py](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1984-geman-stochastic-relaxation/benchmarks/benchmark_geman_geman.py)

Run with defaults:

```bash
python papers/1984-geman-stochastic-relaxation/benchmarks/benchmark_geman_geman.py
```

Flags:

- `--size`
  Image size for procedural examples.
- `--noise`
  Pixel-flip probability for the main examples.
- `--sweeps`
  Number of Gibbs sweeps.
- `--output-dir`
  Directory for CSV and PNG outputs.

Example:

```bash
python papers/1984-geman-stochastic-relaxation/benchmarks/benchmark_geman_geman.py \
  --size 100 \
  --noise 0.2 \
  --sweeps 100 \
  --output-dir papers/1984-geman-stochastic-relaxation/benchmarks/outputs
```

## Bespoke Image Benchmark

Script:

- [benchmarks/benchmark_geman_geman_from_image.py](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1984-geman-stochastic-relaxation/benchmarks/benchmark_geman_geman_from_image.py)

Run with defaults:

```bash
python papers/1984-geman-stochastic-relaxation/benchmarks/benchmark_geman_geman_from_image.py \
  --input-image papers/1984-geman-stochastic-relaxation/input_images/pic1.jpeg
```

Flags:

- `--input-image`
  Source image path. Required.
- `--noise`
  Pixel-flip probability after binarization.
- `--sweeps`
  Number of Gibbs sweeps.
- `--max-dim`
  Optional maximum image dimension after resizing.
- `--threshold`
  Manual grayscale threshold. If omitted, Otsu thresholding is used.
- `--invert`
  Invert the binarized image.
- `--fixed-temperature`
  Fixed Gibbs temperature.
- `--anneal-initial-temperature`
  Initial annealing temperature.
- `--anneal-alpha`
  Geometric decay factor across sweeps.
- `--eta`
  Data-fidelity weight.
- `--coupling`
  Smoothness weight.
- `--output-dir`
  Directory for output images and CSV.

Example:

```bash
python papers/1984-geman-stochastic-relaxation/benchmarks/benchmark_geman_geman_from_image.py \
  --input-image papers/1984-geman-stochastic-relaxation/input_images/crisscross.png \
  --max-dim 128 \
  --noise 0.2 \
  --sweeps 60 \
  --threshold 140 \
  --invert \
  --fixed-temperature 1.2 \
  --anneal-initial-temperature 2.8 \
  --anneal-alpha 0.95 \
  --eta 2.2 \
  --coupling 1.1 \
  --output-dir papers/1984-geman-stochastic-relaxation/benchmarks/image_outputs
```

## Bespoke Image Deep Dive

Script:

- [benchmarks/benchmark_geman_geman_image_deep_dive.py](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1984-geman-stochastic-relaxation/benchmarks/benchmark_geman_geman_image_deep_dive.py)

Use this for threshold and sweep studies on one input image.

Required flags:

- `--input-image`
- `--output-dir`

Other flags:

- `--max-dim`
- `--invert`
- `--noise`
- `--eta`
- `--coupling`
- `--fixed-temperature`
- `--anneal-initial-temperature`
- `--anneal-alpha`
- `--repeats`

Example:

```bash
python papers/1984-geman-stochastic-relaxation/benchmarks/benchmark_geman_geman_image_deep_dive.py \
  --input-image papers/1984-geman-stochastic-relaxation/input_images/pic1.jpeg \
  --output-dir papers/1984-geman-stochastic-relaxation/benchmarks/image_outputs/pic1_deep_dive
```

## Method Comparison Benchmark

Script:

- [benchmarks/benchmark_geman_geman_method_comparison.py](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1984-geman-stochastic-relaxation/benchmarks/benchmark_geman_geman_method_comparison.py)

This compares:

- deterministic local descent
- fixed-temperature Gibbs
- annealed Gibbs

Flags:

- `--size`
- `--noise`
- `--sweeps`
- `--output-dir`

Run:

```bash
python papers/1984-geman-stochastic-relaxation/benchmarks/benchmark_geman_geman_method_comparison.py
```

## Output Locations

Defaults:

- procedural examples:
  [benchmarks/outputs](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1984-geman-stochastic-relaxation/benchmarks/outputs)
- bespoke images:
  [benchmarks/image_outputs](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1984-geman-stochastic-relaxation/benchmarks/image_outputs)
- method comparison:
  [benchmarks/method_comparison_outputs](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1984-geman-stochastic-relaxation/benchmarks/method_comparison_outputs)

## Notes

- the current implementation restores binary images, not general grayscale photographs directly
- for bespoke images, the clean latent image is the binarized version produced before synthetic noise is added
- thin or high-frequency patterns may fail under the simple smoothness prior; this is a model limitation, not just a tuning issue
