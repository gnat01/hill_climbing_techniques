# Geman-Geman

This directory implements and explains the 1984 Geman-Geman paper as a binary-image restoration problem under a Gibbs / MRF-style prior with stochastic local updates and annealing.

The practical interpretation in this repo is:

- start from a clean binary image
- corrupt it with pixel-flip noise
- treat the noisy image as the observed image
- restore the latent clean image by stochastic relaxation

## Read This First

- [summary.md](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1984-geman-stochastic-relaxation/summary.md)
  Technical seminar-style summary of the paper.
- [implementation.md](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1984-geman-stochastic-relaxation/implementation.md)
  What is implemented here and why.
- [Geman_sweep_details.md](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1984-geman-stochastic-relaxation/Geman_sweep_details.md)
  Exact meaning of a sweep, temperature handling, and what cyclic reheating would mean.
- [comparing_Geman_to_diffusion_models.md](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1984-geman-stochastic-relaxation/comparing_Geman_to_diffusion_models.md)
  Careful note connecting Geman-Geman to modern diffusion-model intuition without overstating the link.

## Code

- [python/geman_geman.py](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1984-geman-stochastic-relaxation/python/geman_geman.py)
  Core Python implementation.
- [r/geman_geman.R](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1984-geman-stochastic-relaxation/r/geman_geman.R)
  Compact R implementation for parallel pedagogy and diagnostics.
- [tests/test_geman_geman.py](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1984-geman-stochastic-relaxation/tests/test_geman_geman.py)
  Tests for the current implementation.

## What Is Implemented

The current implementation uses:

- binary images with pixel values in `{-1, +1}`
- an Ising-style smoothness prior
- a data-fidelity term tying the restored image to the noisy observation
- single-pixel Gibbs updates
- fixed-temperature Gibbs restoration
- annealed Gibbs restoration with geometric cooling across sweeps

The current annealed schedule is:

- fixed temperature within a sweep
- temperature decays geometrically between sweeps

## Procedural Benchmarks

The procedural benchmark is:

- [benchmarks/benchmark_geman_geman.py](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1984-geman-stochastic-relaxation/benchmarks/benchmark_geman_geman.py)

It generates several self-contained binary examples and compares:

- noisy image
- fixed-temperature Gibbs
- annealed restoration

Run:

```bash
cd /Users/gn/work/learn/python/hill_climbing_techniques
python papers/1984-geman-stochastic-relaxation/benchmarks/benchmark_geman_geman.py
```

Optional:

```bash
python papers/1984-geman-stochastic-relaxation/benchmarks/benchmark_geman_geman.py --size 100 --noise 0.2 --sweeps 100
```

Main outputs:

- [benchmarks/outputs/accuracy_summary_across_examples.png](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1984-geman-stochastic-relaxation/benchmarks/outputs/accuracy_summary_across_examples.png)
- [benchmarks/outputs/example_summary.csv](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1984-geman-stochastic-relaxation/benchmarks/outputs/example_summary.csv)
- per-example montages and traces in the same output directory

## Bespoke Image Benchmarks

The bespoke-image benchmark is:

- [benchmarks/benchmark_geman_geman_from_image.py](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1984-geman-stochastic-relaxation/benchmarks/benchmark_geman_geman_from_image.py)

This path is for user-supplied images. It:

- loads an input image
- auto-derives size
- optionally downsamples
- converts to grayscale
- binarizes automatically unless overridden
- adds synthetic flip noise
- restores with fixed-temperature and annealed Gibbs

Run:

```bash
python papers/1984-geman-stochastic-relaxation/benchmarks/benchmark_geman_geman_from_image.py \
  --input-image papers/1984-geman-stochastic-relaxation/input_images/pic1.jpeg
```

Useful flags:

- `--output-dir`
- `--max-dim`
- `--threshold`
- `--invert`
- `--noise`
- `--sweeps`
- `--fixed-temperature`
- `--anneal-initial-temperature`
- `--anneal-alpha`
- `--eta`
- `--coupling`

Current example outputs live under:

- [benchmarks/image_outputs](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1984-geman-stochastic-relaxation/benchmarks/image_outputs)

## Pic1 Deep Dive

There is also a focused deep-dive script for bespoke-image studies:

- [benchmarks/benchmark_geman_geman_image_deep_dive.py](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1984-geman-stochastic-relaxation/benchmarks/benchmark_geman_geman_image_deep_dive.py)

Current deep dive:

- [benchmarks/image_outputs/pic1_deep_dive](/Users/gn/work/learn/python/hill_climbing_techniques/papers/1984-geman-stochastic-relaxation/benchmarks/image_outputs/pic1_deep_dive)

That study currently includes:

- accuracy vs threshold
- accuracy vs number of sweeps

## How To Read The Results

Success cases in this module are typically:

- coherent binary shapes
- thick boundaries
- locally supported structures

Failure modes are also important:

- thin structures can be eroded
- high-frequency textures can be destroyed
- strong prior mismatch can make the restored image worse than the noisy one

This is expected for a simple ferromagnetic Ising-style prior. The model prefers local agreement, so it is naturally better at restoring smooth binary shapes than oscillatory or intricate patterns.

## What This Module Already Shows Well

- binary image restoration as posterior optimization
- stochastic local updates as a viable denoising mechanism
- improvement from noisy image to restored image on suitable shapes
- the difference between fixed-temperature Gibbs and annealed restoration
- clear failure cases when the prior is mismatched to the signal

## Current Status

This Geman-Geman module is a solid `V1`:

- core theory is documented
- code exists in Python and R
- tests pass
- procedural examples exist
- bespoke-image support exists
- diffusion-comparison note exists
- sweep semantics are documented explicitly

The next natural upgrades, if needed later, are:

- repeated-run statistics on the procedural examples
- parameter studies for `eta` and `coupling`
- fixed vs annealed vs deterministic baseline comparison
- checkerboard updates
