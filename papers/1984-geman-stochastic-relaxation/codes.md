# Codes in Python and R

## What Can Be Coded

### Python

- Binary image denoising with an Ising-style prior and Gibbs updates.
- Simulated annealing for MAP restoration using single-pixel flips.
- Efficient local energy-difference computation over 4-neighbor or 8-neighbor grids.
- Visualization of temperature, posterior energy, and restored image quality.

### R

- Small-lattice pedagogical implementations for binary or ternary image restoration.
- Heatmaps and trace plots showing convergence behavior under different schedules.
- Comparison of pure Gibbs sampling versus annealed Gibbs sampling.

## Extensions

- Move from binary images to Potts models with $q$ labels.
- Add checkerboard updates and blocked Gibbs moves for speed.
- Compare Gibbs annealing against graph cuts on submodular binary energies.
- Add a deterministic local optimizer for post-annealing polishing.
- Extend from image restoration to generic pairwise MRF optimization problems.
