# Comparing Geman-Geman to Diffusion Models

This note is here to make a careful conceptual connection between the 1984 Geman-Geman paper and the current wave of diffusion-based generative image models. There is a real relationship in spirit, but it is easy to overstate it. The right comparison is: Geman-Geman is a historically important probabilistic image-restoration framework that shares some broad conceptual themes with diffusion models, but it is not a direct precursor in the modern deep-learning sense.

## Why The Comparison Is Reasonable

At a high level, both Geman-Geman and diffusion models are about stochastic image transformation under a probabilistic model.

In Geman-Geman:

- an observed noisy image $y$ is assumed to come from a corrupted version of a latent clean image $x$
- one defines a posterior distribution

$$
p(x \mid y) \propto p(y \mid x) p(x)
$$

- the prior $p(x)$ is often represented as a Gibbs or Markov random field
- restoration is performed by stochastic local updates, often with annealing

In diffusion models:

- a clean image is progressively corrupted by a forward noising process
- a model is trained to learn the reverse denoising process
- generation or restoration proceeds through a sequence of stochastic denoising steps

So in both cases:

- there is a noisy-to-clean image transformation
- uncertainty is modeled probabilistically
- stochastic dynamics are central rather than incidental
- the denoising process is iterative

This is enough to justify a conceptual comparison.

## What Carries Over Conceptually

### 1. Denoising as Probabilistic Inference

Geman-Geman treats image restoration as inference under a posterior energy. Modern diffusion models also support a denoising interpretation, although in a different formalism. In both settings, “denoising” is not just filtering; it is the recovery of plausible clean structure under a probabilistic model.

### 2. Stochastic Dynamics Matter

Neither framework is purely deterministic at its core. Geman-Geman uses Gibbs-style stochastic relaxation and annealing. Diffusion models use stochastic or quasi-stochastic reverse denoising trajectories. In both cases, randomness is part of the mechanism for navigating a complicated image space.

### 3. Local Corruption and Global Structure

Geman-Geman uses local clique interactions to impose global visual coherence. Diffusion models, especially in the learned reverse process, also recover globally coherent structure from noisy inputs through many small denoising steps. The mechanics are different, but the broad idea that local updates can accumulate into global structure is shared.

### 4. Energy or Score Bias Toward Plausible Images

In Geman-Geman, plausible images are those with low posterior energy. In modern diffusion and score-based models, the learned denoising field or score function points toward regions of higher data density. These are not the same mathematical objects, but both frameworks bias trajectories toward visually plausible image configurations.

## Where The Comparison Breaks

This is where precision matters.

### 1. Geman-Geman Is Not A Diffusion Model

The Geman-Geman paper does not define the forward-noising and learned reverse-denoising framework used in DDPMs, score-based diffusion models, or modern latent diffusion systems. It is a Gibbs/Markov-random-field restoration paper, not a neural generative-model paper.

### 2. No Learned Neural Network Parameterization

Geman-Geman uses explicit probabilistic structure:

- likelihood term
- Gibbs prior
- local conditional distributions

Diffusion models, by contrast, usually rely on large neural networks trained on large datasets to approximate denoising or score functions.

### 3. Local Conditional Updates Versus Learned Reverse-Time Dynamics

Geman-Geman updates one site or a block of sites using conditionals derived from the Gibbs distribution. Diffusion models use learned denoising transitions that operate in a very different high-dimensional function-approximation regime. The resemblance is conceptual, not algorithmically close.

### 4. MAP/Posterior Restoration Versus General-Purpose Generation

Geman-Geman is fundamentally about restoring or inferring an image under a specified observation model. Modern diffusion models are often trained as broad generative models capable of unconditional or conditional synthesis well beyond a single restoration problem.

## The Best Historical Framing

The strongest honest claim is:

Geman-Geman is an important early example of probabilistic image restoration through stochastic iterative denoising under an explicit image prior. In that broad sense, it is part of the intellectual background for later probabilistic and generative approaches to images, including the modern intuition that one can move from noise toward image structure through a sequence of stochastic refinement steps.

The weaker claims that should be avoided are:

- “Geman-Geman is basically diffusion”
- “diffusion models come directly from Geman-Geman”
- “the Geman-Geman algorithm is an early DDPM”

Those are not technically defensible.

## How To Use This Comparison In This Repository

When we implement Geman-Geman here, the comparison to diffusion models should be used in a limited way:

- to help modern readers understand why the paper still feels relevant
- to connect old image restoration ideas to current generative-model intuition
- to show a historical line from explicit probabilistic image priors to modern learned denoising systems

But the implementation itself should remain faithful to the paper:

- Gibbs or MRF-style prior
- posterior energy
- stochastic relaxation
- annealing or temperature-based inference
- denoising demonstrations on simple images

That fidelity matters more than forcing a trendy analogy.

## Short Version

Geman-Geman and diffusion models are related at the level of probabilistic denoising intuition, stochastic iterative refinement, and moving from noisy images toward structured ones. But Geman-Geman is an explicit Gibbs-field restoration framework, while diffusion models are learned modern generative denoisers. The connection is real, but it is conceptual and historical rather than direct or identity-like.
