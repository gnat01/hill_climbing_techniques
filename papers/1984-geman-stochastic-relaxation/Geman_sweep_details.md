# Geman Sweep Details

## What A Sweep Means Here

In the current Geman-Geman implementation, one sweep means:

- every pixel is visited once
- the visit order is randomized
- during that sweep, the temperature is held fixed

So a sweep is one full lattice pass under a single temperature value.

## Temperature Handling

The temperature does **not** change within a sweep.

Instead:

- sweep `0` uses the initial temperature
- sweep `1` uses a lower temperature
- sweep `2` uses a lower one again
- and so on

The current annealed run uses a geometric cooling schedule across sweeps:

$$
T_s = T_0 \alpha^s, \qquad 0 < \alpha < 1
$$

where:

- $s$ is the sweep index
- $T_0$ is the starting temperature
- $\alpha$ is the geometric decay factor

So the correct interpretation is:

- fixed temperature within each sweep
- decaying temperature between sweeps

## Why This Is Reasonable

This is a standard and clean implementation choice.

It means each sweep has a clear thermodynamic meaning:

- at sweep $s$, the whole lattice is updated under temperature $T_s$

rather than having temperature drift continuously during the same pass through the image.

## Cyclic Reheating

If cyclic reheating were implemented, the temperature schedule would no longer decay monotonically.

Instead, temperature would sometimes be raised again after cooling, for example:

- cool for several sweeps
- reheat to a higher temperature
- cool again

Conceptually, this would:

- reintroduce randomness after the system has become too rigid
- help the process escape bad basins or over-smoothed states
- act like a controlled “shake” of the current restored image

So cyclic reheating would turn the schedule from:

- monotone cooling

into:

- cooling plus occasional temperature resets or pulses

That is a legitimate extension, but it is **not** the same as the current implementation.

## Current Status

Current Geman-Geman implementation:

- random pixel order within each sweep
- one visit per pixel per sweep
- fixed temperature within sweep
- geometric cooling across sweeps

Cyclic reheating:

- not currently implemented
- would be a future extension if we want to study reheating or temperature-pulsing effects
