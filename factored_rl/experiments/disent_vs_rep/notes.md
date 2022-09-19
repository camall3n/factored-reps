# Disent vs. Representation

## Experiment 00

### Purpose

1. Measure disentanglement for multiple representations & compare
2. Visualize & verify transforms work correctly

### Setup

- Transforms:
  - identity
  - rotate
  - permute_factors
  - permute_states
  - images
- Envs:
  - gridworld
  - taxi
- Metrics:
  - DCI (regressor)
  - MIG (mutual information)

### Findings

- `permute` transforms look reasonable
- Bug in `rotate` => int casting was causing states to overlap
  - Fixed. Updated `rotate` transform looks reasonable
- Need better way to visualize `images` transform; people are still confused
- Disentanglement results are mixed:
  - `permute_states` destroys disentanglement; `permute_factors` has no effect
  - `rotate` destroys disentanglement for gridworld, less so for taxi
  - `images` more destructive for gridworld than taxi
    - Might be related to regressor?
    - Taxi has more relevant pixels per factor?
    - => __*Need to reproduce.*__ Images may have been blank!
