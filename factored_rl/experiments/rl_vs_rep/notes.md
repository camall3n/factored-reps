# RL vs. Representation

## Experiment 00 - [trials 03 & 04]

### Purpose

1. Measure RL performance for a selection of different representations.
    - In particular, do disentangled representations lead to improved RL performance?

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
- Other details:
  - n_training_episodes: 200
  - n_steps_per_episode: 500

### Findings

- Noisy initial results for 10 random seeds => increasing to 100 seeds helped.
- Gridworld results show clear learning for `identity` and `rotate`.
  - Rotate does surprisingly well.
- Taxi results still far too noisy to be useful.
- Results for `images` surprisingly poor, especially for gridworld[effectively]one-hot)
  - => Discovered exploring starts bug. Taxi goal depot was unintentionally fixed.
  - => Discovered int casting bug. All images were unintentionally blank.

-----

## Experiment 01 - [trials 05 & 06]

### Purpose

1. Reproduce previous gridworld results after fixing several bugs and updated config settings
2. Get proper results for images
3. Increase taxi training time to see if learning is happening (leaving goal fixed)

### Setup

- gridworld:
  - (previous settings)
- taxi:
  - n_training_episodes: 2000
  - n_steps_per_episode: 500

### Findings

- Image-based training works => gridworld converges to expert performance
- Extended training helps distinguish taxi performance curves
  - Domain may still be too difficult
- Sensible results for the representations:
  - Ground truth (`identity`) performs well in both environments
    - `identity` beats `permute_factors` which beats `permute_states`
  - Gridworld `rotate` matches `identity`, but on taxi, `rotate` is much worse
    - Possibly due to multiple-variable entanglement
  - Results for `images` are mixed:
    - On gridworld, MLP and CNN perform well
      - some slight ringing indicates learning rate may need tuning
    - On Taxi, MLP `images` performed poorly
    - Taxi CNN failed due to architecture size mismatch
- Accidentally disabled noise for all experiments

-----

## Experiment 02 - [trials 07 & 08]

### Purpose

1. Speed up experiments & shrink cnn to allow cpu training
2. Make taxi easier
    - Enforce `depot_dropoff_only`
3. Re-enable noise and compare against no-noise
4. Ensure CNN works for taxi

### Setup

- taxi:
  - n_steps_per_episode: 50 (to leverage exploring starts better)
  - n_training_episodes: 10000 (to ensure adequate learning time)
- gridworld:
  - n_steps_per_episode: 50 (to match new taxi settings)
  - n_training_episodes: 500 (to decrease max training time by half)

### Findings

- No difference between noise and no-noise experiments => stick with noise
- Mixed results for `images`:
  - Significant ringing on gridworld & catastrophic forgetting on taxi suggests learning rate is too high
  - On gridworld, `images` converges to expert while `identity` and `rotate` plateau before that => all three may require more tuning
- Results still sensible for vector representations.
  - `permute_states` is consistently the worst (though taxi MLP also sucks)
  - `identity` consistently beats `permute_factors` which beats `permute_states`
  - `rotate` still has a bigger effect on taxi than on gridworld w.r.t. `identity`

-----
