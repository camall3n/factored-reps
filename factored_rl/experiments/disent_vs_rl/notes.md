# RL vs. Disentanglement

## Experiment 00

### Purpose

1. Determine whether any of the existing disentanglement metrics correlates with RL performance

### Setup

- envs
  - gridworld
  - taxi

- Summarize the disentanglement results to get a disentanglement score for each env/transform/metric
- Summarize the RL results to get a performance score for each env/transform/agent
- Merge the results on their (env/transform) to get disent vs. RL results for each metric

### Findings

- In gridworld, RL performance can vary widely for representations with the same disentanglement value
- In taxi, RL performance might be very roughly correlated with disentanglement score
  - But still need to tune `images` transform (and probably the others as well)
    - And anyway, does the CNN really _use_ the representation we're measuring here?
    - Probably should measure the penultimate layer or conv output or something
      - And probably should re-train RL agent with the fixed representation
- Bottom line: none of the metrics is well correlated with RL performance across both domains

-----
