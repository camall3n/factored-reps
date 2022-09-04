import numpy as np

from visgrid.agents.expert.gridworld_expert import GridworldExpert

class DistanceOracle:
    def __init__(self, env):
        self.env = env
        self.expert = GridworldExpert()
        states = np.indices((env.rows, env.cols)).T.reshape(-1, 2)
        for s in states:
            for sp in states:
                # Pre-compute all pairwise distances
                self.expert.GoToGridPosition(env, s, sp)

    def pairwise_distances(self, indices, s0, s1):
        init_states = s0[indices]
        next_states = s1[indices]

        distances = [
            self.expert.GoToGridPosition(self.env, s, sp)[1]
            for s, sp in zip(init_states, next_states)
        ]

        return distances

#%%
if __name__ == '__main__':
    import seeding
    import numpy as np
    import random

    from visgrid.envs import GridworldEnv
    import matplotlib.pyplot as plt

    seeding.seed(0, np, random)
    env = GridworldEnv(rows=6, cols=6)
    env.plot()

    oracle = DistanceOracle(env)

    distances = [v[-1] for k, v in oracle.expert.saved_directions.items()]

    plt.hist(distances, bins=36)
    plt.show()
