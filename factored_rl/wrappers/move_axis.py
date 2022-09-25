import copy
import numpy as np
import gym

class MoveAxisWrapper(gym.ObservationWrapper):
    """
    Move the desired axis from source to destination
    """
    def __init__(self, env: gym.Env, source: int, destination: int):
        super().__init__(env)
        self.source = source
        self.destination = destination

        example_obs = np.empty(env.observation_space.shape)
        new_shape = np.moveaxis(example_obs, source, destination).shape
        self.observation_space = copy.deepcopy(env.observation_space)
        self.observation_space._shape = new_shape

    def observation(self, obs):
        obs = np.moveaxis(obs, self.source, self.destination)
        return obs

class MoveAxisToCHW(MoveAxisWrapper):
    def __init__(self, env: gym.Env):
        """
        Move the channel axis from HWC to CHW
        """
        super().__init__(env, -1, -3)

class MoveAxisToHWC(MoveAxisWrapper):
    def __init__(self, env: gym.Env):
        """
        Move the channel axis from CHW to HWC
        """
        super().__init__(env, -3, -1)
