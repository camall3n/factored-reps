from typing import SupportsFloat

import gym
import numpy as np
from scipy.stats import special_ortho_group

class RotationWrapper(gym.ObservationWrapper):
    """
    Applies a random (fixed) rotation matrix to all observations
    """
    def __init__(self, env: gym.Env, axes=None):
        super().__init__(env)

        ob_space = env.observation_space
        assert np.issubdtype(ob_space.dtype, np.floating)
        if axes is None:
            self.n_dims = ob_space.shape[0]
            self.axes = np.arange(self.n_dims)
        else:
            self.axes = np.asarray(axes)
            self.n_dims = len(self.axes)

        if self.n_dims == 2:
            self._rotation_matrix = self._get_rotation_matrix(np.pi / 4)
        else:
            self._rotation_matrix = special_ortho_group.rvs(self.n_dims,
                                                            random_state=self.np_random)

    def observation(self, obs):
        factors = obs[self.axes]
        r_factors = self._rotation_matrix @ factors
        r_factors /= np.sqrt(self.n_dims) # rescale so that factor values stay within same range
        obs[self.axes] = r_factors
        return obs

    def _get_rotation_matrix(self, radians: float):
        return np.array([
            [np.cos(radians), -np.sin(radians)],
            [np.sin(radians), np.cos(radians)],
        ])
