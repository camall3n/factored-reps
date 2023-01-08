# Adapted from https://github.com/samlobel/q_functionals

import gym
from gym import spaces
import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import Legendre

def cartesian_sum(max_sum, n_dim):
    ranks_to_add = list(range(0, max_sum + 1))
    if n_dim == 1: # base case.
        return [[r] for r in ranks_to_add]
    cartesian_products_to_return = []
    for cartesian_pair in cartesian_sum(max_sum, n_dim - 1):
        sum_cartesian_pair = sum(cartesian_pair)
        for r in ranks_to_add:
            if (sum_cartesian_pair + r) <= max_sum:
                cartesian_products_to_return.append(cartesian_pair + [r])
    return cartesian_products_to_return

class BasisFunction:
    basis_element_multiplicity = 1 # Number of terms per basis element

    def __init__(self, ndim: int, rank: int) -> None:
        if ndim < 1:
            raise ValueError('Input must have at least 1 dimension')
        if rank < 0:
            raise ValueError('Rank must be >= 0')
        self.ndim = ndim
        self.rank = rank

        self.basis_terms = np.array(cartesian_sum(self.rank, self.ndim), dtype=np.float32)
        self.basis_terms = self.basis_terms.reshape(-1, self.ndim) # (n_features, ndim)
        self.n_features = len(self.basis_terms)

    def _get_basis_features(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Cannot get features with abstract base class. Use a specific basis.")

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute a batch of basis features for a batch of observations
        """
        is_batch = (obs.ndim == 2)
        if not is_batch:
            obs = obs[np.newaxis, :]
        features = self._get_basis_features(obs)
        if not is_batch:
            obs = obs.squeeze(0)
            features = features.squeeze(axis=0)
        return features

class IdentityBasisFunction(BasisFunction):
    basis_element_multiplicity = 1

    def __init__(self, ndim: int) -> None:
        super().__init__(ndim=ndim, rank=0)
        self.basis_terms = np.ones((1, ndim))
        self.n_features = ndim

    def _get_basis_features(self, obs):
        return obs

class PolynomialBasisFunction(BasisFunction):
    basis_element_multiplicity = 1

    def _get_basis_features(self, obs):
        basis_terms = self.basis_terms
        obs_repeated = np.repeat(obs[..., np.newaxis], repeats=basis_terms.shape[0], axis=-1)
        obs_power = obs_repeated**basis_terms.T
        obs_prod = np.prod(obs_power, axis=-2)
        return obs_prod

class LegendreBasisFunction(BasisFunction):
    basis_element_multiplicity = 1

    def __init__(self, ndim: int, rank: int):
        self.rank = rank
        self.legendre_coefficients = self.compute_legendre_matrix(self.rank) # (1, npow, npow)
        self.powers_needed = np.arange(self.rank + 1, dtype=np.float32) # (npow, )
        super().__init__(ndim, rank)

    def compute_legendre_matrix(self, rank: int) -> np.ndarray:
        max_order = rank + 1
        coefficients_for_orders = []
        for order in range(max_order):
            legendre_poly = Legendre.basis(order, [-1, 1]).convert(kind=Polynomial)
            legendre_coefs = list(legendre_poly.coef)
            coefficients = legendre_coefs + [0] * (max_order - len(legendre_coefs))
            coefficients_for_orders.append(np.array(coefficients, dtype=np.float32))

        legendre_coefs = np.array(coefficients_for_orders)
        legendre_coefs = legendre_coefs.reshape((1, *legendre_coefs.shape))
        return legendre_coefs

    def _get_basis_features(self, obs: np.ndarray):
        basis_terms = self.basis_terms.astype(np.int64)

        npow = self.powers_needed.shape[0]
        obs_repeated = np.repeat(obs[..., np.newaxis], repeats=npow, axis=-1) # (B, ndim, npow)
        obs_powers = obs_repeated**self.powers_needed # (B, ndim, npow)
        obs_powers = np.swapaxes(obs_powers, -1, -2) # (B, npow, ndim)

        legendre_1d_eval = np.matmul(self.legendre_coefficients, obs_powers) # (B, npow, ndim)
        legendre_1d_eval = np.swapaxes(legendre_1d_eval, 1, 2) # (B, ndim, npow)

        indices = basis_terms.T[np.newaxis, ...] # (1, ndim, n_feat)
        indices = np.repeat(indices, repeats=legendre_1d_eval.shape[0], axis=0) # (B, ndim, n_feat)
        accumulator = np.take_along_axis(legendre_1d_eval, axis=-1, indices=indices) # (same)
        features = np.prod(accumulator, axis=-2).astype(np.float32) # (B, n_feat)
        return features

class FourierBasisFunction(BasisFunction):
    basis_element_multiplicity = 2

    def __init__(self, ndim: int, rank: int, half_period: np.floating = 2.0):
        super().__init__(ndim, rank)
        self.half_period = half_period

    def _get_basis_features(self, obs):
        basis_terms = np.swapaxes(self.basis_terms, 0, 1) # (ndim, n_basis_terms)
        thetas = np.matmul(obs, basis_terms) * (np.pi / self.half_period) # (batch, n_basis_terms)

        sines = np.sin(thetas) # (batch, n_basis_terms)
        cosines = np.cos(thetas) # (batch, n_basis_terms)

        features = np.concatenate((sines, cosines), axis=-1) # (batch, multiplicity*n_basis_terms)
        return features

class NormalizedRBFBasis(BasisFunction):
    pass

class BasisWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, basis_fn: BasisFunction):
        super().__init__(env)
        self.basis_fn = basis_fn

        ob_space = env.observation_space
        assert isinstance(ob_space, spaces.Box)
        assert np.issubdtype(ob_space.dtype, np.floating)
        assert len(ob_space.shape) == 1, 'Basis wrapper requires flattened observations'
        assert ob_space.is_bounded('both')
        if not (np.allclose(ob_space.high, 1) and np.allclose(ob_space.low, -1)):
            raise ValueError('Observation space must have range [-1, 1] for each dimension')

        n_features = self.basis_fn.basis_element_multiplicity * len(self.basis_fn.basis_terms)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(n_features, ))

    def observation(self, obs):
        return self.basis_fn(obs)

class PolynomialBasisWrapper(BasisWrapper):
    def __init__(self, env: gym.Env, rank: int):
        ndim = env.observation_space.shape[0]
        super().__init__(env, basis_fn=PolynomialBasisFunction(ndim, rank))

class LegendreBasisWrapper(BasisWrapper):
    def __init__(self, env: gym.Env, rank: int):
        ndim = env.observation_space.shape[0]
        super().__init__(env, basis_fn=LegendreBasisFunction(ndim, rank))

class FourierBasisWrapper(BasisWrapper):
    def __init__(self, env: gym.Env, rank: int):
        ndim = env.observation_space.shape[0]
        half_period = (env.observation_space.high[0] - env.observation_space.low[0])
        super().__init__(env, basis_fn=FourierBasisFunction(ndim, rank, half_period))
