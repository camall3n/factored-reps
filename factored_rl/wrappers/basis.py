import gym
from gym import spaces
import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import Legendre
import torch

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

class BasisWrapper(gym.ObservationWrapper):
    basis_element_multiplicity = 1 # Number of terms per basis element

    def __init__(self, env: gym.Env, rank):
        super().__init__(env)
        self.rank = rank

        ob_space = env.observation_space
        assert isinstance(ob_space, spaces.Box)
        assert np.issubdtype(ob_space.dtype, np.floating)
        assert len(ob_space.shape) == 1, 'Basis wrapper requires flattened observations'
        assert ob_space.is_bounded('both')
        if not (np.allclose(ob_space.high, 1) and np.allclose(ob_space.low, -1)):
            raise ValueError('Observation space must have range [-1, 1] for each dimension')

        self.ndim = ob_space.shape[0]

        self.basis_terms = np.array(cartesian_sum(self.rank, self.ndim), dtype=np.float32)
        self.basis_terms = self.basis_terms.reshape(-1, self.ndim) # (n_basis_terms, ndim)
        n_basis_terms = len(self.basis_terms)

        n_features = self.basis_element_multiplicity * n_basis_terms
        self.observation_space = spaces.Box(low=-1, high=1, shape=(n_features, ))

    def _get_basis_features(self, obs):
        raise NotImplementedError(
            "Cannot get features with abstract base class. Use a specific basis.")

    def get_basis_features(self, obs):
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

    def observation(self, obs):
        return self.get_basis_features(obs)

class PolynomialBasis(BasisWrapper):
    basis_element_multiplicity = 1

    def __init__(self, env, rank):
        super().__init__(env, rank)

    def _get_basis_features(self, obs):
        basis_terms = self.basis_terms
        obs_repeated = np.repeat(obs[..., np.newaxis], repeats=basis_terms.shape[0], axis=-1)
        obs_power = obs_repeated**basis_terms.T
        obs_prod = np.prod(obs_power, axis=-2)
        return obs_prod

class FourierBasis(BasisWrapper):
    basis_element_multiplicity = 2

    def __init__(self, env, rank):
        super().__init__(env, rank)
        self.half_period = (self.observation_space.high[0] - self.observation_space.low[0])

    def _get_basis_features(self, obs):
        basis_terms = np.swapaxes(self.basis_terms, 0, 1) # (ndim, n_basis_terms)
        thetas = np.matmul(obs, basis_terms) * (np.pi / self.half_period) # (batch, n_basis_terms)

        sines = np.sin(thetas) # (batch, n_basis_terms)
        cosines = np.cos(thetas) # (batch, n_basis_terms)

        features = np.concatenate((sines, cosines), axis=-1) # (batch, multiplicity*n_basis_terms)
        return features

class LegendreBasis(BasisWrapper):
    basis_element_multiplicity = 1

    def __init__(self, env, rank):
        self.rank = rank
        self.legendre_coefficients = self.compute_legendre_matrix(self.rank) # (1, npow, npow)
        self.powers_needed = np.arange(self.rank + 1, dtype=np.float32) # (npow, )
        super().__init__(env, rank)

    def compute_legendre_matrix(self, rank):
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
        # TODO: convert this N-D torch.gather to numpy
        accumulator = torch.gather(
            torch.as_tensor(legendre_1d_eval, dtype=torch.float),
            dim=-1,
            index=torch.as_tensor(indices, dtype=torch.long),
        ).detach().numpy() # (B, n_dim, n_feat)
        features = np.prod(accumulator, axis=-2).astype(np.float32) # (B, n_feat)
        return features

class NormalizedRBFBasis(BasisWrapper):
    pass
