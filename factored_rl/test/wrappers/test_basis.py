import pytest

import matplotlib.pyplot as plt
import numpy as np

from factored_rl.wrappers.basis import BasisWrapper, IdentityBasisFunction, PolynomialBasisFunction
from factored_rl.wrappers import PolynomialBasisWrapper as PolynomialWrapper
from factored_rl.wrappers import LegendreBasisWrapper as LegendreWrapper
from factored_rl.wrappers import FourierBasisWrapper as FourierWrapper
from visgrid.envs.point import BoundedPointEnv

def get_curves(wrapper: BasisWrapper, ndim: int = 1, rank: int = 1, n_segments: int = 1000):
    action = 2 * np.ones(ndim, dtype=np.float32) / n_segments

    line_env = BoundedPointEnv(ndim)
    poly_env = wrapper(line_env, rank)

    def get_points(env):
        env.reset(x=(-1.0 * np.ones(ndim)))
        obs = []
        for _ in range(n_segments + 1):
            obs.append(env.step(action)[0])
        obs = np.stack(obs)
        return obs

    line_points = get_points(line_env)
    poly_points = get_points(poly_env)

    return line_points, poly_points

def visualize_points(wrapper: BasisWrapper, rank: int = 1):
    line_points, poly_points = get_curves(wrapper=wrapper, rank=rank)
    fig, ax = plt.subplots()
    orders = poly_points.shape[-1]
    basis_name = wrapper.__name__.replace('Wrapper', '')
    legend_columns = 2 if basis_name == 'FourierBasis' else 1
    if legend_columns == 2:
        labels = [f'sin({i})' for i in range(orders // 2)]
        labels += [f'cos({i})' for i in range(orders // 2)]
    else:
        labels = [str(x) for x in np.arange(orders)]
    ax.plot(line_points, poly_points, label=labels)
    ax.set_xlabel('x')
    ax.set_ylabel('features')
    ax.legend(ncol=legend_columns, loc='lower right')
    ax.set_title(basis_name)
    plt.show()

def get_output_shape(wrapper: BasisWrapper, ndim: int, rank: int):
    return get_curves(wrapper=wrapper, ndim=ndim, rank=rank)[-1].shape[-1]

@pytest.mark.parametrize("ndim,rank", [(n, r) for n in range(1, 5) for r in range(4)])
def test_basic_shapes(ndim, rank):
    basis_fn = PolynomialBasisFunction(ndim=ndim, rank=rank)
    env = BoundedPointEnv(ndim)
    basis_wrapper = PolynomialWrapper(env, rank)
    assert basis_fn.n_features == basis_wrapper.observation_space.shape[0]
    assert get_output_shape(wrapper=PolynomialWrapper, ndim=ndim, rank=rank)

def test_errors():
    with pytest.raises(ValueError):
        PolynomialBasisFunction(ndim=0, rank=1)
    with pytest.raises(ValueError):
        PolynomialBasisFunction(ndim=1, rank=-1)

@pytest.mark.parametrize("ndim", range(1, 5))
def test_identity_basis_fn(ndim):
    basis_fn = IdentityBasisFunction(ndim=ndim)
    assert basis_fn.n_features == ndim

def test_polynomial_1d():
    assert get_output_shape(wrapper=PolynomialWrapper, ndim=1, rank=0) == 1 # 0
    assert get_output_shape(wrapper=PolynomialWrapper, ndim=1, rank=1) == 2 # 0, x
    assert get_output_shape(wrapper=PolynomialWrapper, ndim=1, rank=2) == 3 # 0, x, x^2
    assert get_output_shape(wrapper=PolynomialWrapper, ndim=1, rank=3) == 4 # 0, x, x^2, x^3

def test_polynomial_2d():
    assert get_output_shape(wrapper=PolynomialWrapper, ndim=2, rank=0) == 1 # 0
    assert get_output_shape(wrapper=PolynomialWrapper, ndim=2, rank=1) == 3 # 0, x, y
    assert get_output_shape(wrapper=PolynomialWrapper, ndim=2, rank=2) == 6 # 0, x, y, x^2, y^2, xy
    assert get_output_shape(wrapper=PolynomialWrapper, ndim=2, rank=3) == 10
    # 0, x, y, x^2, y^2, x^3, y^3, xy, xy^2, yx^2

def test_polynomial_3d():
    assert get_output_shape(wrapper=PolynomialWrapper, ndim=3, rank=1) == 4 # 0, x, y, z
    assert get_output_shape(wrapper=PolynomialWrapper, ndim=3, rank=2) == 10
    # 0, x, y, z, x^2, y^2, z^2, xy, yz, xz
    assert get_output_shape(wrapper=PolynomialWrapper, ndim=3, rank=3) == 20
    # 0, x, y, z, x^2, y^2, z^2, x^3, y^3, z^3
    # xy, yz, xz, xyz, xy^2, xz^2, yx^2, yz^2, zx^2, zy^2

@pytest.mark.parametrize('rank, n_feats', zip([0, 1, 2, 3], [1, 2, 3, 4]))
def test_legendre_1d(rank, n_feats):
    # same shapes as polynomial
    assert get_output_shape(wrapper=LegendreWrapper, ndim=1, rank=rank) == n_feats

@pytest.mark.parametrize('rank, n_feats', zip([0, 1, 2, 3], [1, 3, 6, 10]))
def test_legendre_2d(rank, n_feats):
    assert get_output_shape(wrapper=LegendreWrapper, ndim=2, rank=rank) == n_feats

@pytest.mark.parametrize('rank, n_feats', zip([0, 1, 2, 3], [1, 4, 10, 20]))
def test_legendre_3d(rank, n_feats):
    assert get_output_shape(wrapper=LegendreWrapper, ndim=3, rank=rank) == n_feats

@pytest.mark.parametrize('rank, n_feats', zip([0, 1, 2, 3], [2, 4, 6, 8]))
def test_fourier_1d(rank, n_feats):
    # twice as many features as polynomial
    assert get_output_shape(wrapper=FourierWrapper, ndim=1, rank=rank) == n_feats

@pytest.mark.parametrize('rank, n_feats', zip([0, 1, 2, 3], [2, 6, 12, 20]))
def test_fourier_2d(rank, n_feats):
    assert get_output_shape(wrapper=FourierWrapper, ndim=2, rank=rank) == n_feats

@pytest.mark.parametrize('rank, n_feats', zip([0, 1, 2, 3], [2, 8, 20, 40]))
def test_fourier_3d(rank, n_feats):
    assert get_output_shape(wrapper=FourierWrapper, ndim=3, rank=rank) == n_feats

def test_4d():
    assert get_output_shape(wrapper=PolynomialWrapper, ndim=4, rank=2) == 15
    # 0, w, x, y, z, w^2, x^2, y^2, z^2, wx, wy, wz, xy, xz, yz
    assert get_output_shape(wrapper=LegendreWrapper, ndim=4, rank=2) == 15
    assert get_output_shape(wrapper=FourierWrapper, ndim=4, rank=2) == 30

if __name__ == '__main__':
    visualize_points(wrapper=PolynomialWrapper, rank=5)
    visualize_points(wrapper=LegendreWrapper, rank=5)
    visualize_points(wrapper=FourierWrapper, rank=3)
