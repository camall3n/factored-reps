import matplotlib.pyplot as plt
import numpy as np

from factored_rl.wrappers import PolynomialBasisWrapper as Polynomial
from factored_rl.wrappers import LegendreBasisWrapper as Legendre
from factored_rl.wrappers import FourierBasisWrapper as Fourier
from visgrid.envs.point import BoundedPointEnv

def get_curves(basis, ndim=1, rank=1, n_points=1000):
    action = 2 * np.ones(ndim, dtype=np.float32) / n_points

    line_env = BoundedPointEnv(ndim)
    poly_env = basis(line_env, rank)

    def get_points(env):
        env.reset(x=(-1.0 * np.ones(ndim)))
        obs = []
        for _ in range(n_points + 1):
            obs.append(env.step(action)[0])
        obs = np.stack(obs)
        return obs

    line_points = get_points(line_env)
    poly_points = get_points(poly_env)

    return line_points, poly_points

def visualize_points(basis, rank=1):
    line_points, poly_points = get_curves(basis=basis, rank=rank)
    fig, ax = plt.subplots()
    orders = poly_points.shape[-1]
    legend_columns = basis.basis_element_multiplicity
    if legend_columns == 2:
        labels = [f'sin({i})' for i in range(orders // 2)]
        labels += [f'cos({i})' for i in range(orders // 2)]
    else:
        labels = [str(x) for x in np.arange(orders)]
    ax.plot(line_points, poly_points, label=labels)
    ax.set_xlabel('x')
    ax.set_ylabel('features')
    ax.legend(ncol=legend_columns, loc='lower right')
    ax.set_title(f'{basis.__name__}')
    plt.show()

def get_output_shape(basis, ndim, rank):
    return get_curves(basis=basis, ndim=ndim, rank=rank)[-1].shape[-1]

def test_polynomial_1d():
    assert get_output_shape(basis=Polynomial, ndim=1, rank=0) == 1 # 0
    assert get_output_shape(basis=Polynomial, ndim=1, rank=1) == 2 # 0, x
    assert get_output_shape(basis=Polynomial, ndim=1, rank=2) == 3 # 0, x, x^2
    assert get_output_shape(basis=Polynomial, ndim=1, rank=3) == 4 #  0, x, x^2, x^3

def test_polynomial_2d():
    assert get_output_shape(basis=Polynomial, ndim=2, rank=0) == 1 # 0
    assert get_output_shape(basis=Polynomial, ndim=2, rank=1) == 3 # 0, x, y
    assert get_output_shape(basis=Polynomial, ndim=2, rank=2) == 6 # 0, x, y, x^2, y^2, xy
    assert get_output_shape(basis=Polynomial, ndim=2, rank=3) == 10
    # 0, x, y, x^2, y^2, x^3, y^3, xy, xy^2, yx^2

def test_polynomial_3d():
    assert get_output_shape(basis=Polynomial, ndim=3, rank=1) == 4 # 0, x, y, z
    assert get_output_shape(basis=Polynomial, ndim=3, rank=2) == 10
    # 0, x, y, z, x^2, y^2, z^2, xy, yz, xz
    assert get_output_shape(basis=Polynomial, ndim=3, rank=3) == 20
    # 0, x, y, z, x^2, y^2, z^2, x^3, y^3, z^3
    # xy, yz, xz, xyz, xy^2, xz^2, yx^2, yz^2, zx^2, zy^2

def test_legendre_1d():
    assert get_output_shape(basis=Legendre, ndim=1, rank=0) == 1
    assert get_output_shape(basis=Legendre, ndim=1, rank=1) == 2
    assert get_output_shape(basis=Legendre, ndim=1, rank=2) == 3
    assert get_output_shape(basis=Legendre, ndim=1, rank=3) == 4

def test_legendre_2d():
    # same shapes as polynomial
    assert get_output_shape(basis=Legendre, ndim=2, rank=0) == 1
    assert get_output_shape(basis=Legendre, ndim=2, rank=1) == 3
    assert get_output_shape(basis=Legendre, ndim=2, rank=2) == 6
    assert get_output_shape(basis=Legendre, ndim=2, rank=3) == 10

def test_legendre_3d():
    assert get_output_shape(basis=Legendre, ndim=3, rank=1) == 4
    assert get_output_shape(basis=Legendre, ndim=3, rank=2) == 10
    assert get_output_shape(basis=Legendre, ndim=3, rank=3) == 20

def test_fourier_1d():
    # twice as many features as polynomial
    assert get_output_shape(basis=Fourier, ndim=1, rank=0) == 2
    assert get_output_shape(basis=Fourier, ndim=1, rank=1) == 4
    assert get_output_shape(basis=Fourier, ndim=1, rank=2) == 6
    assert get_output_shape(basis=Fourier, ndim=1, rank=3) == 8

def test_fourier_2d():
    # twice as many features as polynomial
    assert get_output_shape(basis=Fourier, ndim=2, rank=0) == 2
    assert get_output_shape(basis=Fourier, ndim=2, rank=1) == 6
    assert get_output_shape(basis=Fourier, ndim=2, rank=2) == 12
    assert get_output_shape(basis=Fourier, ndim=2, rank=3) == 20

def test_fourier_3d():
    assert get_output_shape(basis=Fourier, ndim=3, rank=1) == 8
    assert get_output_shape(basis=Fourier, ndim=3, rank=2) == 20
    assert get_output_shape(basis=Fourier, ndim=3, rank=3) == 40

def test_4d():
    assert get_output_shape(basis=Polynomial, ndim=4, rank=2) == 15
    # 0, w, x, y, z, w^2, x^2, y^2, z^2, wx, wy, wz, xy, xz, yz
    assert get_output_shape(basis=Legendre, ndim=4, rank=2) == 15
    assert get_output_shape(basis=Fourier, ndim=4, rank=2) == 30

if __name__ == '__main__':
    visualize_points(basis=Polynomial, rank=5)
    visualize_points(basis=Legendre, rank=5)
    visualize_points(basis=Fourier, rank=3)
