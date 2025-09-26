import pytest
pytest_param = pytest.mark.parametrize

from numpy import empty, full, linspace
from numpy.random import default_rng
from numpy.testing import assert_allclose

from pyregress import PolySet


def gold_standard_bases(x, p_type, grad=False):
    n_pts, n_xdims = x.shape
    if n_xdims == 1:
        n_bases = 8
        x = x.reshape(-1)
        p = empty((n_bases, n_pts))
        if grad:
            gp = empty((n_bases, n_pts))
        if p_type == 'power':
            p[0] = full(n_pts, 1, dtype='float64')
            p[1] = x
            p[2] = x**2
            p[3] = x**3
            p[4] = x**4
            p[5] = x**5
            p[6] = x**6
            p[7] = x**7
            if grad:
                gp[0] = full(n_pts, 0, dtype='float64')
                gp[1] = full(n_pts, 1, dtype='float64')
                gp[2] = 2 * x
                gp[3] = 3 * x**2
                gp[4] = 4 * x**3
                gp[5] = 5 * x**4
                gp[6] = 6 * x**5
                gp[7] = 7 * x**6
        elif p_type == 'Legendre':
            p[0] = full(n_pts, 1, dtype='float64')
            p[1] = x
            p[2] = (3 * x**2 - 1) / 2
            p[3] = (5 * x**3 - 3 * x) / 2
            p[4] = (35 * x**4 - 30 * x**2 + 3) / 8
            p[5] = (63 * x**5 - 70 * x**3 + 15 * x) / 8
            p[6] = (231 * x**6 - 315 * x**4 + 105 * x**2 - 5) / 16
            p[7] = (429 * x**7 - 693 * x**5 + 315 * x**3 - 35 * x) / 16
            if grad:
                gp[0] = full(n_pts, 0, dtype='float64')
                gp[1] = full(n_pts, 1, dtype='float64')
                gp[2] = (6 * x) / 2
                gp[3] = (15 * x**2 - 3) / 2
                gp[4] = (140 * x**3 - 60 * x) / 8
                gp[5] = (315 * x**4 - 210 * x**2 + 15) / 8
                gp[6] = (1386 * x**5 - 1260 * x**3 + 210 * x) / 16
                gp[7] = (3003 * x**6 - 3465 * x**4 + 945 * x**2 - 35) / 16
        elif p_type ==  'Chebyshev':
            p[0] = full(n_pts, 1, dtype='float64')
            p[1] = x
            p[2] = 2 * x**2 - 1
            p[3] = 4 * x**3 - 3 * x
            p[4] = 8 * x**4 - 8 * x**2 + 1
            p[5] = 16 * x**5 - 20 * x**3 + 5 * x
            p[6] = 32 * x**6 - 48 * x**4 + 18 * x**2 - 1
            p[7] = 64 * x**7 - 112 * x**5 + 56 * x**3 - 7 * x
            if grad:
                gp[0] = full(n_pts, 0, dtype='float64')
                gp[1] = full(n_pts, 1, dtype='float64')
                gp[2] = 4 * x
                gp[3] = 12 * x**2 - 3
                gp[4] = 32 * x**3 - 16 * x
                gp[5] = 80 * x**4 - 60 * x**2 + 5
                gp[6] = 192 * x**5 - 192 * x**3 + 36 * x
                gp[7] = 448 * x**6 - 560 * x**4 + 168 * x**2 - 7
        elif p_type == 'Laguerre':
            p[0] = full(n_pts, 1, dtype='float64')
            p[1] = -x + 1
            p[2] = (x**2 - 4 * x + 2) / 2
            p[3] = (-x**3 + 9 * x**2 - 18 * x + 6) / 6
            p[4] = (x**4 - 16 * x**3 + 72 * x**2 - 96 * x + 24) / 24
            p[5] = (-x**5 + 25 * x**4 - 200 * x**3 + 600 * x**2 - 600 * x + 120) / 120
            p[6] = (x**6 - 36 * x**5 + 450 * x**4 - 2400 * x**3 + 5400 * x**2 - 4320 * x + 720) / 720
            p[7] = (-x**7 + 49 * x**6 - 882 * x**5 + 7350 * x**4 - 29400 * x**3 + 52920 * x**2 - 35280 * x + 5040) / 5040
            if grad:
                gp[0] = full(n_pts, 0, dtype='float64')
                gp[1] = full(n_pts, -1, dtype='float64')
                gp[2] = (2 * x - 4) / 2
                gp[3] = (-3 * x**2 + 18 * x - 18) / 6
                gp[4] = (4 * x**3 - 48 * x**2 + 144 * x - 96) / 24
                gp[5] = (-5 * x**4 + 100 * x**3 - 600 * x**2 + 1200 * x - 600) / 120
                gp[6] = (6 * x**5 - 180 * x**4 + 1800 * x**3 - 7200 * x**2 + 10800 * x - 4320) / 720
                gp[7] = (-7 * x**6 + 294 * x**5 - 4410 * x**4 + 29400 * x**3 - 88200 * x**2 + 105840 * x - 35280) / 5040
        elif p_type == 'Hermite':
            p[0] = full(n_pts, 1, dtype='float64')
            p[1] = x
            p[2] = x**2 - 1
            p[3] = x**3 - 3 * x
            p[4] = x**4 - 6 * x**2 + 3
            p[5] = x**5 - 10 * x**3 + 15 * x
            p[6] = x**6 - 15 * x**4 + 45 * x**2 - 15
            p[7] = x**7 - 21 * x**5 + 105 * x**3 - 105 * x
            if grad:
                gp[0] = full(n_pts, 0, dtype='float64')
                gp[1] = full(n_pts, 1, dtype='float64')
                gp[2] = 2 * x
                gp[3] = 3 * x**2 - 3
                gp[4] = 4 * x**3 - 12 * x
                gp[5] = 5 * x**4 - 30 * x**2 + 15
                gp[6] = 6 * x**5 - 60 * x**3 + 90 * x
                gp[7] = 7 * x**6 - 105 * x**4 + 315 * x**2 - 105
        if not grad:
            return p.T
        else:
            return p.T, gp.T.reshape([n_pts, n_bases, -1])
    elif n_xdims == 2:
        max_order = 3
        n_bases = 10
        p = empty((n_pts, n_bases))
        if grad:
            gp = full((n_pts, n_bases, n_xdims), 0, dtype='float64')
        if p_type == 'power':
            p[:, 0] = full(n_pts, 1, dtype='float64')
            p[:, 1] = x[:, 0]
            p[:, 2] = x[:, 1]
            p[:, 3] = x[:, 0]**2
            p[:, 4] = x[:, 0] * x[:, 1]
            p[:, 5] = x[:, 1]**2
            p[:, 6] = x[:, 0]**3
            p[:, 7] = x[:, 0]**2 * x[:, 1]
            p[:, 8] = x[:, 0] * x[:, 1]**2
            p[:, 9] = x[:, 1]**3
            if grad:
                gp[:, 1, 0] = 1
                gp[:, 2, 1] = 1
                gp[:, 3, 0] = 2 * x[:, 0]
                gp[:, 4, 0] = x[:, 1]
                gp[:, 4, 1] = x[:, 0]
                gp[:, 5, 1] = 2 * x[:, 1]
                gp[:, 6, 0] = 3 * x[:, 0]**2
                gp[:, 7, 0] = 2 * x[:, 0] * x[:, 1]
                gp[:, 7, 1] = x[:, 0]**2
                gp[:, 8, 0] = x[:, 1]**2
                gp[:, 8, 1] = x[:, 0] * 2 * x[:, 1]
                gp[:, 9, 1] = 3 * x[:, 1]**2
        elif p_type == 'Legendre':
            p[:, 0] = full(n_pts, 1, dtype='float64')
            p[:, 1] = x[:, 0]
            p[:, 2] = x[:, 1]
            p[:, 3] = (3 * x[:, 0]**2 - 1) / 2
            p[:, 4] = x[:, 0] * x[:, 1]
            p[:, 5] = (3 * x[:, 1]**2 - 1) / 2
            p[:, 6] = (5 * x[:, 0]**3 - 3 * x[:, 0]) / 2
            p[:, 7] = (3 * x[:, 0]**2 - 1) / 2 * x[:, 1]
            p[:, 8] = x[:, 0] * (3 * x[:, 1]**2 - 1) / 2
            p[:, 9] = (5 * x[:, 1]**3 - 3 * x[:, 1]) / 2
            if grad:
                gp[:, 1, 0] = 1
                gp[:, 2, 1] = 1
                gp[:, 3, 0] = (6 * x[:, 0]) / 2
                gp[:, 4, 0] = x[:, 1]
                gp[:, 4, 1] = x[:, 0]
                gp[:, 5, 1] = (6 * x[:, 1]) / 2
                gp[:, 6, 0] = (15 * x[:, 0]**2 - 3) / 2
                gp[:, 7, 0] = (6 * x[:, 0]) / 2 * x[:, 1]
                gp[:, 7, 1] = (3 * x[:, 0]**2 - 1) / 2
                gp[:, 8, 0] = (3 * x[:, 1]**2 - 1) / 2
                gp[:, 8, 1] = x[:, 0] * (6 * x[:, 1]) / 2
                gp[:, 9, 1] = (15 * x[:, 1]**2 - 3) / 2
        elif p_type == 'Chebyshev':
            p[:, 0] = full(n_pts, 1, dtype='float64')
            p[:, 1] = x[:, 0]
            p[:, 2] = x[:, 1]
            p[:, 3] = (2 * x[:, 0]**2 - 1)
            p[:, 4] = x[:, 0] * x[:, 1]
            p[:, 5] = (2 * x[:, 1]**2 - 1)
            p[:, 6] = (4 * x[:, 0]**3 - 3 * x[:, 0])
            p[:, 7] = (2 * x[:, 0]**2 - 1) * x[:, 1]
            p[:, 8] = x[:, 0] * (2 * x[:, 1]**2 - 1)
            p[:, 9] = (4 * x[:, 1]**3 - 3 * x[:, 1])
            if grad:
                gp[:, 1, 0] = 1
                gp[:, 2, 1] = 1
                gp[:, 3, 0] = 4 * x[:, 0]
                gp[:, 4, 0] = x[:, 1]
                gp[:, 4, 1] = x[:, 0]
                gp[:, 5, 1] = 4 * x[:, 1]
                gp[:, 6, 0] = 12 * x[:, 0]**2 - 3
                gp[:, 7, 0] = 4 * x[:, 0] * x[:, 1]
                gp[:, 7, 1] = (2 * x[:, 0]**2 - 1)
                gp[:, 8, 0] = (2 * x[:, 1]**2 - 1)
                gp[:, 8, 1] = x[:, 0] * 4 * x[:, 1]
                gp[:, 9, 1] = 12 * x[:, 1]**2 - 3
        elif p_type == 'Laguerre':
            p[:, 0] = full(n_pts, 1, dtype='float64')
            p[:, 1] = -x[:, 0] + 1
            p[:, 2] = -x[:, 1] + 1
            p[:, 3] = (x[:, 0]**2 - 4 * x[:, 0] + 2) / 2
            p[:, 4] = (-x[:, 0] + 1) * (-x[:, 1] + 1)
            p[:, 5] = (x[:, 1]**2 - 4 * x[:, 1] + 2) / 2
            p[:, 6] = (-x[:, 0]**3 + 9 * x[:, 0]**2 - 18 * x[:, 0] + 6) / 6
            p[:, 7] = (x[:, 0]**2 - 4 * x[:, 0] + 2) / 2 * (-x[:, 1] + 1)
            p[:, 8] = (-x[:, 0] + 1) * (x[:, 1]**2 - 4 * x[:, 1] + 2) / 2
            p[:, 9] = (-x[:, 1]**3 + 9 * x[:, 1]**2 - 18 * x[:, 1] + 6) / 6
            if grad:
                gp[:, 1, 0] = -1
                gp[:, 2, 1] = -1
                gp[:, 3, 0] = (2 * x[:, 0] - 4) / 2
                gp[:, 4, 0] = (x[:, 1] - 1)
                gp[:, 4, 1] = (x[:, 0] - 1)
                gp[:, 5, 1] = (2 * x[:, 1] - 4) / 2
                gp[:, 6, 0] = (-3 * x[:, 0]**2 + 18 * x[:, 0] - 18) / 6
                gp[:, 7, 0] = (2 * x[:, 0] - 4) / 2 * (-x[:, 1] + 1)
                gp[:, 7, 1] = -(x[:, 0]**2 - 4 * x[:, 0] + 2) / 2
                gp[:, 8, 0] = -(x[:, 1]**2 - 4 * x[:, 1] + 2) / 2
                gp[:, 8, 1] = (-x[:, 0] + 1) * (2 * x[:, 1] - 4) / 2
                gp[:, 9, 1] = (-3 * x[:, 1]**2 + 18 * x[:, 1] - 18) / 6
        elif p_type == 'Hermite':
            p[:, 0] = full(n_pts, 1, dtype='float64')
            p[:, 1] = x[:, 0]
            p[:, 2] = x[:, 1]
            p[:, 3] = (x[:, 0]**2 - 1)
            p[:, 4] = x[:, 0] * x[:, 1]
            p[:, 5] = (x[:, 1]**2 - 1)
            p[:, 6] = (x[:, 0]**3 - 3 * x[:, 0])
            p[:, 7] = (x[:, 0]**2 - 1) * x[:, 1]
            p[:, 8] = x[:, 0] * (x[:, 1]**2 - 1)
            p[:, 9] = (x[:, 1]**3 - 3 * x[:, 1])
            if grad:
                gp[:, 1, 0] = 1
                gp[:, 2, 1] = 1
                gp[:, 3, 0] = 2 * x[:, 0]
                gp[:, 4, 0] = x[:, 1]
                gp[:, 4, 1] = x[:, 0]
                gp[:, 5, 1] = 2 * x[:, 1]
                gp[:, 6, 0] = 3 * x[:, 0]**2 - 3
                gp[:, 7, 0] = 2 * x[:, 0] * x[:, 1]
                gp[:, 7, 1] = (x[:, 0]**2 - 1)
                gp[:, 8, 0] = (x[:, 1]**2 - 1)
                gp[:, 8, 1] = x[:, 0] * 2 * x[:, 1]
                gp[:, 9, 1] = 3 * x[:, 1]**2 - 3
        if not grad:
            return p
        else:
            return p, gp
    elif n_xdims == 3:
        max_order = 2
        n_bases = 10
        p = empty((n_pts, n_bases))
        if grad:
            gp = full((n_pts, n_bases, n_xdims), 0, dtype='float64')
        if p_type == 'power':
            p[:, 0] = full(n_pts, 1, dtype='float64')
            p[:, 1] = x[:, 0]
            p[:, 2] = x[:, 1]
            p[:, 3] = x[:, 2]
            p[:, 4] = x[:, 0]**2
            p[:, 5] = x[:, 0] * x[:, 1]
            p[:, 6] = x[:, 0] * x[:, 2]
            p[:, 7] = x[:, 1]**2
            p[:, 8] = x[:, 1] * x[:, 2]
            p[:, 9] = x[:, 2]**2
            if grad:
                gp[:, 1, 0] = 1
                gp[:, 2, 1] = 1
                gp[:, 3, 2] = 1
                gp[:, 4, 0] = 2 * x[:, 0]
                gp[:, 5, 0] = x[:, 1]
                gp[:, 5, 1] = x[:, 0]
                gp[:, 6, 0] = x[:, 2]
                gp[:, 6, 2] = x[:, 0]
                gp[:, 7, 1] = 2 * x[:, 1]
                gp[:, 8, 1] = x[:, 2]
                gp[:, 8, 2] = x[:, 1]
                gp[:, 9, 2] = 2 * x[:, 2]
        elif p_type == 'Legendre':
            p[:, 0] = full(n_pts, 1, dtype='float64')
            p[:, 1] = x[:, 0]
            p[:, 2] = x[:, 1]
            p[:, 3] = x[:, 2]
            p[:, 4] = (3 * x[:, 0]**2 - 1) / 2
            p[:, 5] = x[:, 0] * x[:, 1]
            p[:, 6] = x[:, 0] * x[:, 2]
            p[:, 7] = (3 * x[:, 1]**2 - 1) / 2
            p[:, 8] = x[:, 1] * x[:, 2]
            p[:, 9] = (3 * x[:, 2]**2 - 1) / 2
            if grad:
                gp[:, 1, 0] = 1
                gp[:, 2, 1] = 1
                gp[:, 3, 2] = 1
                gp[:, 4, 0] = (6 * x[:, 0]) / 2
                gp[:, 5, 0] = x[:, 1]
                gp[:, 5, 1] = x[:, 0]
                gp[:, 6, 0] = x[:, 2]
                gp[:, 6, 2] = x[:, 0]
                gp[:, 7, 1] = (6 * x[:, 1]) / 2
                gp[:, 8, 1] = x[:, 2]
                gp[:, 8, 2] = x[:, 1]
                gp[:, 9, 2] = (6 * x[:, 2]) / 2
        elif p_type == 'Chebyshev':
            p[:, 0] = full(n_pts, 1, dtype='float64')
            p[:, 1] = x[:, 0]
            p[:, 2] = x[:, 1]
            p[:, 3] = x[:, 2]
            p[:, 4] = 2 * x[:, 0]**2 - 1
            p[:, 5] = x[:, 0] * x[:, 1]
            p[:, 6] = x[:, 0] * x[:, 2]
            p[:, 7] = 2 * x[:, 1]**2 - 1
            p[:, 8] = x[:, 1] * x[:, 2]
            p[:, 9] = 2 * x[:, 2]**2 - 1
            if grad:
                gp[:, 1, 0] = 1
                gp[:, 2, 1] = 1
                gp[:, 3, 2] = 1
                gp[:, 4, 0] = 4 * x[:, 0]
                gp[:, 5, 0] = x[:, 1]
                gp[:, 5, 1] = x[:, 0]
                gp[:, 6, 0] = x[:, 2]
                gp[:, 6, 2] = x[:, 0]
                gp[:, 7, 1] = 4 * x[:, 1]
                gp[:, 8, 1] = x[:, 2]
                gp[:, 8, 2] = x[:, 1]
                gp[:, 9, 2] = 4 * x[:, 2]
        elif p_type == 'Laguerre':
            p[:, 0] = full(n_pts, 1, dtype='float64')
            p[:, 1] = -x[:, 0] + 1
            p[:, 2] = -x[:, 1] + 1
            p[:, 3] = -x[:, 2] + 1
            p[:, 4] = (x[:, 0]**2 - 4 * x[:, 0] + 2) / 2
            p[:, 5] = (-x[:, 0] + 1) * (-x[:, 1] + 1)
            p[:, 6] = (-x[:, 0] + 1) * (-x[:, 2] + 1)
            p[:, 7] = (x[:, 1]**2 - 4 * x[:, 1] + 2) / 2
            p[:, 8] = (-x[:, 1] + 1) * (-x[:, 2] + 1)
            p[:, 9] = (x[:, 2]**2 - 4 * x[:, 2] + 2) / 2
            if grad:
                gp[:, 1, 0] = -1
                gp[:, 2, 1] = -1
                gp[:, 3, 2] = -1
                gp[:, 4, 0] = (2 * x[:, 0] - 4) / 2
                gp[:, 5, 0] = x[:, 1] - 1
                gp[:, 5, 1] = x[:, 0] - 1
                gp[:, 6, 0] = x[:, 2] - 1
                gp[:, 6, 2] = x[:, 0] - 1
                gp[:, 7, 1] = (2 * x[:, 1] - 4) / 2
                gp[:, 8, 1] = x[:, 2] - 1
                gp[:, 8, 2] = x[:, 1] - 1
                gp[:, 9, 2] = (2 * x[:, 2] - 4) / 2
        elif p_type == 'Hermite':
            p[:, 0] = full(n_pts, 1, dtype='float64')
            p[:, 1] = x[:, 0]
            p[:, 2] = x[:, 1]
            p[:, 3] = x[:, 2]
            p[:, 4] = x[:, 0]**2 - 1
            p[:, 5] = x[:, 0] * x[:, 1]
            p[:, 6] = x[:, 0] * x[:, 2]
            p[:, 7] = x[:, 1]**2 - 1
            p[:, 8] = x[:, 1] * x[:, 2]
            p[:, 9] = x[:, 2]**2 - 1
            if grad:
                gp[:, 1, 0] = 1
                gp[:, 2, 1] = 1
                gp[:, 3, 2] = 1
                gp[:, 4, 0] = 2 * x[:, 0]
                gp[:, 5, 0] = x[:, 1]
                gp[:, 5, 1] = x[:, 0]
                gp[:, 6, 0] = x[:, 2]
                gp[:, 6, 2] = x[:, 0]
                gp[:, 7, 1] = 2 * x[:, 1]
                gp[:, 8, 1] = x[:, 2]
                gp[:, 8, 2] = x[:, 1]
                gp[:, 9, 2] = 2 * x[:, 2]
        if not grad:
            return p
        else:
            return p, gp


@pytest_param("dimensionality", [1, 2, 3])
@pytest_param("poly_type", ['power', 'Legendre', 'Chebyshev', 'Laguerre', 'Hermite'])
@pytest_param("gradient", [False, True])
def test_lin_regress(dimensionality, poly_type, gradient):
    my_rng = default_rng(seed=42)
    rand = my_rng.random
    if dimensionality == 1:
        x = linspace(-1, 1, 200).reshape((-1, 1))
        poly_order = 7
    elif dimensionality == 2:
        x = 2 * rand((4, dimensionality)) - 1
        poly_order = 3
    elif dimensionality == 3:
        x = 2 * rand((30, dimensionality)) - 1
        poly_order = 2
    my_polyset = PolySet(dimensionality, poly_order, ptype=poly_type, x_range=None)
    if not gradient:
        p = my_polyset(x)
        print(f'{p.shape = }')
        print(p)
        p_gs = gold_standard_bases(x, poly_type)
        print(p_gs)
    else:
        p, gp = my_polyset(x, gradient)
        print(f'{p.shape = }, {gp.shape = }')
        p_gs, gp_gs = gold_standard_bases(x, poly_type, gradient)
        print(gp, gp_gs)
    assert_allclose(p, p_gs)
    if gradient:
        assert_allclose(gp, gp_gs)