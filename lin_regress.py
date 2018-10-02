r"""
Create a module to support general multiple ordinary linear regression
(here, linear is does not refer to a restriction on the behavior of the
resulting function, but linear in the regression coefficients; ordinary
refers to the use of a least-squares objective; multiple refers to
the handling of multiple x dimensions; and general refers to the optional
use of a weight to specify the ratio of squared errors for each data point).

To clarify notation, the regression is written as
:math:`$y(x) = H(x) \times \theta$`, where :math:`$\theta$` is a vector
of linear coefficients and :math:`$H(x)$` is a matrix of basis functions
for the variates, :math:`$x$`.
The least-squares solution for the coefficients is
:math:`\theta = (H^T(X_d) \times W \times H(X_d))^{-1}
H^T(X_d) \times W \times Y_d`.

This module provides the polynomial bases up to order six (but they are
easily extended). Alternative bases can also be created by the user. In
one dimension (single variate), one could simply use numpy's mathematical
functions as bases, e.g. bases=[const, one, sin, log]. In higher dims.
one can combine these using lambda, e.g. f = lambda x: x[0]**2 * sin(x[1]),
and then combine f with other bases: bases=[const, one, f].

Created Feb. 2018 @author: Sean T. Smith
"""
from numpy import empty, ones, arange, concatenate, atleast_2d, sum, prod
from numpy.linalg import solve


class OrdLinRegress:
    """
    Create a class to perform the ordinary linear regression.
    Instantiating an object results in a least-squares calculation for the
    coefficients. And calling the object will evaluate the regression.
    """
    def __init__(self, Xd, Yd, bases, W=None):
        """
        Arguments:
            Xd:  array-2D,
                location in x of the data for each point,
                shape: (# of points/observations, # of x dims.).
            Yd:  array-1D,
                values of the data at points corresponding to 1st dim. of Xd.
            bases:  list of callables,
                list of basis functions by group, when called (with argument X)
                each element of this list must return basis functions as an
                array with shape: (# of points in X, # of basis functions).
            W:  array-1D (optional),
                weights corresponding to the ratio of squared errors for each
                data point. If omitted, all values are set to unity.
        """
        n, nx = Xd.shape
        if not W:
            W = ones(n)
        self.bases = bases
        Hd = concatenate([f(Xd) for f in self.bases], axis=1)
        # Least-squares solution:
        self.Σinv = (Hd.T * W) @ Hd
        self.θ = solve(self.Σinv, Hd.T @ (W * Yd))
        sq_err = sum((Yd - self(Xd))**2 / W)
        self.σ2_mean = sq_err / (n - nx - 2)
        self.σ2_mode = sq_err / (n - nx + 2)

    def __call__(self, X):
        """
        Arguments:
            X:  array-2D,
                locations in x where to evaluate the regression,
                shape: (# of x dims., # of points).
        Returns:
            Y:  array-1D,
                regression response evaluated at each point in X.
        """
        H = concatenate([f(X) for f in self.bases], axis=1)
        Y = H @ self.θ
        return Y

    def regression_error(self, X):
        """
        Arguments:
            X:  array-2D,
                locations in x where to evaluate the regression,
                shape: (# of x dims., # of points).
        Returns:
            σy2:  array-1D,
                variance of the regression uncertainty at each point in X.
        """
        H = concatenate([f(X) for f in self.bases], axis=1)
        σy2 = sum(H * solve(self.Σinv, H.T).T, axis=1)
        return σy2


def const(x):
    x = atleast_2d(x)  # in case x is a scalar, assume multiple points in 1D
    return ones((x.shape[0], 1))

def one(x):
    x = atleast_2d(x)  # in case x is a scalar, assume multiple points in 1D
    return x

def two(x):
    x = atleast_2d(x)  # in case x is a scalar, assume multiple points in 1D
    n, nx = x.shape
    nH = int((nx * (nx + 1)) / 2)
    H = empty((n, nH))
    iH = 0
    # Loop through the unique combinations of x[i] * x[j]:
    for i in range(nx):
        for j in range(i, nx):
            # Evaluate the quadratic:
            H[:, iH] = x[:, i] * x[:, j]
            iH += 1
    return H

def three(x):
    x = atleast_2d(x)  # in case x is a scalar, assume multiple points in 1D
    n, nx = x.shape
    nH = int(prod(arange(nx, nx + 3)) / prod(arange(1, 3+1)))
    H = empty((n, nH))
    _recurse_poly(x, 3, 0, H, 0, [0]*nx)
    return H

def four(x):
    x = atleast_2d(x)  # in case x is a scalar, assume multiple points in 1D
    n, nx = x.shape
    nH = int(prod(arange(nx, nx + 4)) / prod(arange(1, 4+1)))
    H = empty((n, nH))
    _recurse_poly(x, 4, 0, H, 0, [0]*nx)
    return H

def five(x):
    x = atleast_2d(x)  # in case x is a scalar, assume multiple points in 1D
    n, nx = x.shape
    nH = int(prod(arange(nx, nx + 5)) / prod(arange(1, 5+1)))
    H = empty((n, nH))
    _recurse_poly(x, 5, 0, H, 0, [0]*nx)
    return H

def six(x):
    x = atleast_2d(x)  # in case x is a scalar, assume multiple points in 1D
    n, nx = x.shape
    nH = int(prod(arange(nx, nx + 6)) / prod(arange(1, 6+1)))
    H = empty((n, nH))
    _recurse_poly(x, 6, 0, H, 0, [0]*nx)
    return H

def _recurse_poly(x, p, ip, H, iH, index):
    """
    Recursively evaluate all polynomial terms of order 'p' for input x -
    which has a shape of (# of x dimension, # of unique points).
    The output, H, must be provided and pre-allocated with a shape of
    (int(prod(arange(nd, nd + p)) / prod(arange(1, p+1))), # of unique points).
    The remaining inputs must be initialized as follows:
    ip=0,  iH=0,  index=([0] * # of x dimensions).
    """
    nx = x.shape[1]
    ind_tmp = index.copy()
    if ip == 0:
        j_min = 0
    else:
        j_min = index[ip - 1]
    if ip < p - 1:
        # recursively loop though the unique combinations:
        for j in range(j_min, nx):
            ind_tmp[ip] = j
            H, iH = _recurse_poly(x, p, ip+1, H, iH, ind_tmp)
    else:
        for j in range(j_min, nx):
            ind_tmp[ip] = j
            # evaluate the monomial:
            H[:, iH] = prod([x[:, j] for j in ind_tmp], axis=0)
            iH += 1  # progress the array index
    return H, iH


if __name__ == '__main__':
    from numpy import array, linspace, meshgrid, sqrt
    from numpy.random import rand, randn
    from matplotlib import use as mpl_use
    mpl_use('Qt5Agg')
    from matplotlib import cm, pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Example 1:
    nx = 1
    n = 20
    b = [const, one]
    θ = array([0.2, 2])
    σ = 0.2
    Xd = 3 * rand(n, nx) + 5
    Hd = concatenate([f(Xd) for f in b], axis=1)
    Yd = Hd @ θ + σ * randn(n)

    my_regress = OrdLinRegress(Xd, Yd, b)
    err = sqrt(my_regress.σ2_mean)
    Xr = linspace(0, 3, 50).reshape((-1, 1)) + 5
    Yr = my_regress(Xr)
    print('Example 1:  RMS error = {}'.format(err))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(Xd[:, 0], Yd, 'k.', label='data')
    ax.plot(Xr[:, 0], Yr, 'b', label='regression (mean)')
    ax.fill_between(Xr[:, 0], Yr + err, Yr - err, facecolor='r',
                    alpha=0.25, edgecolor='None', label='data error')
    reg_err = sqrt(my_regress.regression_error(Xr))
    ax.fill_between(Xr[:, 0], Yr + reg_err, Yr - reg_err, facecolor='g',
                    alpha=0.25, edgecolor='None', label='regression error')
    ax.set_title('Example 1')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()


    # # Example 2:
    # nx = 2
    # nd = 30
    # # b = [const, one]
    # # θ = array([0, 1, 2])
    # b = [const, one, two]
    # θ = array([0, 0.5, 0.3, 1, 1.5, 2])
    # σ = 0.1
    # Xd = 2 * rand(nd, nx) - 1
    # Hd = concatenate([f(Xd) for f in b], axis=1)
    # Yd = Hd @ θ + σ * randn(nd)
    #
    # my_regress = OrdLinRegress(Xd, Yd, b)
    # err = sqrt(sum((Yd - my_regress(Xd))**2) / nd)
    # x1 = linspace(-1, 1, 20)
    # X1, X2 = meshgrid(x1, x1, indexing='ij')
    # Xr = concatenate([X1.reshape(-1, 1), X2.reshape(-1, 1)], axis=1)
    # Yr = my_regress(Xr)
    # print('Example 1:  RMS error = {}'.format(err))
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.plot_surface(X1, X2, Yr.reshape(20, 20), alpha=0.75,
    #                 linewidth=0.5, cmap=cm.viridis, rstride=1, cstride=1)
    # ax.scatter(Xd[:, 0], Xd[:, 1], Yd, c='k')
    # ax.set_title('Example 2')
    # ax.set_xlabel('x$_1$')
    # ax.set_ylabel('x$_2$')
    # ax.set_zlabel('y')


    plt.show()
