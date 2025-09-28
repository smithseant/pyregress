r"""
Create a module for general multiple ordinary linear regression (general refers to the optional
use of a weight to specify the ratio of squared errors for each data point; multiple refers to the
dimensionality of x; ordinary refers to the use of a least-squares objective; and linear refers to
the regression coefficients).

To clarify notation, the regression is written as :math:`$y(x) = H(x) \times \beta$`, where
:math:`$\beta$` is a vector of linear coefficients and :math:`$H(x)$` is a matrix of basis
functions for the variates, :math:`$x$`.  The least-squares solution for the coefficients is
:math:`\mu_{\beta} = (H^T(X_d) \times W \times H(X_d))^{-1} H^T(X_d) \times W \times Y_d`.

Created Feb. 2018 @author: Sean T. Smith
"""
from collections import defaultdict as ddict
from abc import ABC, abstractmethod

from numpy import (ndarray, empty, full, arange, atleast_1d, diag_indices, tril, mask_indices,
                   prod, sqrt)
from numpy.linalg import qr, svd, eigh
from numpy.random import default_rng
my_rng = default_rng()
std_norm = my_rng.standard_normal
gamma = my_rng.gamma
from scipy.special import gammainccinv as Qinv
from scipy.linalg import solve_triangular, cho_factor, cho_solve
from scipy.linalg.lapack import dtrtri
from scipy.stats import t

class OrdLinRegress:
    """
    Define a class to perform the ordinary linear regression.  Instantiating an object results in
    a least-squares calculation for the coefficients. Subsequent calls of the object will evaluate
    the regression at the input points.
    """
    def __init__(self, Xd, Yd, bases, W=None, λ=None):
        """
        Arguments:
            Xd:  array-2D,
                location in x of the observed data for each point,
                shape: (# of points/observations, # of x dims.).
            Yd:  array-1D,
                values of the observed data at points corresponding to 1st dim. of Xd.
            bases:  BasisSet object,
                basis functions, when called (with argument X) must return basis functions as an
                array with shape: (# of points in X, # of basis functions).
            W:  array-1D (optional),
                weights corresponding to the ratio of squared errors for each
                data point (w/ a common unknown parameter σ^2 we define W_i = σ^2 / σ_i^2). If omitted, all values are set to unity.
            λ:  float (optional),
                ridge-regression parameter (if `None`, no regularization is used).
        """
        nd, n_xdims = Xd.shape
        if not W:
            W = full(nd, 1.0)
        self.bases = bases
        Hd = self.bases(Xd)
        self.nH = nH = Hd.shape[1]
        # Least-squares solution:
        if λ is not None:
            scaledΣβ_inv = (Hd.T * W) @ Hd  # (scaled by the unknown σ2)
            scaledΣβ_inv[diag_indices(nH)] += λ  # for ridge regression (this term adds stability)
            # # ...using Hermitian eigendecomposition  (slower, but very stable — most comfortable)
            # self.eigΛV = Λ, V = eigh(scaledΣβ_inv)
            # self.μβ = V @ (((Hd.T @ (W * Yd)).T @ V) / Λ).T
            # ...using Cholesky decomposition  (fastest, but not as stable)
            self.choL = cho_factor(scaledΣβ_inv, check_finite=False)
            self.μβ = cho_solve(self.choL, Hd.T @ (W * Yd), check_finite=False)
        elif nd >= nH and λ is None:
            # ...using QR decomposition  (fast & quite stable)
            self.QR = Q, R = qr(sqrt(W).reshape((-1, 1)) * Hd)
            self.μβ = solve_triangular(R, Q.T @ Yd)
        elif nd < nH and λ is None:
            # ...using SVD (slower & most stable — for over- and under-determined)
            sqrtW = sqrt(W)
            self.SVD = U, s, V = svd(sqrtW.reshape((-1, 1)) * Hd, full_matrices=False)
            self.μβ = V.T @ ((U.T @ (sqrtW * Yd)) / s)
        ν = nd - nH
        if λ is not None:
            ν += nH
        self.ν = ν
        if ν > 0:
            s2 = ((Yd - self(Xd))**2 * W).sum() / ν
            # if λ is not None:
            #     s2 = (nH / λ + nd * s2) / (nH + nd)
            self.s2 = s2
            self.σ2_mode = s2 * ν / (ν + 2)
            self.σ2_median = (ν * s2 / 2) / Qinv(ν / 2, 0.5)
            self.σ2_mean = s2 * ν / (ν - 2) if ν > 2 else None

    def __call__(self, Xi):
        """
        Arguments:
            X:  array-2D,
                locations in x where to evaluate the regression, shape: (# of points, # of x dims.)
        Returns:
            Y:  array-1D,
                regression response evaluated at each point in X
        """
        Hi = self.bases(Xi)
        Yi = Hi @ self.μβ
        return Yi

    def prediction_error(self, Xi, percentile=0.6827):
        """
        Arguments:
            X:  array-2D,
                locations in x where to evaluate the regression, shape: (# of points, # of x dims.)
        Returns:
            σy2:  array-1D,
                variance of the noise-free regression uncertainty at each point in X
        """
        Hi = self.bases(Xi)
        if hasattr(self, "eigΛV"):  # ...using Hermitian eigendecomposition
            Λ, V = self.eigΛV
            σy2 = self.s2 * (Hi * (((Hi @ V) / Λ) @ V.T)).sum(axis=1)
        elif hasattr(self, "choL"):  # ...using Cholesky decomposition
            σy2 = self.s2 * (Hi * cho_solve(self.choL, Hi.T, check_finite=False).T).sum(axis=1)
        elif hasattr(self, "QR"):  # ...using QR decomposition
            Q, R = self.QR
            Rinv = dtrtri(R)[0]
            Vβ = Rinv @ Rinv.T
            σy2 = self.s2 * ((Hi @ Vβ) * Hi).sum(axis=1)
        elif hasattr(self, "SVD"):  # ...using SVD
            U, s, V = self.SVD
            HiVTSinv = Hi @ (V.T / s)
            σy2 = self.s2 * (HiVTSinv * HiVTSinv).sum(axis=1)
        return t.ppf(percentile, df=self.ν) * sqrt(σy2)

    def sample_βσ2(self, n_samples):
        σ2_samples = 1 / gamma(self.ν / 2, 2 / (self.ν * self.s2), n_samples)
        norm_samples = sqrt(σ2_samples).reshape((-1, 1)) * std_norm((n_samples, self.nH))
        if hasattr(self, "eigΛV"):  # ...using Hermitian eigendecomposition
            Λ, V = self.eigΛV
            β_samples = self.μβ + ((V / sqrt(Λ)) @ norm_samples.T).T
        elif hasattr(self, "choL"):  # ...using Cholesky decomposition
            L, lower = self.choL
            β_samples = self.μβ + solve_triangular(L, norm_samples.T, trans=0, lower=lower).T
        elif hasattr(self, "QR"):  # ...using QR decomposition
            Q, R = self.QR
            β_samples = self.μβ + solve_triangular(R, norm_samples.T, lower=False).T
        elif hasattr(self, "SVD"):  # ...using SVD
            U, s, V = self.SVD
            β_samples = self.μβ + (norm_samples[:, :s.size] / s) @ V
        return β_samples, σ2_samples

    def sample_pred(self, Xi, n_samples, β_samples=None, σ2_samples=None):
        """Sample a set of noise-free predictions at the specified inference points."""
        if β_samples is None:
            β_samples, _ = self.sample_βσ2(n_samples)
        Hi = self.bases(Xi)
        Yi_samples = Hi @ β_samples.T
        if σ2_samples is not None:
            Yi_samples += sqrt(σ2_samples) * std_norm(n_samples)
        return Yi_samples


class BasisSet(ABC):
    @abstractmethod
    def __init__(self, n_xdims):
        """
        Create a BasisSet object with the attributes `n_xdims` and `n_bases`.
        When possible, include an optional argument `active` (with a default of 'all') so that a
        Boolean mask can be passed to indicate which bases to include and which to exclude.
        """
        # Subclasses are expected to set self.n_xdims & self.n_bases.
        if not hasattr(self, 'n_xdims'):
            raise TypeError('Subclasses of BasisSet must define self.n_xdims.')
        if not hasattr(self, 'n_bases'):
            raise TypeError('Subclasses of BasisSet must define self.n_bases.')
    @abstractmethod
    def __call__(self, x, ret_grad=False):
        """
        With the independent variable `x` conforming to shape (n_pts, n_xdims), return the value
        of the bases evaluated at `x` with shape (n_pts, n_bases), and depending on the optional
        argument `ret_grad` also return the gradient with shape (n_pts, n_xdims, n_bases).
        """
        pass

class Const(BasisSet):
    def __init__(self, n_xdims):
        self.n_xdims = n_xdims
        self.n_bases = 1
    def __call__(self, x, ret_grad=False):
        x_array = atleast_1d(x).reshape((-1, self.n_xdims))
        n_pts = x_array.shape[0]
        H = full((n_pts, 1), 1, dtype="float64")
        if not ret_grad:
            return H
        else:
            Hg = full((n_pts, self.n_xdims, self.n_bases), 0, dtype="float14")
            return H, Hg

class FirstOrd(BasisSet):
    def __init__(self, n_xdims, active="all"):
        self.n_xdims = n_xdims
        if isinstance(active, str) and active == "all":
            self.inc = full(self.n_xdims, True)
        elif isinstance(active, ndarray) and active.shape == (self.n_xdims,):
            self.inc = active
        self.n_bases = self.inc.sum()
    def __call__(self, x, ret_grad=False):
        x_array = atleast_1d(x).reshape((-1, self.n_xdims))
        n_pts = x_array.shape[0]
        H = x_array[:, self.inc]
        if not ret_grad:
            return H
        else:
            Hg = full((n_pts, self.n_xdims, self.n_bases), 0, dtype="float64")
            Hg[:, self.inc, arange(self.n_bases)] = 1
            return H, Hg

class MOrderUnivar(BasisSet):
    def __init__(self, n_xdims, order_m, active="all"):
        self.n_xdims = n_xdims
        self.order = order_m
        if isinstance(active, str) and active == "all":
            self.inc = full(self.n_xdims, True)
        elif isinstance(active, ndarray) and active.shape == (self.n_xdims,):
            self.inc = active
        self.n_bases = self.inc.sum()
    def __call__(self, x, ret_grad=False):
        x_array = atleast_1d(x).reshape((-1, self.n_xdims))
        n_pts = x_array.shape[0]
        H = x_array[:, self.inc]**self.order
        if not ret_grad:
            return H
        else:
            Hg = full((n_pts, self.n_xdims, self.n_bases), 0, dtype="float64")
            i_bases = arange(self.n_bases)
            Hg[:, self.inc, i_bases] = self.order * x_array[:, self.inc]**(self.order - 1)
            return H, Hg

class SecondOrd(BasisSet):
    def __init__(self, n_xdims, active="all"):
        self.n_xdims = n_xdims
        if isinstance(active, str) and active == "all":
            self.inc = full((self.n_xdims, self.n_xdims), False)
            self.inc[mask_indices(self.n_xdims, tril)] = True
        elif isinstance(active, ndarray) and active.shape == (self.n_xdims,):
            self.inc = full((self.n_xdims, self.n_xdims), False)
            self.inc[mask_indices(self.n_xdims, tril)] = True
            self.inc[:, ~active] = False
            self.inc[~active, :] = False
        elif isinstance(active, ndarray) and active.shape == (self.n_xdims, self.n_xdims):
            self.inc = active
        self.n_bases = self.inc.sum()
    def __call__(self, x, ret_grad=False):
        x_array = atleast_1d(x).reshape((-1, self.n_xdims))
        n_pts = x_array.shape[0]
        H = empty((n_pts, self.n_bases))
        iH = 0
        # Loop through the unique combinations of x[i] * x[j]:
        for i in range(self.n_xdims):
            for j in range(self.n_xdims):
                if self.inc[i, j]:
                    # Evaluate the quadratic:
                    H[:, iH] = x_array[:, i] * x_array[:, j]
                    iH += 1
        if not ret_grad:
            return H
        else:
            Hg = full((n_pts, self.n_xdims, self.n_bases), 0, dtype="float64")
            iH = 0
            for i in range(self.n_xdims):
                for j in range(self.n_xdims):
                    if self.inc[i, j]:
                        Hg[:, i, iH] += x_array[:, j]
                        Hg[:, j, iH] += x_array[:, i]
                        iH += 1
            return H, Hg

class BasesList(BasisSet):
    def __init__(self, n_xdims, list_of_basis_sets):
        self.n_xdims = n_xdims
        self.list_of_sets = list_of_basis_sets
        self.n_bases = sum([el.n_bases for el in self.list_of_sets])
    def __call__(self, x, ret_grad=False):
        x_array = atleast_1d(x).reshape((-1, self.n_xdims))
        n_pts = x_array.shape[0]
        H = empty((n_pts, self.n_bases))
        i = 0
        if not ret_grad:
            for el in self.list_of_sets:
                H[:, i:(i + el.n_bases)] = el(x_array)
                i += el.n_bases
            return H
        else:
            Hg = empty((n_pts, self.n_xdims, self.n_bases), dtype="float64")
            for el in self.list_of_sets:
                H[:, i:(i + el.n_bases)], Hg[:, :, i:(i + el.n_bases)] = el(x_array, ret_grad=ret_grad)
            return H, Hg

class PolySet(BasisSet):
    def __init__(self, n_xdims, order, ptype='Legendre', x_range=None):
        """
        Create an object that evaluates multidimensional polynomials at multiple points.
        Arguments:
            n_xdims:  the dimensionality of the independent variable;
            order:  if an int, then all polynomials up to & including the indicated power;
                    if a list of ints, then include only the specified powers;
            ptype:  which of the optional classes of polynomials:
                    'power' (power series),  'Legendre', 'Chebyshev', 'Laguerre' or 'Hermite';
            x_range (optional):  a 2D array (`shape = (2, n_xdims)`) of the low & high values
                                ((low, +σ) for `ptype='Laguerre'` & (-σ, +σ) for `pytype='Hermite'`)
                                 by which the independent variables will be translated & scaled.
        """
        self.n_xdims = n_xdims
        if type(order) == int:
            self.powers = range(order + 1)
        else:
            self.powers = order
        self.n_bases = sum([PolySet.calc_nH(i, self.n_xdims) for i in self.powers])
        self.ptype = ptype
        poly_recursion = dict(power=PolySet.recursion_power,
                              Legendre=PolySet.recursion_Legendre,
                              Chebyshev=PolySet.recursion_Chebyshev,
                              Laguerre=PolySet.recursion_Laguerre,
                              Hermite=PolySet.recursion_Hermite)
        self.recurse_upoly = poly_recursion[self.ptype]
        if x_range is not None:
            self.x_lo = x_range[0].reshape((1, -1))
            self.x_hi = x_range[1].reshape((1, -1))

    @staticmethod
    def calc_nH(order, n_dim):
        '''Calculate the number of bases of a given order for a given number of dimensions.'''
        return int(prod(arange(n_dim, n_dim + order)) / prod(arange(1, order + 1)))
    @staticmethod
    def recursion_power(x, n, p_n, p_prev, dp_n=None, dp_prev=None):
        '''truncated power/Maclaurin series, depending on normalization of `x` (not orthogonal)'''
        p_next = x * p_n
        if dp_n is None:
            return p_next
        else:
            dp_next = p_n + x * dp_n
            return p_next, dp_next
    @staticmethod
    def recursion_Legendre(x, n, p_n, p_prev, dp_n=None, dp_prev=None):
        '''orthogonal for `x` distributed uniformly over the unit box [-1, 1]'''
        p_next = ((2 * n + 1) * x * p_n - n * p_prev) / (n + 1)
        if dp_n is None:
            return p_next
        else:
            dp_next = ((2 * n + 1) * (p_n + x * dp_n) - n * dp_prev) / (n + 1)
            return p_next, dp_next
    @staticmethod
    def recursion_Chebyshev(x, n, p_n, p_prev, dp_n=None, dp_prev=None):
        '''orthogonal for `x` distributed circularly (in the de Moivre way) over [-1, 1]'''
        p_next = 2 * x * p_n - p_prev
        if dp_n is None:
            return p_next
        else:
            dp_next = 2 * (p_n + x * dp_n) - dp_prev
            return p_next, dp_next
    @staticmethod
    def recursion_Laguerre(x, n, p_n, p_prev, dp_n=None, dp_prev=None):
        '''orthogonal for `x` distributed standard exponentially `~Exp(1)`'''
        p_next = ((2 * n + 1 - x) * p_n - n * p_prev) / (n + 1)
        if dp_n is None:
            return p_next
        else:
            dp_next = (-p_n + (2 * n + 1 - x) * dp_n - n * dp_prev) / (n + 1)
            return p_next, dp_next
    @staticmethod
    def recursion_Hermite(x, n, p_n, p_prev, dp_n=None, dp_prev=None):
        '''orthogonal for `x` distributed standard normally `~N(0, 1)`'''
        p_next = x * p_n - n * p_prev
        if dp_n is None:
            return p_next
        else:
            dp_next = p_n + x * dp_n - n * dp_prev
            return p_next, dp_next
    @staticmethod
    def multi_polys(poly_set, order, j_poly, univar_polys, grad_set=None, univar_grads=None,
                    k_dim=0, cum_order=0, product=1, grad_prod=None):
        """
        In the recursive process of evaluating all the multi-dimensional polynomials `poly_set`
        of order `order`, for the next resulting polynomial (indexed as `j_poly`) cumulatively
        multiply (cumulative product stored in `product`) univariate polynomials `univar_polys` by
        looping through each polynomial order that the current dimension `k_dim` may contribute
        while recursing to the next dimension.  By recursing from within the loop, tree branching
        is handled by updating `j_poly` after the final dimension is factored in and the cumulative
        product is stored in `poly_set`.
        """
        n_upolys, n_pts, n_xdims = univar_polys.shape
        if grad_set is not None:
            all_but_k_dim = arange(n_xdims) != k_dim
            if grad_prod is None:
                grad_prod = full((n_pts, n_xdims), 1, dtype='float64')
        if k_dim < n_xdims - 1:
            for i in range(order - cum_order, -1, -1):
                product_i = product * univar_polys[i, :, k_dim]
                if grad_set is None:
                    poly_set, j_poly = PolySet.multi_polys(poly_set, order, j_poly, univar_polys,
                                                           None, None, k_dim + 1, cum_order + i,
                                                           product=product_i)
                else:
                    grad_prod_i = grad_prod.copy()
                    grad_prod_i[:, all_but_k_dim] *= univar_polys[i, :, k_dim].reshape((-1, 1))
                    grad_prod_i[:, k_dim] *= univar_grads[i, :, k_dim]
                    poly_set, grad_set, j_poly = PolySet.multi_polys(poly_set, order, j_poly,
                                                                     univar_polys, grad_set,
                                                                     univar_grads, k_dim + 1,
                                                                     cum_order + i, product_i,
                                                                     grad_prod_i)
        else:
            i = order - cum_order
            poly_set[:, j_poly] = product * univar_polys[i, :, k_dim]
            if grad_set is not None:
                grad_prod[:, all_but_k_dim] *= univar_polys[i, :, k_dim].reshape((-1, 1))
                grad_prod[:, k_dim] *= univar_grads[i, :, k_dim]
                grad_set[:, j_poly] = grad_prod
            j_poly += 1
        if grad_set is None:
            return poly_set, j_poly
        else:
            return poly_set, grad_set, j_poly

    def __call__(self, x, ret_grad=False):
        x_sca = atleast_1d(x).reshape((-1, self.n_xdims))
        n_pts = x_sca.shape[0]
        max_power = max(self.powers)
        if hasattr(self, 'x_lo'):
            if self.ptype == 'Laguerre':
                scaling = self.x_hi - self.x_lo
                x_sca = (x_sca - self.x_lo) / scaling
            else:
                scaling = (self.x_hi - self.x_lo) / 2
                x_sca = (x_sca - self.x_lo) / scaling - 1
        else:
            scaling = 1
       # calculate the uni-variate polynomials, take their products & pack those into the output
        polys_uni = empty((max_power + 1, n_pts, self.n_xdims))
        polys_uni[0] = 1
        if ret_grad:
            grads_uni = empty((max_power + 1, n_pts, self.n_xdims))
            grads_uni[0] = 0
        if max_power >= 1:
            polys_uni[1] = x_sca if self.ptype != 'Laguerre' else 1 - x_sca
            if ret_grad:
                grads_uni[1] = 1 if self.ptype != 'Laguerre' else -1
            for n in range(1, max_power):
                if not ret_grad:
                    polys_uni[n + 1] = self.recurse_upoly(x_sca, n, polys_uni[n], polys_uni[n - 1])
                else:
                    polys_uni[n + 1], grads_uni[n + 1] = self.recurse_upoly(x_sca, n,
                                                                    polys_uni[n], polys_uni[n - 1],
                                                                    grads_uni[n], grads_uni[n - 1])
        if ret_grad:
            grads_uni /= scaling
            grad_set = full((n_pts, self.n_bases, self.n_xdims), 0, dtype='float64')  # or append on poly_set
        poly_set = empty((n_pts, self.n_bases))
        j = 0
        for m in self.powers:
            # Add the product of all combinations of `polys_uni` of order `m` to `poly_set`
            if ret_grad is False:
                poly_set, j = PolySet.multi_polys(poly_set, m, j, polys_uni)
            else:
                poly_set, grad_set, j = PolySet.multi_polys(poly_set, m, j, polys_uni,
                                                            grad_set, grads_uni)
        if ret_grad is False:
            return poly_set
        else:
            return poly_set, grad_set


if __name__ == '__main__':
    from numpy import array, linspace, concatenate, meshgrid, sqrt
    from numpy.random import default_rng
    my_rng = default_rng()
    rand = my_rng.random
    std_norm = my_rng.standard_normal
    from numpy.random import rand, randn
    # from matplotlib import use as mpl_use
    # mpl_use('Qt5Agg')
    from matplotlib import cm, pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Example 1:
    n_xdims = 1
    nd = 6
    bases = BasesList(n_xdims, [Const(n_xdims), FirstOrd(n_xdims), MOrderUnivar(n_xdims, 3)])
    _β = array([0.2, 2, 0.1])
    _σ = 0.5
    x_lo, x_hi = 2, 5
    Xd = (x_hi - x_lo) * rand(nd, n_xdims) + x_lo
    Hd = bases(Xd)
    Yd = Hd @ _β + _σ * randn(nd)
    nr = 200
    n_small, n_large = 75, 1_000

    my_regress = OrdLinRegress(Xd, Yd, bases)
    Xr = (x_hi - x_lo) * linspace(0, 1, nr).reshape((-1, 1)) + x_lo
    Yr = my_regress(Xr)
    reg_err = my_regress.prediction_error(Xr, 0.95)
    β_samples, σ2_samples = my_regress.sample_βσ2(n_large)
    Yr_samples = my_regress.sample_pred(Xr, n_small, β_samples[:n_small])
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    ax.plot(Xd[:, 0], Yd, color='black', marker='.', linestyle='none', label='data')
    p = ax.violinplot((sqrt(σ2_samples) * std_norm(n_large)).reshape((-1, 1)) + Yd,
                      Xd.reshape(-1), widths=0.25, showextrema=False, points=n_large)
    for pc in p['bodies']:
        pc.set_facecolor('tab:red')
        pc.set_alpha(0.5)
    ax.plot(Xr[:, 0], Yr, color='tab:green', label='regression (mean)')
    ax.fill_between(Xr[:, 0], Yr + reg_err, Yr - reg_err, facecolor='tab:green',
                    alpha=0.5, edgecolor='None', label='prediction error (90%)')
    ax.plot(Xr[:, 0], Yr_samples[:,  0], color='black', linewidth=0.25, alpha=0.2,
            label='sampled prediction')
    ax.plot(Xr[:, 0], Yr_samples[:, 1:], color='black', linewidth=0.25, alpha=0.2)
    
    ax.set_xlim(Xr[0], Xr[-1])
    ax.set_ylim(min([Yd.min(), (Yr - reg_err).min(), Yr_samples.min()]),
                max([Yd.max(), (Yr + reg_err).max(), Yr_samples.max()]))
    ax.set_title('Example 1', fontsize=16)
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    ax.legend()
    ax.grid(True, alpha=0.2)

    # Example 2:
    n_xdims = 2
    nd = 30
    bases = BasesList(n_xdims, [Const(n_xdims), FirstOrd(n_xdims), SecondOrd(n_xdims)])
    _β = array([0, 0.5, 0.3, 1, 1.5, 2])
    _σ = 0.1
    Xd = 2 * rand(nd, n_xdims) - 1
    Hd = bases(Xd)
    Yd = Hd @ _β + _σ * std_norm(nd)
    
    my_regress = OrdLinRegress(Xd, Yd, bases)
    err = sqrt(((Yd - my_regress(Xd))**2).sum() / nd)
    x1 = linspace(-1, 1, 20)
    X1, X2 = meshgrid(x1, x1, indexing='ij')
    Xr = concatenate([X1.reshape(-1, 1), X2.reshape(-1, 1)], axis=1)
    Yr = my_regress(Xr)
    print('Example 2:  RMS error = {}'.format(err))
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(X1, X2, Yr.reshape(20, 20), alpha=0.75,
                    linewidth=0.5, cmap=cm.viridis, rstride=1, cstride=1)
    ax.scatter(Xd[:, 0], Xd[:, 1], Yd, c='k')
    ax.set_title('Example 2', fontsize=16)
    ax.set_xlabel('x$_1$', fontsize=16)
    ax.set_ylabel('x$_2$', fontsize=16)
    ax.set_zlabel('y', fontsize=16)

    plt.show()
