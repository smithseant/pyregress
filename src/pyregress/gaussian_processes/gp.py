# -*- coding: utf-8 -*-
"""
Interpolation or regression by means of Gaussian-process inference.
For basic usage see the documentation in the GPI class.

Notation used throughout the code:
  - Primary variables (sometimes used a subscripts):
    X => independent variables,
    Y => dependent variable,
    Z => transformed dependent variable,
    R => distance (radius) in independent variable space,
    K => kernel values (prior covariance matrix),
    φ => (unknown) hyper-parameters of the kernel,
    H => explicit basis functions evaluated at X,
    β => linear coefficients to the basis functions,
    L => lower diagonal of a Cholesky factorization,
    Λ => eigenvalues,
    V => eigenvectors,
    μ => Gaussian location parameter, expected value, median & mode (refers to the posterior),
    Σ => other covariance matrices (refers to the posterior),
  - Additional Subscripts:
    d => data values (observations),
    i => inferred variable values,
    g => gradient of the inferred variable,
    s => sampled values,

Created Sep 2013 @authors: Sean T. Smith & Benjamin B. Schroeder
"""
from warnings import warn
from numpy import (ndarray, array, empty, full, arange, eye, diag, where, squeeze, swapaxes,
                   sum, std, argmax, sqrt, log, pi as π)
from numpy.random import randn
from scipy.special import erfinv
from scipy.linalg import cho_factor, cho_solve, eigh, LinAlgError
from scipy.optimize import minimize
from .kernels import radius, Kernel
from .transforms import BaseTransform

HLOG2PI = 0.5 * log(2 * π)


class GPI:
    """
    Create a Gaussian-process inference (GPR or Kriging) object that can subsequently called for
    regression or interpolation.

    Examples
    --------
    Simple regression in one dimension (w/ known kernel):
    >>> from numpy import array
    >>> from pyregress import GPI, Noise, SquareExp
    >>> Xd = array([0.5, 2.7, 3.6, 6.8, 5.7, 3.4])
    >>> Yd = array([0.0, 1.0, 1.2, 0.5, 0.8, 1.16])
    >>> my_gpi = GPI(Xd, Yd, Noise(w=0.05) + SquareExp(w=2.5, l=2.0))
    >>> print(my_gpi(1.6))
    [0.50832402]

    Probit interpolation in two dimensions w/ underlying linear basis:
    >>> from numpy import array
    >>> from pyregress import GPI, RatQuad, LogNormal, Probit
    >>> Xd = array([[0.00, 0.00], [0.50,-0.10], [1.00, 0.00],
    ...             [0.15, 0.50], [0.85, 0.50], [0.50, 0.85]])
    >>> Yd = array([0.10, 0.30, 0.60, 0.70, 0.90, 0.90])
    >>> K = RatQuad(w=0.6, l=LogNormal(guess=0.3, σ=0.25), α=1)
    >>> my_gpi = GPI(Xd, Yd, K, explicit_basis=[0, 1], transform='Probit', Xscaling='range')
    >>> print(my_gpi( array([[0.10, 0.10], [0.50, 0.42]]) ))
    [0.2361803 0.7968571]
    """
    def __init__(self, Xd, Yd, kernel, Xscaling=None, Ymean=None, explicit_basis=None,
                 transform=None, optimize=True, fast=True):
        """
        Create a GPI object and prepare it for inference.

        Arguments
        ---------
        Xd:  array - 1D or 2D,
            independent-variable observed values. The first index is for multiple observations,
            the second index is for multiple variables (dimensions of the X space).  The second
            index may be omitted for 1D problems.
        Yd:  array-1D [or column-shaped 2D],
            dependent-variable observed values - order corresponding to the first index in `Xd`.
        kernel:  Kernel object,
            prior covariance/autocorrelation — kernel.  Options include: `Noise`, `SquareExp`,
            `GammaExp`, `RatQuad`, or the `Sum` of any of these.
        Xscaling:  None, string or array-1D (optional),
            pre-scaling of the independent variables (kernel anisotropy).  Range scaling: 'range';
            standard deviation scaling: 'std'; and manual scaling: array (length of the second
            index in `Xd`).
        Ymean:  function (optional),
            prior mean function of the dependent variable at `Xd` & `Xi`.  Must accept input with
            of shape `Xd` return array of shape `Yd`.  If omitted, assumes a prior mean of zero.
        explicit_basis:  list of ints (optional),
            explicit basis functions are specified by any combination of the integers:
            0, 1, 2 or 3 - each corresponding to its polynomial order.
        transform:  BaseTransform object (optional),
            specify a dependent variable transformation with a `BaseTransform` object.
            Options include: `Logarithm`, `Logit`, or `Probit`.
        optimize:  bool or string (optional),
            specify whether to optimize the hyper-parameters by maximizing its log posterior.
            If either 'verbose' or 'v', then also report results of optimization.
        
        When`transformation` is provided simultaneously with `Ymean`, the latter is interpreted in
        the untransformed space, but is applied as a difference in the transformed space.
        When `transformation` is provided simultaneously with `explicit_basis`, the later are both
        interpreted and applied in the transformed space.
        Be Aware: Unknown hyperparameters in the kernels are inferred only at their mode (MAP).

        Raises
        ------
        InputError:
            an exception is thrown for incompatible format of any inputs.
        """

        # Independent variables
        if Xd.ndim == 1:
            self.Xd = Xd.reshape((-1, 1)).copy()
        elif Xd.ndim == 2:
            self.Xd = Xd
        self.n_data, self.n_xdims = self.Xd.shape
        if Xscaling is None:
            self.xscale = full(self.n_xdims, 1, dtype='float64')
        elif isinstance(Xscaling, ndarray):
            self.xscale = Xscaling
        elif Xscaling == 'range':
            self.xscale = self.Xd.max(axis=0) - self.Xd.min(axis=0)
        elif Xscaling == 'std':
            self.xscale = std(self.Xd, axis=0)

        # Dependent variable
        if transform is None:
            self.trans = None
            Zd = Yd.reshape((-1, 1)).copy()
        elif transform in ['Logarithm', 'Logit', 'Probit', 'ProbitBeta']:
            self.trans = eval(transform + '(Yd)')
            Zd = self.trans(Yd).reshape((-1, 1))
        elif isinstance(transform, BaseTransform):
            self.trans = transform
            Zd = self.trans(Yd).reshape((-1, 1))

        self.μ_prior = Ymean
        if self.μ_prior is None:
            self.Zd_prime = Zd.reshape((-1, 1))
        elif not self.trans:
            self.Zd_prime = Zd.reshape((-1, 1)) - self.μ_prior(self.Xd).reshape((-1, 1))
        else:
            self.Zd_prime = Zd .reshape((-1, 1))- self.trans(self.μ_prior(self.Xd)).reshape((-1, 1))

        # Explicit bases
        self.basis = explicit_basis
        if self.basis is not None:
            self.bases = explicit_basis
            self.n_β = self.bases.n_bases
            self.Hd = self.bases(self.Xd)

        # Kernel (prior covariance)
        if not isinstance(kernel, Kernel):
            raise InputError("GPI argument kernel must be a Kernel object.", kernel)
        self.kernel = kernel
        self.fast = fast

        # Do as many calculations as possible in preparation for the inference.
        self.Rdd = radius(self.Xd, self.Xd, self.xscale)
        if self.kernel.n_φ > 0 and optimize:
            if optimize == 'verbose' or optimize == 'v':
                self.maximize_posterior_φ(optimize, verbose=True)
            else:
                self.maximize_posterior_φ(optimize, verbose=False)
        else:
            self._one_time_prep()

    def _one_time_prep(self):
        """
        Pre-calculate the expensive operations that need only be performed once in preparation
        for the inference.
        """
        self.Kdd = self.kernel(self.Rdd)
        try:
            if not self.fast:
                raise InputError('not fast')
            # Attempt Cholesky since it is fast, but it can go unstable.
            self.LKdd = cho_factor_gen(self.Kdd)
            self.solve = cho_solve_gen
        except (InputError, ValueError, LinAlgError):
            # Downshift to the slower, but more stable, eigen-decomposition.
            self.LKdd = eigh(self.Kdd)  # sorts eigenvalues small to large
            eig_solve = lambda Λ, V, b: (V @ ((V.T @ b).T / Λ).T)
            self.solve = lambda L, b: eig_solve(*L, b)
            Λ, V = self.LKdd
            # Eigenvalues that are too small require further intervention...
            i_keep = argmax(Λ > 1e-14 * Λ[-1])
            if i_keep < self.n_data:
                warn('The data kernel was automatically modified to maintain positive definiteness'
                     ' & avoid buildup of round-off error.',
                     RuntimeWarning)
                Λ = Λ[i_keep:]
                V = V[:, i_keep:]
                # Λ[:i_keep] = 1e-14  # Targeted noise
                self.LKdd = (Λ, V)
        self.KinvZd = self.solve(self.LKdd, self.Zd_prime)
        if self.basis is not None:
            self.KinvHd = self.solve(self.LKdd, self.Hd)
            LinvΣβ = cho_factor_gen(self.Hd.T @ self.KinvHd)
            self.Σβ = cho_solve_gen(LinvΣβ, eye(self.n_β))
            # ...results in an unnecessary matrix product: (V.T @ I).T, but it's not very expensive.
            self.μβ = cho_solve_gen(LinvΣβ, self.Hd.T @ self.KinvZd)
            self.KinvZd = self.solve(self.LKdd, self.Zd_prime - self.Hd @ self.μβ)
        return self

    def posterior_φ(self, φ, ret_grad=True, trans=True):
        """
        Negative log of the hyper-parameter posterior & its gradient.

        Arguments
        ---------
        φ:  array-1D,
            hyper parameters in an array for the minimization routine.
        ret_grad:  bool or string (optional),
            when ret_grad is True, also return lnP_grad.
        trans:  bool (optional),
            indicate whether the provided φ is in its transformed space.

        Returns
        -------
        lnP_neg:  float,
            negative log of the hyper-parameter posterior.
        lnP_grad:  array-1D (optional - depending on argument ret_grad),
            gradient of lnP_neg with respect to each hyper-parameter.
        """
        if len(φ.shape) > 1:   # Corrects odd behavior of scipy's minimize
            φ = φ[0]           # Corrects odd behavior of scipy's minimize
        n_pts, n_φ = self.n_data, self.kernel.n_φ
        K = self.kernel.Kφ(φ, self.Rdd, ret_grad=ret_grad, trans=trans)
        lnprior = self.kernel.ln_priors(φ, ret_grad=ret_grad, trans=trans)
        if ret_grad:
            K, Kg = K
            lnprior, dlnprior = lnprior
        try:
            # For covariance matrices Cholesky is fast, but less stable.
            LK = cho_factor(K)
            solve = cho_solve
            ln_detK = sum(log(diag(LK[0])))  # actually: ln(det(K))/2
        except LinAlgError as e:
            # Downshifting to the slower, but more stable, eigen method.
            LK = eigh(K)
            eig_solve = lambda Λ, V, b: (V @ ((V.T @ b).T / Λ).T)
            solve = lambda L, b: eig_solve(*L, b)
            Λ, V = LK
            Λmin = 1e-12 * Λ[-1]
            if Λ[0] < Λmin:
                Λ = where(Λ < Λmin, Λmin, Λ)
                LK = (Λ, V)
            ln_detK = 0.5 * sum(log(Λ))  # actually: ln(det(K))/2
        KinvZd = solve(LK, self.Zd_prime)

        lnP_neg = n_pts * HLOG2PI + ln_detK + 0.5 * self.Zd_prime.T @ KinvZd - lnprior
        lnP_neg = squeeze(lnP_neg)

        if self.basis is not None:
            n_β = self.n_β
            KinvHd = solve(LK, self.Hd)
            invΣβ = self.Hd.T @ KinvHd
            LinvΣβ = cho_factor(invΣβ)
            Σβ = cho_solve(LinvΣβ, eye(n_β))
            μβ = cho_solve(LinvΣβ, self.Hd.T @ KinvZd)
            KinvHdμβ = KinvHd @ μβ
            lnP_neg -= squeeze(n_β * HLOG2PI - sum(log(diag(LinvΣβ[0]))) +
                               0.5 * μβ.T @ invΣβ @ μβ)

        if not ret_grad:
            return lnP_neg
        # else ret_grad:
        Kinv = solve(LK, eye(n_pts))
        Kinv_αα = Kinv - KinvZd @ KinvZd.T
        lnP_grad = empty(n_φ)
        for j in range(n_φ):
            lnP_grad[j] = 0.5 * sum(Kinv_αα.T * Kg[:, :, j]) - dlnprior[j]
        if self.basis is not None:
            Δ2 = KinvHdμβ.T - 2 * KinvZd.T
            for j in range(n_φ):
                βKgβ = KinvHd.T @ Kg[:, :, j] @ KinvHd
                lnP_grad[j] -= 0.5*(sum(βKgβ.T * Σβ) + Δ2 @ Kg[:, :, j] @ KinvHdμβ)
        return lnP_neg, lnP_grad

    def maximize_posterior_φ(self, trans=True, verbose=True):
        """
        Find the maximum of the hyper-parameter posterior (minimum of -ln(P)).

        Arguments
        ---------
        trans - specify whether to solve for φ in a transformed space.
        verbose - specify whether to print status of the minimization routine.
        """
        # Warning: running this routine manually requires additional calculations before inference
        #          can be performed properly.
        # Setup hyper-parameters & map values from a single array
        φ = self.kernel.get_φ(trans=trans)
        f = self.posterior_φ
        # Perform minimization
        out = minimize(f, φ, (False, trans), method='Nelder-Mead',
                       options={'maxiter':10000, 'xatol':1e-5, 'fatol':3e-5})
        # out = minimize(f, φ, (False, trans), method='Powell')
        # out = minimize(f, φ, (False, trans), method='CG')
        # out = minimize(f, φ, (True, trans), method='BFGS', jac=True)
        # Use the optimized value:
        φ = out.x
        self.kernel.update_p(φ, trans=trans, set=True)
        if verbose:
            print(f'Optimize φ: {out.nfev} post. evals. & {out.nit} iters. gave f={out.fun:5.2g} '
                  f'& p = {self.kernel.p}')
        # Reevaluate the variables needed for inference:
        self._one_time_prep()
        return self, φ

    def __call__(self, Xi, kernel_terms='noisefree', infer_std=False, exclude_mean=False,
                 untransform=True, infer_grad=False):
        """
        Make inferences (interpolation or regression) of the posterior's sufficient statistics,
        μ_Y|Yd & (σ_Y|Yd or Σ_Y|Yd) at the provided locations, `Xi`. This inference is limited to
        a single value of each hyper-parameters (the MAP).

        Arguments
        ---------
        Xi:  array-2D,
            independent variables - where to make inferences. The first index is for multiple
            inferences, and second index must match that of the argument Xd when calling GPI.
        kernel_terms:  'noisefree', 'all', int or list of ints (optional),
            Applies only when kernel argument to GPI is a KernelSum:
            if 'noisefree', infer an underlying smoothly correlated regression function by assuming
                            that any uncorrelated Noise term in the specified kernel applies to the
                            observed values (Yd) only and not to the inferred values (Yi);
            if 'all', use all of the terms in the specified kernel (eg. simulate new observations);
            if int or list of ints, then use only that indexed subset of terms in order;
            Note: including a Noise term at this stage is not compatible with gradient inference.
        infer_std:  bool, float, or 'covar' (optional),
            if True, return the inferred standard deviation;
            if float in (0, 1), return the uncertainty for this inner quantile,
            if 'covar', return the full posterior covariance matrix.
        exclude_mean:  bool (optional),
            if `False` make inference inclusive of the prior mean & basis functions,
            if `True` return the partial inference that is relative to them.
        untransform:  bool (optional),
            if False, any inverse transformation is suppressed.
        infer_grad:  bool (optional),
            whether to return the inferred gradient of the dependent variable.

        Returns
        -------
        μ_post:  array-1D,
            inferred mean/median/mode at each location in the argument Xi
            (for a non-linear inverse transform, this value represents only the median —
             and only for monotonic transforms).
        μ_post_grad:  array-2D (optional — depending on infer_grad),
            inferred mean/median/mode of the gradient at each location in Xi.
        σ_post:  array - 1D or 2D or list (optional - depending on infer_std),
            inferred standard deviation, analogous half-width (given an inner quantile), or the
            full covariance (for an inv. trans, returns the low & high ends of the inner quantile).
        σ_post_grad:  array-2D (optional — depending on both infer_std & infer_grad),
            inferred standard deviation of the gradient, or full covariance.

        Raises
        ------
        InputError:
            an exception is thrown for incompatible format of any inputs.
        """
        # Raise errors before any calculations are performed (structure parallels testing)
        if self.trans and untransform is True:
            if infer_std == "covar":
                raise InputError(error_trans_covar)
            if infer_std and infer_grad:
                raise InputError(error_trans_grad)
            if (self.μ_prior or self.basis is not None) and exclude_mean:
                raise InputError(error_trans_exclude)

        # Independent variables
        if isinstance(Xi, float) or isinstance(Xi, int) and self.Xd.shape[1] == 1:
            Xi = array([[Xi]], dtype='f8')  # a 1D problem at a single point
        elif Xi.ndim == 1 and self.n_xdims == 1:
            Xi = Xi.copy().reshape((-1, 1))  # a 1D problem at n points
        elif Xi.ndim == 1 and self.n_xdims == Xi.shape[0]:
            Xi = Xi.copy().reshape((1, -1))  # an nD problem at a single point
        n_inf = Xi.shape[0]
        ind_i = arange(n_inf)
        ind_ig = arange(n_inf * (1 + self.n_xdims))

        # Evaluate the prior mean at Xi and shift to the transformed space
        if self.μ_prior:
            if not infer_grad:
                μYi_prior = self.μ_prior(Xi)
                if self.trans:
                    μZi_prior = self.trans(μYi_prior)
                else:
                    μZi_prior = μYi_prior  # apply the identity transform (using a pointer)
            else:
                μYi_prior, μYg_prior = self.μ_prior(Xi, ret_grad=True)
                if self.trans:
                    μZi_prior, μZg_prior = self.trans(μYi_prior, grad_y=μYg_prior)
                else:
                    μZi_prior, μZg_prior = μYi_prior, μYg_prior

        # Evaluate the explicit bases
        if self.basis is not None:
            if not infer_grad:
                Hi = self.bases(Xi)
            else:
                Hig = full((n_inf, (1 + self.n_xdims), self.n_β), 0, dtype='float64')
                Hi, Hg = self.bases(Xi, infer_grad)
                Hg = swapaxes(Hg, 1, 2)
                Hig[:, 0, :] = Hi
                Hig[:, 1:, :] = Hg

        # Distance & auto-covariance
        Rid = radius(Xi, self.Xd, self.xscale)
        Kid = self.kernel(Rid, sum_terms=kernel_terms, i_grad=infer_grad, s=self.xscale)
        if infer_std:
            Rii = radius(Xi, Xi, self.xscale)
            Kii = self.kernel(Rii, sum_terms=kernel_terms, ii_grad=infer_grad, s=self.xscale)

        # Inference
        μZi = Kid[:n_inf] @ self.KinvZd
        if infer_grad:
            μZg = (Kid[n_inf:] @ self.KinvZd).reshape((self.n_xdims, n_inf)).T
        if not exclude_mean:
            if self.μ_prior:
                μZi += μZi_prior
                if infer_grad:
                    μZg += μZg_prior
            if self.basis is not None:
                μZi += Hi @ self.μβ
                if infer_grad:
                    μZg += Hg @ self.μβ.reshape(-1)
        if infer_std:
            if isinstance(infer_std, float):
                zp = sqrt(2) * erfinv(infer_std)
                infer_std = True
            elif infer_std is True:
                zp = 1
            Σii = Kii - Kid @ self.solve(self.LKdd, Kid.T)
            if self.basis is not None:
                if not infer_grad:
                    tmp = Hi - Kid @ self.KinvHd
                else:
                    tmp = Hig.T.reshape(((1 + self.n_xdims), -1)).T - Kid @ self.KinvHd
                Σii += tmp @ (self.Σβ @ tmp.T)

        # Untransform and return the requested variables
        if not self.trans or not untransform:
            # when `not self.trans`, imply the identity untransform...
            if infer_std is False:
                if not infer_grad:
                    return μZi.reshape(-1)
                else:
                    return μZi.reshape(-1), μZg
            elif infer_std is True:
                if not infer_grad:
                    σZi = sqrt(Σii[ind_i, ind_i]).reshape((-1, 1))
                    return μZi.reshape(-1), zp * σZi.reshape(-1)
                else:
                    σZig = sqrt(Σii[ind_ig, ind_ig]).reshape(-1)
                    σZi, σZg = σZig[:n_inf], σZig[n_inf:].reshape((self.n_xdims, n_inf)).T
                    return (μZi.reshape(-1), μZg), (zp * σZi.reshape(-1), zp * σZg)
            elif infer_std == "covar":
                if not infer_grad:
                    return μZi.reshape(-1), Σii
                else:
                    return (μZi.reshape(-1), μZg), Σii
        else:
            if infer_std is False:
                warn(warn_trans_μ, RuntimeWarning)
                if not infer_grad:
                    μYi = self.trans(μZi, inverse=True)
                    return μYi.reshape(-1)
                else:
                    μYi, μYg = self.trans(μZi, grad_z=μZg, inverse=True)
                    return μYi.reshape(-1), μYg
            elif infer_std is True or isinstance(infer_std, float):
                warn(warn_trans_μ + warn_trans_σ, RuntimeWarning)
                if not infer_grad:
                    μYi = self.trans(μZi, inverse=True)
                    σZi = sqrt(Σii[ind_i, ind_i]).reshape((-1, 1))
                    Zi_lohi = array([(μZi - zp * σZi).reshape(-1), (μZi + zp * σZi).reshape(-1)])
                    Yi_lohi = self.trans(Zi_lohi, inverse=True)
                    return μYi.reshape(-1), Yi_lohi
                else:
                    raise InputError(error_trans_grad)  # raised above, here for logical symmetry
            elif infer_std == "covar":
                raise InputError(error_trans_covar)  # raised above, here for logical symmetry

    def sample(self, Xs, n_samples=1, kernel_terms='noisefree', exclude_mean=False, sample_grad=False):
        """
        Sample the Gaussian process at specified locations.

        Arguments
        ---------
        Xs:  array-2D,
            independent variables - where to sample. First index is for multiple (correlated)
            inferences, and second index must match that of the argument Xd from GPI.__init__.
        n_samples: int (optional),
            allows the calculation of multiple samples at once.
        kernel_terms:  'noisefree', 'all', int or list of ints (optional),
            Applies only when kernel argument to GPI is a KernelSum:
            if 'noisefree', infer an underlying smoothly correlated regression function by assuming
                            that any uncorrelated Noise term in the specified kernel applies to the
                            observed values (Yd) only and not to the inferred values (Yi);
            if 'all', use all of the terms in the specified kernel (eg. simulate new observations);
            if int or list of ints, then use only that indexed subset of terms in order;
            Note: including a Noise term at this stage is not compatible with gradient inference.
        exclude_mean:  bool (optional),
            if False include prior mean and basis functions, otherwise don't.
        sample_grad:  bool (optional),
            whether to return the inferred gradient of the dependent variable.

        Returns
        -------
        Ys:  array-2D,
            sample value from the poster at each location in Xs.
        μ_post_grad:  array-2D,
            mean gradient of the posterior at each location in Xs.

        Raises
        ------
        InputError:
            an exception is thrown for incompatible format of any inputs.
        """
        n_xdims = self.n_xdims
        if isinstance(Xs, float) or isinstance(Xs, int) and self.Xd.shape[1] == 1:
            Xs = array([[Xs]], dtype='f8')  # a 1D problem at a single point
        elif Xs.ndim == 1 and self.n_xdims == 1:
            Xs = Xs.copy().reshape((-1, 1))  # a 1D problem at n points
        elif Xs.ndim == 1 and self.n_xdims == Xs.shape[0]:
            Xs = Xs.copy().reshape((1, -1))  # an nD problem at a single point
        n_pts = Xs.shape[0]
        μ_post, Σ = self.__call__(Xs, infer_std='covar', untransform=False,
                                  kernel_terms=kernel_terms, exclude_mean=exclude_mean,
                                  infer_grad=sample_grad)
        if not sample_grad:
            n_z = n_pts
        else:
            μ_post, μ_post_grad = μ_post
            n_z = n_pts * (1 + self.n_xdims)
            μ_post_both = empty(n_z)
            μ_post_both[:n_pts] = μ_post.reshape(-1)
            μ_post_both[n_pts:] = μ_post_grad.T.reshape(-1)
            μ_post = μ_post_both

        Z = randn(n_z, n_samples)
        Λ, V = eigh(Σ)
        Ys = μ_post + (sqrt(Λ.clip(min=0)) * Z.T) @ V.T
        if not sample_grad:
            Ys = Ys.reshape((n_samples, n_pts))
            if self.trans:
                Ys = self.trans(Ys, inverse=True)
        else:
            Ys = Ys.reshape((n_samples, (1 + n_xdims), n_pts))
            Ys = Ys[:, 0, :], Ys[:, 1:, :]
            if self.trans:
                Ys = self.trans(Ys[0], grad_z=Ys[1], inverse=True)
        return Ys


def cho_factor_gen(A, lower=False, **others):
    """Generalize scipy's cho_factor to handle arrays of length zero."""
    if A.size == 0:
        return empty(A.shape), lower
    else:
        try:
            return cho_factor(A, lower=lower, **others)
        except LinAlgError as e:
            e.args += (("GPI method __init__ failed to factor data kernel. This often indicates "
                        "that X has near duplicates or the noise kernel has too small of weight."),)
            raise e


def cho_solve_gen(C, b, **others):
    """Generalize scipy's cho_solve to handle arrays of length zero."""
    if C[0].size == 0:
        return empty(b.shape)
    else:
        return cho_solve(C, b, **others)


warn_trans_μ = ("Using `untransform=True`, limits the meaning of the returned `μ` values to the"
                " median (& only for monotonic transformations).  ")
warn_trans_σ = ("`infer_std=True` limits the meaning of the returned `σ` values to the 68.2% inner"
                " quantile region.  For other quantiles use the sample method.")
error_trans_exclude = ("Inference parameter `exclude_mean=True` is incompatible with transformation"
                       " & untransform.")
error_trans_covar = ("Inference parameter `infer_std=='covar'` is incompatible with transformation"
                     " & untransform.")
error_trans_grad = ("Combining these inference parameters is incompatible: infer_std is not False, "
                    "grad=True, transform is not None, and untransform=True — consider sampling.")

class GPError(Exception):
    """Base class for exceptions in the pyregress module."""
    pass

class InputError(GPError):  # -- not a ValueError? --
    """Exception raised for errors in input arguments."""
    def __init__(self, msg, input_argument=None):
        """
        Initialize an InputError.

        Arguments
        ---------
            msg:  string,
                explanation of the error.
            input_argument:  any (optional),
                input argument that is the source of error. Provided so the value can be reported
                when the error is caught.
        """
        self.args = (msg,)
        self.input_argument = input_argument

class ValidationError(GPError):
    def __init__(self, msg, n_dims=None, N3=None, N2=None):
        """
        Initialize a ValidationError when the inference is failing its cross validation.

        Arguments
        ---------
            msg:  string,
                explanation of the error.
            n_dims:  integer (optional),
                Number of cross validation data points.
            N3:  integer (optional),
                Number of points that have abs(std_err) > 3.0
            N2:  integer (optional),
                Number of points that have abs(std_err) > 2.0
        """
        self.args = (msg % (n_dims, N3),)
        self.n_data = n_dims
        self.N3 = N3
        self.N2 = N2


if __name__ == "__main__":
    from numpy import array, linspace, meshgrid
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from ..lin_regress import PolySet
    from .kernels import Noise, SquareExp, RatQuad
    from .transforms import Probit
    from .hyper_params import LogNormal

    # Example 1  (1D regression w/ six data points & a known kernel):
    # Setup...
    # (for 1D problems, the shape of the input data is flexible)
    Xd = array([0.5, 2.7, 3.6, 6.8, 5.7, 3.4])  # or w/ `.reshape((-1, 1))`
    Yd = array([0.0, 1.0, 1.2, 0.5, 0.8, 1.16])  # or w/ `.reshape((-1, 1))`
    my_gpi = GPI(Xd, Yd, Noise(w=0.05) + SquareExp(w=2.5, l=2.0))
    # Inference at a point...
    # (again, for 1D problems w/ a single point, any of these three forms are accepted)
    xi = 1.6  # `array([1.6])`  or  `array([1.6]).reshape((-1, 1))`
    yi_μ = my_gpi(xi)  # infer the posterior mean (underlying regression function) at this point
    yi_μ, yi_σ = my_gpi(xi, infer_std=True)  # infer the posterior σ as well
    yi_μ, dyi_μ = my_gpi(xi, infer_grad=True)  # infer the gradient as well
    (yi_μ, dyi_μ), (yi_σ, dyi_σ) = my_gpi(xi, infer_std=True, infer_grad=True)  # get both
    yi_samp = my_gpi.sample(xi, 100)  # sample the posterior at this point
    # Plotting (w/ inference of the whole function)...
    Xi = linspace(0, 7.5, 200)
    Yi_μ, Yi_65 = my_gpi(Xi, infer_std=0.65)  # specified inner quantile of the posterior
    Yi_μ, Yi_95 = my_gpi(Xi, infer_std=0.95)  # specified inner quantile of the posterior
    Yi_samp = my_gpi.sample(Xi, 30)
    # create a tangent line from the gradient
    δ = 0.25  # half width of the tangent line
    Xig = (xi + δ * array([-1.0, 1.0])).reshape(-1, 1)
    Yig = (yi_μ + dyi_μ * δ * array([-1.0, 1.0])).reshape(-1, 1)
    plt.figure(figsize=(7, 5))
    plt.plot(Xd, Yd, 'bo', label='observed data')
    plt.plot(xi, yi_μ, 'go', label='example regression point')
    plt.plot(Xig, Yig, 'g-', linewidth=3.0, label='inferred slope at that point')
    plt.plot(Xi, Yi_μ, 'g-', linewidth=1.5, alpha=0.65, label='inferred mean')
    plt.fill_between(Xi, Yi_μ - Yi_65, Yi_μ + Yi_65, color='tab:green', alpha=0.50,
                     label='posterior inner 65% quantile')
    plt.fill_between(Xi, Yi_μ - Yi_95, Yi_μ + Yi_95, color='tab:green', alpha=0.15,
                     label='posterior inner 95% quantile')
    plt.plot(Xi, Yi_samp[0], 'b-', linewidth=0.5, alpha=0.15, label='posterior samples')
    plt.plot(Xi, Yi_samp[1:].T, 'k-', linewidth=0.5, alpha=0.15)
    plt.xlim(Xi[0], Xi[-1])
    plt.grid(True, alpha=0.1)
    plt.title('Example #1  (1D regression w/ 6 data points & a known kernel)',
            fontsize=12)
    plt.xlabel('independent variable, X', fontsize=12)
    plt.ylabel('dependent variable, Y', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)

    # Example 2  (probit interpolation in 2D w/ a linear basis & 6 data points):
    # Setup...
    Xd = array([[0.00, 0.00], [0.50,-0.10], [1.00, 0.00],
                [0.15, 0.50], [0.85, 0.50], [0.50, 0.85]])
    Yd = array([0.10, 0.30, 0.60, 0.70, 0.90, 0.90])
    bases = PolySet(2, 1, x_range=array([[0, -0.5], [1, 1]]))
    K = RatQuad(w=0.6, l=LogNormal(guess=0.3, σ=0.25), α=1)
    my_gpi = GPI(Xd, Yd, K, explicit_basis=bases, transform='Probit', Xscaling='range')
    # Inference at a point...
    print('Example 2: optimized value of the hyper-parameter:', my_gpi.kernel.get_φ())
    xi = array([[0.10, 0.10], [0.50, 0.42]])
    yi_μ = my_gpi(xi)
    # Plotting (w/ inference of the whole function)...
    Ni = (30, 32)
    xi_1 = linspace(-0.2, 1.2, Ni[0])
    xi_2 = linspace(-0.3, 1.0, Ni[1])
    Xi_1, Xi_2 = meshgrid(xi_1, xi_2, indexing='ij')
    Xi = array([Xi_1.reshape(-1), Xi_2.reshape(-1)]).T
    Yi_μ, Yi_lohi = my_gpi(Xi, infer_std=0.99)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(Xi_1, Xi_2, Yi_μ.reshape(Ni),
                    alpha=0.75, linewidth=0.5, cmap=mpl.colormaps['jet'], rstride=1, cstride=1)
    ax.plot_surface(Xi_1, Xi_2, Yi_lohi[0].reshape(Ni),
                    alpha=0.25, linewidth=0.25, color='black', rstride=1, cstride=1)
    ax.plot_surface(Xi_1, Xi_2, Yi_lohi[1].reshape(Ni),
                    alpha=0.25, linewidth=0.25, color='black', rstride=1, cstride=1)
    ax.scatter(Xd[:, 0], Xd[:, 1], Yd, c='black', s=35)
    ax.set_zlim([0.0, 1.0])
    ax.set_title('Example #2  (probit interpolation in 2D w/ a linear basis)',
                fontsize=12)
    ax.set_xlabel('independent variable, X1', fontsize=12)
    ax.set_ylabel('independent variable, X2', fontsize=12)
    ax.set_zlabel('dependent variable, Y', fontsize=12)

    plt.show()
