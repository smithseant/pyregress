# -*- coding: utf-8 -*-
"""
Interpolation or regression by means of Gaussian-process inference.
For basic usage see the documentation in the GPI class.

Performance:
    Calculation time will greatly depend on which Blas/Lapack libs are used. Some default
    python/numpy/scipy packages are based on unoptimized libs (including linux repositories),
    Anaconda provides optimized libs.
Notation used throughout the code:
    X => independent variables,
    Y => dependent variable,
    Z => transformed dependent variable,
    d => data values (observations),
    i => inferred values,
    yd_s => sampled values,
    μ => Gaussian location parameter & expected value (for various values),
    K => kernel values (prior covariance matrix),
    φ => (unknown) hyper-parameters of the kernel,
    R => distance (radius) in independent variable space,
    Σ => other covariance matrices,
    H => explicit basis functions evaluated at X,
    Θ => linear coefficients to the basis functions,
    p => derivative (prime) of a variable,
    L => lower diagonal of a Cholesky factorization,
    Λ => eigenvalues,
    V => eigenvectors.

Created Sep 2013 @authors: Sean T. Smith & Benjamin B. Schroeder
"""
from copy import deepcopy
from warnings import warn
from numpy import (ndarray, array, empty, zeros, ones, arange, eye, diag,
                   where, squeeze, sum, prod, std, count_nonzero,
                   amin, amax, maximum, argmax, abs, sqrt, log, pi as π)
from numpy.random import randn
from numba import jit
from scipy.linalg import cho_factor, cho_solve, eigh, LinAlgError
from scipy.optimize import minimize
from .kernels import Kernel
from .transforms import BaseTransform, Logarithm, Logit, Probit, ProbitBeta

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
    >>> myGPI = GPI(Xd, Yd, Noise(w=0.05) + SquareExp(w=2.5, l=2.0))
    >>> print(myGPI(1.6))
    [0.50832402]

    Probit interpolation in two dimensions w/ underlying linear basis:
    >>> from numpy import array
    >>> from pyregress import GPI, RatQuad, LogNormal, Probit
    >>> Xd = array([[0.00, 0.00], [0.50,-0.10], [1.00, 0.00],
    ...             [0.15, 0.50], [0.85, 0.50], [0.50, 0.85]])
    >>> Yd = array([0.10, 0.30, 0.60, 0.70, 0.90, 0.90])
    >>> K = RatQuad(w=0.6, l=LogNormal(guess=0.3, σ=0.25), α=1)
    >>> myGPI = GPI(Xd, Yd, K, explicit_basis=[0, 1], transform=Probit, Xscaling='range')
    >>> print(myGPI( array([[0.10, 0.10], [0.50, 0.42]]) ))
    [0.2361803 0.7968571]
    """
    def __init__(self, Xd, Yd, kernel, Xscaling=None, Ymean=None, explicit_basis=None, transform=None,
                 optimize=True, fast=True):
        """
        Create a GPI object and prepare it for inference.

        Arguments
        ---------
        Xd:  array - 1D or 2D,
            independent-variable observed values. The first index is for multiple observations,
            the second index is for multiple variables (dimensions of the X space). The second
            index may be omitted for 1D problems.
        Yd:  array-1D [or column-shaped 2D],
            dependent-variable observed values - order corresponding to the first index in Xd.
        kernel:  Kernel object,
            prior covariance/autocorrelation — kernel. Options include: Noise, SquareExp, GammaExp,
            RatQuad, or the sum of any of these.
        Xscaling:  string or array-1D (optional),
            pre-scaling of the independent variables (kernel anisotropy). Range scaling: 'range';
            standard deviation scaling: 'std'; and manual scaling: array (length of the second
            index in Xd).
        Ymean:  function (optional),
            prior mean of the dependent variable at Xd & Xi. It must accept input in form of Xd,
            and must provide output the same shape as Yd. If omitted, assumes a prior mean of zero.
        explicit_basis:  list of ints (optional),
            explicit basis functions are specified by any combination of the integers:
            0, 1, 2 or 3 - each corresponding to its polynomial order.
        transform:  string or BaseTransform object (optional),
            specify a dependent variable transformation with the name of a BaseTransform class
            (as a string) or a BaseTransform object.
            Options include: Logarithm, Logit, Probit, or ProbitBeta.
        optimize:  bool or string (optional),
            specify whether to optimize the hyper-parameters by maximizing its log posterior.
            If either 'verbose' or 'v', then also report results of optimization.
        
        When a transformation is provided simultaneously with Ymean, the latter is interpreted in
        the untransformed space, but is applied as a difference in the transformed space.
        When a transformation is provided simultaneously with explicit_basis, the later are both
        interpreted and applied in the transformed space.
        Be Aware: Unknown hyperparameters in the kernels are inferred only at their mode (MAP).

        Raises
        ------
        InputError:
            an exception is thrown for incompatible format of any inputs.
        """

        # Independent variables
        if Xd.ndim == 1:
            self.Xd = Xd.reshape((-1, 1)).copy()  # assume multiple observations in 1D
        elif Xd.ndim == 2:
            self.Xd = Xd
        else:
            raise InputError("GPI argument Xd must be a 1D or 2D array.", Xd)
        self.n_pts, self.n_dims = self.Xd.shape
        if Xscaling is None:
            self.xscale = ones(self.n_dims)
        elif isinstance(Xscaling, ndarray):
            self.xscale = Xscaling
        elif Xscaling == 'range':
            self.xscale = self.Xd.max(axis=0) - self.Xd.min(axis=0)
        elif Xscaling == 'std':
            self.xscale = std(self.Xd, axis=0)
        else:
            raise InputError("GPI argument Xscaling must be one of: None, False, True, 'range', "
                             "'std', or 1D array (same length as the 2nd dim. of Xd)", Xscaling)
        # Dependent variable
        if Yd.shape[0] != self.n_pts:
            raise InputError("GPI argument Yd must have the same length as the 1st dim. of Xd.", Yd)
        self.Yd = Yd.reshape((-1, 1)).copy()
        if transform is None:
            self.trans = None
        elif transform in ['Logarithm', 'Logit', 'Probit', 'ProbitBeta']:
            self.trans = eval(transform + '(self.Yd)')
            self.Yd = self.trans(self.Yd)
        elif isinstance(transform, BaseTransform):
            self.trans = transform
            self.Yd = self.trans(self.Yd)
        else:
            raise InputError("GPI argument transform must be a BaseTransform "
                             "class.", transform)
        self.μ_prior = Ymean
        if self.μ_prior is not None:
            if self.trans is None:
                self.Yd -= self.μ_prior(self.Xd).reshape((-1, 1))
            else:
                self.Yd -= self.trans(self.μ_prior(self.Xd).reshape(-1, 1))
        self.basis = explicit_basis
        if self.basis is not None:
            self.n_θ, self.Hd = self._basis(self.Xd)
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

    def __call__(self, Xi, infer_std=False, untransform=True,
                 kernel_terms='noiseless', exclude_mean=False, grad=False):
        return self.inference(Xi, infer_std=infer_std, kernel_terms=kernel_terms,
                              exclude_mean=exclude_mean, untransform=untransform, grad=grad)

    def _basis(self, X, grad=False):
        """Calculate the basis functions given independent variables."""
        if not (isinstance(self.basis, list) and
                all([[0, 1, 2, 3].count(val) == 1 for val in self.basis])):
            # TODO: Compare number of data points to degrees of freedom.
            raise InputError("GPI argument explicit_basis must be a list with: 0, 1, 2 and/or 3.",
                             self.basis)
        # TODO: implement an interface for user defined basis functions.
        # elif isinstance(self.basis, basis_callable):

        n_dims = self.n_dims
        n_pts = X.shape[0]
        n_θ = sum([int(prod(arange(n_dims, n_dims + p)) /
                      prod(arange(1, p + 1))) for p in self.basis])  # general
        H = empty((n_pts, n_θ))
        j = 0
        if 0 in self.basis:
            H[:, j] = 1.0
            j += 1
        if 1 in self.basis:
            H[:, j:(j + n_dims)] = X
            j += n_dims
        if 2 in self.basis:
            for ix in range(n_dims):
                for jx in range(ix, n_dims):
                    H[:, j] = X[:, ix] * X[:, jx]
                    j += 1
        if 3 in self.basis:
            for ix in range(n_dims):
                for jx in range(ix, n_dims):
                    for kx in range(jx, n_dims):
                        H[:, j] = X[:, ix] * X[:, jx] * X[:, kx]
                        j += 1
        if not grad:
            return n_θ, H
        else:
            Hp = zeros((n_pts, n_dims, n_θ))
            j = 0
            if 0 in self.basis:
                j += 1
            if 1 in self.basis:
                for ix in range(n_dims):
                    Hp[:, ix, j] = 1.0
                    j += 1
            if 2 in self.basis:
                for ix in range(n_dims):
                    for jx in range(ix, n_dims):
                        Hp[:, ix, j] += X[:, jx]
                        Hp[:, jx, j] += X[:, ix]
                        j += 1
            if 3 in self.basis:
                for ix in range(n_dims):
                    for jx in range(ix, n_dims):
                        for kx in range(jx, n_dims):
                            Hp[:, ix, j] += X[:, jx] * X[:, kx]
                            Hp[:, jx, j] += X[:, ix] * X[:, kx]
                            Hp[:, kx, j] += X[:, ix] * X[:, jx]
                            j += 1
            return n_θ, H, Hp

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
        except (InputError, LinAlgError):
            # Downshift to the slower, but more stable, eigen-decomposition.
            self.LKdd = eigh(self.Kdd)  # sorts eigenvalues small to large
            eig_solve = lambda Λ, V, b: (V @ ((V.T @ b).T / Λ).T)
            self.solve = lambda L, b: eig_solve(*L, b)
            Λ, V = self.LKdd
            # Eigenvalues that are too small require further intervention...
            i_keep = argmax(Λ > 1e-14 * Λ[-1])
            if i_keep < self.n_pts:
                warn('The data kernel was automatically modified to maintain positive definiteness '
                     '& avoid buildup of round-off error.',
                     RuntimeWarning)
                Λ = Λ[i_keep:]
                V = V[:, i_keep:]
                # Λ[:i_keep] = 1e-14  # Targeted noise
                self.LKdd = (Λ, V)
        self.α = self.solve(self.LKdd, self.Yd)
        if self.basis is not None:
            self.β = self.solve(self.LKdd, self.Hd)
            LinvΣθ = cho_factor_gen(self.Hd.T @ self.β)
            self.Σθ = cho_solve_gen(LinvΣθ, eye(self.n_θ))
            # ...results in an unnecessary matrix product: (V.T @ I).T,
            #    but it should not be very expensive.
            self.μΘ = cho_solve_gen(LinvΣθ, self.Hd.T @ self.α)
            HdμΘ = self.Hd @ self.μΘ
            self.βμΘ = self.solve(self.LKdd, HdμΘ)
        return self

    def posterior_φ(self, φ, grad=True, trans=True):
        """
        Negative log of the hyper-parameter posterior & its gradient.

        Arguments
        ---------
        φ:  array-1D,
            hyper parameters in an array for the minimization routine.
        grad:  bool or string (optional),
            when grad is True, also return lnP_grad.
        trans:  bool (optional),
            indicate whether the provided φ is in its transformed space.

        Returns
        -------
        lnP_neg:  float,
            negative log of the hyper-parameter posterior.
        lnP_grad:  array-1D (optional - depending on argument grad),
            gradient of lnP_neg with respect to each hyper-parameter.
        """
        if len(φ.shape) > 1:   # Corrects odd behavior of scipy's minimize
            φ = φ[0]           # Corrects odd behavior of scipy's minimize
        n_pts, n_φ = self.n_pts, self.kernel.n_φ
        K = self.kernel.Kφ(φ, self.Rdd, grad=grad, trans=trans)
        lnprior = self.kernel.ln_priors(φ, grad=grad, trans=trans)
        if grad:
            K, Kp = K
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
        α = solve(LK, self.Yd)

        lnP_neg = n_pts * HLOG2PI + ln_detK + 0.5 * self.Yd.T @ α - lnprior
        lnP_neg = squeeze(lnP_neg)

        if self.basis is not None:
            n_θ = self.n_θ
            β = solve(LK, self.Hd)
            invΣθ = self.Hd.T @ β
            LinvΣθ = cho_factor(invΣθ)
            Σθ = cho_solve(LinvΣθ, eye(n_θ))
            μΘ = cho_solve(LinvΣθ, self.Hd.T @ α)
            βμΘ = β @ μΘ
            lnP_neg -= squeeze(n_θ * HLOG2PI - sum(log(diag(LinvΣθ[0]))) +
                               0.5 * μΘ.T @ invΣθ @ μΘ)

        if not grad:
            return lnP_neg
        # else grad:
        invK = solve(LK, eye(n_pts))
        invK_αα = invK - α @ α.T
        lnP_grad = empty(n_φ)
        for j in range(n_φ):
            lnP_grad[j] = 0.5 * sum(invK_αα.T * Kp[:, :, j]) - dlnprior[j]
        if self.basis is not None:
            Δ2 = βμΘ.T - 2 * α.T
            for j in range(n_φ):
                βKpβ = β.T @ Kp[:, :, j] @ β
                lnP_grad[j] -= 0.5*(sum(βKpβ.T * Σθ) + Δ2 @ Kp[:, :, j] @ βμΘ)
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

    # def inference(self, Xi, infer_std=False, grad=False, kernel_terms='noiseless',
    #               exclude_mean=False, untransform=True):

    #     # Independent variables
    #     if Xi.ndim == 1:
    #         Xi = Xi.reshape((-1, 1))
    #     if Xi.ndim != 2 or Xi.shape[1] != self.n_dims:
    #         raise InputError("GPI object argument Xi must be a 2D array "
    #                          "(length of 2nd index must match that of Xd.)", Xi)
    #     n_dat, n_dims = self.Xd.shape
    #     n_inf = Xi.shape[0]

    #     # Mixed i-d kernel & inference of posterior mean
    #     Rid = radius(Xi, self.Xd, self.xscale)

    #     Kid = self.kernel(Rid, sum_terms=kernel_terms, grad=grad)
    #     if grad:
    #         Kid, Kid_grad = Kid

    #     if self.basis is None or exclude_mean:
    #         μ_post = Kid @ self.α
    #     else:
    #         μ_post = Kid @ (self.α - self.βμΘ)
    #         n_θ, *Hi = self._basis(Xi, grad=grad)
    #         if not grad:
    #             Hi = Hi[0]
    #         else:
    #             Hi, Hpi = Hi

    #         μ_post += Hi @ self.μΘ

    #     if grad:
    #         μ_post_grad = empty((n_inf, n_dims))
    #         if self.basis is None or exclude_mean:
    #             for i in range(n_dims):
    #                 μ_post_grad[:, i] = (Kid_grad[:, :, i] @ self.α).reshape(-1)
    #                 μ_post_grad[:, i] /= self.xscale[i]
    #         else:
    #             for i in range(n_dims):
    #                 μ_post_grad[:, i] = (Kid_grad[:, :, i] @ (self.α - self.βμΘ)).reshape(-1)
    #                 μ_post_grad[:, i] /= self.xscale[i]
    #             μ_post_grad[:, :] += (Hpi @ self.μΘ).reshape(μ_post_grad.shape)

    #     # Dependent variable
    #     if self.μ_prior is not None and not exclude_mean:
    #         μi = self.μ_prior(Xi).reshape((-1, 1))
    #         if self.trans is None or not untransform:
    #             μ_post = μ_post.reshape(μi.shape) + μi
    #         else:
    #             μ_post = μ_post.reshape(μi.shape) + self.trans(μi)

    #     # Inference of posterior covariance
    #     if infer_std:
    #         Rii = radius(Xi, Xi, self.xscale)
    #         Kii = self.kernel(Rii, sum_terms=kernel_terms)
    #         Σ_post = Kii - Kid @ self.solve(self.LKdd, Kid.T)
    #         if self.basis is not None:
    #             A = Hi - Kid @ self.β
    #             Σ_post += A @ (self.Σθ @ A.T)
    #         σ2_post = maximum(0.0, diag(Σ_post))
    #         σ_post = sqrt(σ2_post).reshape((-1, 1))

    #     # Inverse transformation of the dependent variable
    #     if self.trans is not None and untransform:
    #         if not grad:
    #             μ_post = self.trans(μ_post, inverse=True)
    #         else:
    #             μ_post, μ_post_grad = self.trans(μ_post, inverse=True, grad_z=μ_post_grad)
    #         if infer_std:
    #             σ_post = [self.trans(μ_post - σ_post, inverse=True),
    #                       self.trans(μ_post + σ_post, inverse=True)]
    #             σ_post = [σ_post[0] - μ_post, μ_post - σ_post[1]]

    #     if grad:
    #         μ_post = μ_post, μ_post_grad

    #     # TODO: Should we return μ_post as a 1-D array (rather than 2-D)?
    #     if not infer_std:
    #         return μ_post
    #     elif infer_std == 'covar':
    #         return μ_post, Σ_post
    #     else:
    #         return μ_post, σ_post

    def inference(self, Xi, kernel_terms='noiseless', infer_std=False,
                      exclude_mean=False, untransform=True, grad=False):
        """
        Make inferences (interpolation or regression) at specified locations. Limited to a single
        value of each hyper-parameters. Invoked when the GPI object is called as a function.

        Arguments
        ---------
        Xi:  array-2D,
            independent variables - where to make inferences. The first index is for multiple
            inferences, and second index must match that of the argument Xd when calling GPI.
        kernel_terms:  'noiseless', 'all', int or list of ints (optional),
            Applies only when kernel argument to GPI is a KernelSum:
            if 'noiseless', infer an underlying smoothly correlated regression function by assuming
                            that any uncorrelated Noise term in the specified kernel applies to the
                            observed values (Yd) only and not to the inferred values (Yi);
            if 'all', use all of the terms in the specified kernel (eg. simulate new observations);
            if int or list of ints, then use only that indexed subset of terms in order;
            Note: including a Noise term at this stage is not compatible with gradient inference.
        infer_std:  bool or 'covar' (optional),
            if True, return the inferred standard deviation;
            if 'covar', return the full posterior covariance matrix.
        exclude_mean:  bool (optional),
            if False include prior mean and basis functions, otherwise don't.
        untransform:  bool (optional),
            if False, any inverse transformation is suppressed.
        grad:  bool (optional),
            whether to return the inferred gradient of the dependent variable.

        Returns
        -------
        μ_post:  array-1D,
            inferred mean/median/mode at each location in the argument Xi
            (for a non-linear inverse transform, this value represents only the median —
             and only for monotonic transforms).
        μ_post_grad:  array-2D (optional — depending on grad),
            inferred mean/median/mode of the gradient at each location in Xi.
        σ_post:  array - 1D or 2D or list (optional - depending on infer_std),
            inferred standard deviation or full covariance (for an inverse transformation, returns
            the distance from μ_post to both ends of the 68.2% inner percentile — lower then upper).
        σ_post_grad:  array-2D (optional — depending on both infer_std & grad),
            inferred standard deviation of the gradient, or full covariance.

        Raises
        ------
        InputError:
            an exception is thrown for incompatible format of any inputs.

        Note
        ----
        If μ_prior was specified for GPI class object, that function will be applied to Xi data.
        """
        # Independent variables
        if isinstance(Xi, float) or isinstance(Xi, int) and self.Xd.shape[1] == 1:
            Xi = array([[Xi]], dtype='f8')  # a 1D problem at a single point
        elif Xi.ndim == 1 and self.n_dims == 1:
            Xi = Xi.copy().reshape((-1, 1))  # a 1D problem at n points
        elif Xi.ndim == 1 and self.n_dims == Xi.shape[0]:
            Xi = Xi.copy().reshape((1, -1))  # an nD problem at a single point
        elif Xi.ndim != 2 or Xi.shape[1] != self.n_dims:
            raise InputError("GPI object argument Xi must be a 2D array "
                             "(length of 2nd index must match that of Xd.)", Xi)
        n_dat, n_dims = self.Xd.shape
        n_inf = Xi.shape[0]

        # Mixed i-d kernel & inference of posterior mean
        Rid = radius(Xi, self.Xd, self.xscale)
        Kid = self.kernel(Rid, sum_terms=kernel_terms, grad=grad)

        if grad is False:
            if self.basis is None or (self.basis is not None and exclude_mean):
                μ_post = (Kid @ self.α).reshape(-1)
                if infer_std is False:
                    if self.trans is None or not untransform:
                        if self.μ_prior is None or exclude_mean:
                            return μ_post  # Options 0.0.0.0.0 & 0.0.0.0.0.a checked
                        else:
                            return μ_post + self.μ_prior(Xi).reshape(-1)  # Option 0.0.0.0.1a checked
                    else:
                        if self.μ_prior is None:
                            warn(trans_warn_μ, RuntimeWarning)
                            return self.trans(μ_post, inverse=True) # Option 0.0.0.1b.0 checked
                        elif not exclude_mean:
                            μ_prior_trans = self.μ_prior(Xi).reshape(-1)
                            return self.trans(μ_post + μ_prior_trans, inverse=True) # Options 0.0.0.1a.1a, 0.0.0.1b.1a & 0.0.0.1b.1b checked
                        else:
                            raise InputError("Inference parameter exclude_mean is incompatible"
                                             " with transformation & untransform.", exclude_mean)
                else:
                    Rii = radius(Xi, Xi, self.xscale)
                    Kii = self.kernel(Rii, sum_terms=kernel_terms)
                    Σ_post = Kii - Kid @ self.solve(self.LKdd, Kid.T)
                    if self.basis is not None:
                        A = Hi - Kid @ self.β
                        Σ_post += A @ (self.Σθ @ A.T)
                    if infer_std is True:
                        σ2_post = maximum(0.0, diag(Σ_post))
                        σ_post = sqrt(σ2_post).reshape(-1)
                        std_out = σ_post
                    elif infer_std == 'covar':
                        std_out = Σ_post
                    if self.trans is None or not untransform:
                        if self.μ_prior is None or exclude_mean:
                            return μ_post, std_out  # Options 0.0.1a.0.0 & 0.0.1b.0.0 checked
                        else:
                            μ_prior = self.μ_prior(Xi).reshape(-1)
                            return μ_post + μ_prior, std_out  # Options 0.0.1a.0.1a & 0.0.1a.1a.1a checked
                    else:
                        if self.μ_prior is None and not infer_std == 'covar':
                            warn(trans_warn_μ + trans_warn_σ, RuntimeWarning)
                            std_out = [self.trans(μ_post - std_out, inverse=True),
                                       self.trans(μ_post + std_out, inverse=True)]
                            μ_post = self.trans(μ_post, inverse=True)
                            std_out = array([μ_post - std_out[0], std_out[1] - μ_post])
                            return μ_post, std_out  # Option 0.0.1a.1b.0 checked
                        elif not exclude_mean and not infer_std == 'covar':
                            μ_prior_trans = self.μ_prior(Xi).reshape(-1)
                            μ_post = self.trans(μ_post + μ_prior_trans, inverse=True)
                            warn(trans_warn_μ + trans_warn_σ, RuntimeWarning)
                            std_out = [self.trans(μ_post - std_out, inverse=True),
                                       self.trans(μ_post + std_out, inverse=True)]
                            μ_post = self.trans(μ_post, inverse=True)
                            std_out = array([μ_post - std_out[0], std_out[1] - μ_post])
                            return μ_post, std_out  # Option 0.0.1a.1b.1a checked
                        elif infer_std == 'covar':
                            raise InputError("Inference parameter infer_std=='covar' is"
                                             " incompatible with transformation & untransform.",
                                             infer_std)  # Option 0.0.1b.1b.1a checked
                        else:
                            raise InputError("Inference parameter exclude_mean is incompatible"
                                             " with transformation & untransform.", exclude_mean)  # Option 0.0.1a.1b.1b checked
            else:
                μ_post = (Kid @ (self.α - self.βμΘ)).reshape(-1)
                n_θ, *Hi = self._basis(Xi)
                Hi = Hi[0]
                μ_post += (Hi @ self.μΘ).reshape(-1)
                if infer_std is False:
                    if self.μ_prior is None:
                        if self.trans is None or not untransform:
                            return μ_post  # Options 0.1.0.0.0 & 0.1.0.0.0.a checked
                        else:
                            return self.trans(μ_post, inverse=True) # Option 0.1.0.1b.0 checked
                    else:
                        μ_prior = self.μ_prior(Xi).reshape(-1)
                        if self.trans is None or not untransform:
                            return μ_post + μ_prior  # Options 0.1.0.0.1a & 0.1.0.1a.1a checked
                        else:
                            μ_post = μ_post + self.trans(μ_prior)
                            return self.trans(μ_post, inverse=True) # Option 0.1.0.1b.1a checked
                else:
                    Rii = radius(Xi, Xi, self.xscale)
                    Kii = self.kernel(Rii, sum_terms=kernel_terms)
                    Σ_post = Kii - Kid @ self.solve(self.LKdd, Kid.T)
                    A = Hi - Kid @ self.β
                    Σ_post += A @ (self.Σθ @ A.T)
                    if infer_std is True:
                        σ2_post = maximum(0.0, diag(Σ_post))
                        σ_post = sqrt(σ2_post).reshape(-1)
                        std_out = σ_post
                    elif infer_std == 'covar':
                        std_out = Σ_post
                    if self.μ_prior is None:
                        # Note: the basis conditional, above, addresses the exclude_mean option.
                        if self.trans is None or not untransform:
                            return μ_post, std_out  # Options 0.1.1a.0.0 & 0.1.1b.0.0 checked
                            # (Options 0.1.1a.1a.0 & 0.1.1b.1a.0 unchecked, but should work.)
                        else:
                            warn(trans_warn_μ + trans_warn_σ, RuntimeWarning)
                            std_out = [self.trans(μ_post - std_out, inverse=True),
                                       self.trans(μ_post + std_out, inverse=True)]
                            μ_post = self.trans(μ_post, inverse=True)
                            std_out = array([μ_post - std_out[0], std_out[1] - μ_post])
                            return μ_post, std_out  # Option 0.1.1a.1b.0 checked
                            # (Option 0.1.1b.1b.0 unchecked, but should work.)
                    else:
                        μ_prior = self.μ_prior(Xi).reshape(-1)
                        if self.trans is None or not untransform:
                            μ_post += μ_prior
                            return μ_post, std_out  # Options 0.1.1a.0.1a & 0.1.1a.1a.1a checked
                            # (Options 0.1.1b.0.1a & 0.1.1b.1a.1a unchecked, but should work.)
                        else:
                            μ_post += self.trans(μ_prior)
                            μ_post = self.trans(μ_post, inverse=True)
                            warn(trans_warn_μ + trans_warn_σ, RuntimeWarning)
                            std_out = [self.trans(μ_post - std_out, inverse=True),
                                       self.trans(μ_post + std_out, inverse=True)]
                            μ_post = self.trans(μ_post, inverse=True)
                            std_out = array([μ_post - std_out[0], std_out[1] - μ_post])
                            return μ_post, std_out  # Option 0.1.1a.1b.1a checked
                            # (Option 0.1.1b.1b.1a unchecked, but should work.)
        else:
            Kid, Kgd = Kid
            μ_post_grad = empty((n_inf, n_dims))
            if self.basis is None or (self.basis is not None and exclude_mean):
                μ_post = (Kid @ self.α).reshape(-1)
                for i in range(n_dims):
                    μ_post_grad[:, i] = (Kgd[i, :, :] @ self.α).reshape(-1)
                    μ_post_grad[:, i] /= self.xscale[i]
                if infer_std is False:
                    if self.trans is None or not untransform:
                        if self.μ_prior is None or exclude_mean:
                            return μ_post, μ_post_grad  # Options 1.0.0.0.0 & 1.0.0.0.0.a checked
                        else:
                            μ_prior, μ_prior_grad = self.μ_prior(Xi, grad=True)
                            μ_post += μ_prior.reshape(-1)
                            μ_post_grad += μ_prior_grad
                            return μ_post, μ_post_grad  # Options 1.0.0.0.1a & 1.0.0.1a.1a checked
                    else:
                        if self.μ_prior is None:
                            return self.trans(μ_post, inverse=True, grad_z=μ_post_grad)  # Option 1.0.0.1b.0 checked
                        elif not exclude_mean:
                            μ_prior_trans, μ_prior_grad_trans = self.μ_prior(Xi, grad=True)
                            μ_post += μ_prior_trans.reshape(-1)
                            μ_post_grad += μ_prior_grad_trans
                            return self.trans(μ_post, inverse=True, grad_z=μ_post_grad)  # Option 1.0.0.1b.1a checked
                        else:
                            raise InputError("Inference parameter exclude_mean is incompatible"
                                             " with transformation & untransform.", exclude_mean)  # Option 1.0.0.1b.1b checked
                else:
                    raise NotImplementedError("Although some pieces are in place, inferring variance"
                                              " & gradients simultaneously is not fully implemented")
                    Rii = radius(Xi, Xi, self.xscale)
                    Kii, Kgi, Kgg = self.kernel(Rii, sum_terms=kernel_terms, grad='gg')
                    # TODO: Add a warning that combined kernels are not implemented for gradient inference with uncertainty.
                    Kii_both = empty((n_inf * (1 + n_dims), n_inf * (1 + n_dims)))
                    Kii_both[:n_inf, :n_inf] = Kii
                    Kii_both[n_inf:, :n_inf] = Kgi.reshape((n_inf * n_dims, n_inf))
                    Kii_both[:n_inf, n_inf:] = Kgi.reshape((n_inf * n_dims, n_inf)).T
                    Kii_both[n_inf:, n_inf:] = Kgg
                    Kid_both = empty((n_inf * (1 + n_dims), n_inf))
                    Kid_both[:n_inf] = Kid
                    Kid_both[n_inf:] = Kgd.reshape((n_inf * n_dims, n_inf))
                    Σ_post = Kii_both - Kid_both @ self.solve(self.LKdd, Kid_both.T)
                    # TODO: when infer_std == 'covar', then manage the full covariance matrix.
                    # if self.basis is not None:
                    #     A = Hi - Kid @ self.β
                    #     Σ_post += A @ (self.Σθ @ A.T)
                    #     # TODO: add the corresponding contribution to the gradient's covariance using Hpi which was defined on line 461.

                    if infer_std is True:
                        σ2_post = maximum(0.0, diag(Σ_post))
                        σ_post = sqrt(σ2_post).reshape(-1)
                        std_out = σ_post
                    elif infer_std == 'covar':
                        # TODO: Current progress. Adding covariance of the gradient to the kernel.
                        std_out = Σ_post
                    # TODO: Calculate the std of the gradient
                    if self.trans is None or not untransform:
                        if self.μ_prior is None or exclude_mean:
                            return μ_post, μ_post_grad, std_out, std_grad_out
                        else:
                            μ_prior, μ_prior_grad = self.μ_prior(Xi, grad=True)
                            μ_post += μ_prior.reshape(-1)
                            μ_post_grad += μ_prior_grad
                            return  μ_post, μ_post_grad, std_out, std_grad_out
                    else:
                        if self.μ_prior is None:
                            μ_post, μ_post_grad = self.trans(μ_post, inverse=True, grad_z=μ_post_grad)
                            # TODO: Handle the transformation of the inferred std. Maybe provide a warning for True and 'not implemented' error for 'covar'.
                        elif not exclude_mean:
                            μ_prior_trans, μ_prior_grad_trans = self.μ_prior(Xi, grad=True)
                            μ_post += μ_prior_trans.reshape(-1)
                            μ_post_grad += μ_prior_grad_trans
                            μ_post, μ_post_grad = self.trans(μ_post, inverse=True, grad_z=μ_post_grad)
                            # TODO: Handle the transformation of the inferred std. Maybe provide a warning for True and 'not implemented' error for 'covar'.
                        else:
                            raise InputError("inference parameter exclude_mean is incompatible"
                                             "with transformation & untransform — ", exclude_mean)
                        return μ_post, μ_post_grad, std_out, std_grad_out
            else:
                μ_post = Kid @ (self.α - self.βμΘ)
                n_θ, *Hi = self._basis(Xi, grad=True)
                Hi, Hpi = Hi
                μ_post += Hi @ self.μΘ
                for i in range(n_dims):
                    μ_post_grad[:, i] = (Kgd[i, :, :] @ (self.α - self.βμΘ)).reshape(-1)
                    μ_post_grad[:, i] /= self.xscale[i]
                μ_post_grad[:, :] += (Hpi @ self.μΘ).reshape(μ_post_grad.shape)
                if infer_std is False:
                    if self.μ_prior is None:
                        if self.trans is None or not untransform:
                            return μ_post, μ_post_grad
                        else:
                            return self.trans(μ_post, inverse=True, grad_z=μ_post_grad)
                    else:
                        μ_prior, μ_prior_grad = self.μ_prior(Xi, grad=True)
                        if self.trans is None or not untransform:
                            μ_post += μ_prior.reshape((-1, 1))
                            μ_post_grad += μ_prior_grad
                            return μ_post, μ_post_grad
                        else:
                            μ_prior_trans, μ_prior_grad_trans = self.trans(μ_prior, grad_z=μ_prior_grad)
                            μ_post += μ_prior_trans
                            μ_post_grad += μ_prior_grad_trans
                            μ_post, μ_post_grad = self.trans(μ_post, inverse=True, grad_z=μ_post_grad)
                            return μ_post, μ_post_grad
                else:
                    raise NotImplementedError("Although some pieces are in place, inferring variance"
                                              " & gradients simultaneously is not fully implemented")
                    Rii = radius(Xi, Xi, self.xscale)
                    Kii = self.kernel(Rii, sum_terms=kernel_terms)
                    Σ_post = Kii - Kid @ self.solve(self.LKdd, Kid.T)
                    A = Hi - Kid @ self.β
                    Σ_post += A @ (self.Σθ @ A.T)
                    if infer_std is True:
                        σ2_post = maximum(0.0, diag(Σ_post))
                        σ_post = sqrt(σ2_post).reshape((-1, 1))
                        std_out = σ_post
                    elif infer_std == 'covar':
                        std_out = Σ_post
                    # TODO: Calculate the std of the gradient
                    if self.μ_prior is None:
                        if self.trans is None or not untransform:
                            return μ_post, μ_post_grad
                        else:
                            μ_post, μ_post_grad = self.trans(μ_post, inverse=True, grad_z=μ_post_grad)
                            # TODO: Handle the transformation of the inferred std.
                            return μ_post, μ_post_grad, std_out, std_grad_out
                    else:
                        μ_prior, μ_prior_grad = self.μ_prior(Xi, grad=True)
                        μ_prior = μ_prior.reshape((-1, 1))
                        if self.trans is None or not untransform:
                            μ_post += μ_prior
                            μ_post_grad += μ_prior_grad
                            return μ_post, μ_post_grad, std_out, std_grad_out
                        else:
                            μ_prior_trans, μ_prior_grad_trans = self.trans(μ_prior, grad_z=μ_prior_grad)
                            μ_post += μ_prior_trans
                            μ_post_grad += μ_prior_grad_trans
                            μ_post, μ_post_grad = self.trans(μ_post, inverse=True, grad_z=μ_post_grad)
                            # TODO: Handle the transformation of the inferred std.
                            return μ_post, μ_post_grad, std_out, std_grad_out


    def sample(self, Xs, n_samples=1, kernel_terms='noiseless', exclude_mean=False, grad=False):
        """
        Sample the Gaussian process at specified locations.

        Arguments
        ---------
        Xs:  array-2D,
            independent variables - where to sample. First index is for multiple (correlated)
            inferences, and second index must match that of the argument Xd from GPI.__init__.
        n_samples: int (optional),
            allows the calculation of multiple samples at once.
        kernel_terms:  'noiseless', 'all', int or list of ints (optional),
            Applies only when kernel argument to GPI is a KernelSum:
            if 'noiseless', infer an underlying smoothly correlated regression function by assuming
                            that any uncorrelated Noise term in the specified kernel applies to the
                            observed values (Yd) only and not to the inferred values (Yi);
            if 'all', use all of the terms in the specified kernel (eg. simulate new observations);
            if int or list of ints, then use only that indexed subset of terms in order;
            Note: including a Noise term at this stage is not compatible with gradient inference.
        exclude_mean:  bool (optional),
            if False include prior mean and basis functions, otherwise don't.
        grad:  bool (optional),
            whether to return the inferred gradient of the dependent variable.

        Returns
        -------
        Ys:  array-2D,
            sample value from the poster at each location in Xs.
        μpost_grad:  array-2D,
            mean gradient of the posterior at each location in Xs.

        Raises
        ------
        InputError:
            an exception is thrown for incompatible format of any inputs.
        """
        n_pts = Xs.shape[0]
        μpost, Σ = self.inference(Xs, infer_std='covar', untransform=False, kernel_terms=kernel_terms,
                                  exclude_mean=exclude_mean, grad=grad)
        if grad:
            μpost, μpost_grad = μpost

        Z = randn(n_pts, n_samples)
        Λ, V = eigh(Σ)
        Λ = maximum(Λ, 0)
        Ys = empty((n_pts, n_samples))
        for i in range(n_samples):
            Ys[:, i] = μpost[:, 0] + V @ (sqrt(Λ) * Z[:, i])
        if self.trans is not None:
            Ys = self.trans(Ys, inverse=True)
        if not grad:
            return Ys
        else:
            return Ys, μpost_grad

    def loo(self, return_data=False, plot_results=False):
        """
        Perform a leave-one-out cross-validation analysis on the data following the procedure
        outlined by Sacks & Welch at SAMSI 2010.

        Arguments
        ---------
        return_data:  bool (optional),
            indicate whether the analysis results should be returned
            (predicted Y values, predicted std., standardized residuals).
        plot_results:  bool (optional),
            indicate whether plots of the analysis results should be created.

        Raises
        ------
        ValidationError:
            an exception is thrown when any standardized residuals are greater than three.
        """
        Xd_red, Yd_red = empty((self.n_pts-1, self.n_dims)), empty((self.n_pts-1, 1))
        Cov_copy = deepcopy(self.kernel)
        Yd_pred, Yd_std = empty(self.n_pts), empty(self.n_pts)
        for i in range(self.n_pts):
            Xd_red[:i, :], Xd_red[i:, :] = self.Xd[:i, :], self.Xd[i+1:, :]
            Yd_red[:i, :], Yd_red[i:, :] = self.Yd[:i, :], self.Yd[i+1:, :]
            tmpGP = GPI(Xd_red, Yd_red, Cov_copy, Xscaling=self.xscale,
                        Ymean=self.μ_prior, explicit_basis=self.basis,
                        transform=self.trans)
            tmp_out = tmpGP(self.Xd[i, :].reshape(1, -1), infer_std=True)
            Yd_pred[i], Yd_std[i] = tmp_out[0][0], tmp_out[1][0]
        std_res = (self.Yd[:, 0] - Yd_pred) / Yd_std
        if plot_results:
            from matplotlib.pyplot import figure, plot, xlabel, ylabel
            figure()
            plot(std_res, 'o')
            plot([0, self.n_pts + 1], [-2.0, -2.0],
                 color='orange', linestyle='--', linewidth=2.0)
            plot([0, self.n_pts + 1], [+2.0, +2.0],
                 color='orange', linestyle='--', linewidth=2.0)
            plot([0, self.n_pts + 1], [-3.0, -3.0],
                 color='red', linestyle='-', linewidth=2.0)
            plot([0, self.n_pts + 1], [+3.0, +3.0],
                 color='red', linestyle='-', linewidth=2.0)
            xlabel('Index of Provided Value')
            ylabel('Standard Residual')
            figure()
            plot(self.Yd[:, 0], Yd_pred, 'o')
            plot([amin(self.Yd), amax(self.Yd)],
                 [amin(self.Yd), amax(self.Yd)],
                 color='black', linestyle='-', linewidth=2.0)
            xlabel('Provided Value')
            ylabel('Predicted Value')
        N2 = count_nonzero(abs(std_res) > 2.0)
        N3 = count_nonzero(abs(std_res) > 3.0)
        if N3 > 0:
            raise ValidationError(f"GPI object failed its cross validation - of {self.n_pts:d} "
                                  f"data points, {N3:d} had std. resid. values greater than 3.")
        if return_data:
            return Yd_pred, Yd_std, std_res
        else:
            return None


@jit(nopython=True)
def radius(x, y, scale):
    """Calculate the distance matrix (radius)."""
    # Started with scipy.spatial.distance.cdist(X, Y, 'seuclidean', V=xscale);
    # Next, used numpy (with tile);
    # Currently prefer the simplicity of element operations with numba's jit.
    n_xpts, n_dims = x.shape
    n_ypts, n_dims = y.shape
    r = empty((n_dims, n_xpts, n_ypts))
    for k in range(n_dims):
        for i in range(n_xpts):
            for j in range(n_ypts):
                r[k, i, j] = (x[i, k] - y[j, k]) / scale[k]
    return r


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


trans_warn_μ = ('Using untransform=True, limits the meaning of the returned μ values to the median'
                ' (& only for monotonic transformations).  ')
trans_warn_σ = ('infer_std=True limits the meaning of the returned σ values to the 68.2% inner'
                ' percentile region.  For other quantiles use the sample method.')

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
        self.n_pts = n_dims
        self.N3 = N3
        self.N2 = N2


if __name__ == "__main__":
    from numpy import array, linspace, meshgrid
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from pyregress import GPI,  Noise, SquareExp, RatQuad, LogNormal, Probit

    # TODO: Provide examples that verify!

    # Example 1  (1D regression w/ six data points & a known kernel):
    # setup...
    # Xd = array([0.5, 2.7, 3.6, 6.8, 5.7, 3.4]).reshape((-1, 1))
    Xd = array([0.5, 2.7, 3.6, 6.8, 5.7, 3.4])  # for 1D problems, this form is accepted
    Yd = array([0.0, 1.0, 1.2, 0.5, 0.8, 1.16])
    # Yd = array([0.0, 1.0, 1.2, 0.5, 0.8, 1.16]).reshape((-1, 1))  # also accepted
    myGPI = GPI(Xd, Yd, Noise(w=0.05) + SquareExp(w=2.5, l=2.0))
    # inference...
    # print(myGPI(1.6))  # no gradient
    xi = 1.6  # for 1D problems w/ a single point, any of these three forms are accepted
    # xi = array([1.6])
    # xi = array([1.6]).reshape((-1, 1))
    yi, yi_grad = myGPI(xi, grad=True)  # infer the gradient as well
    # create a tangent line (for plotting)...
    δ = 0.25  # half width of the tangent line
    Xig = (xi + δ * array([-1.0, 1.0])).reshape(-1, 1)
    Yig = (yi + yi_grad * δ * array([-1.0, 1.0])).reshape(-1, 1)
    # across x for visual effect...
    Xi = linspace(0, 7.5, 200)
    Yi, Yistd = myGPI(Xi, infer_std=True)
    # plot it all...
    plt.figure(figsize=(7, 5))
    plt.plot(Xd, Yd, 'ko', label='observed data')
    plt.plot(Xi, Yi, 'b-', linewidth=2.0, label='inferred mean')
    plt.fill_between(Xi, Yi-Yistd, Yi+Yistd, alpha=0.25,
                    label='inferred mean +/- std')
    plt.plot(xi, yi, 'ro', label='example regression point')
    plt.plot(Xig, Yig, 'r-', linewidth=3.0, label='inferred slope')
    plt.xlim(Xi[0], Xi[-1])
    plt.title('Example #1  (1D regression w/ 6 data points & a known kernel)',
            fontsize=12)
    plt.xlabel('independent variable, X', fontsize=12)
    plt.ylabel('dependent variable, Y', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)

    # Example 2  (probit interpolation in 2D w/ a linear basis & 6 data points):
    # setup...
    Xd = array([[0.00, 0.00], [0.50,-0.10], [1.00, 0.00],
                [0.15, 0.50], [0.85, 0.50], [0.50, 0.85]])
    Yd = array([0.10, 0.30, 0.60, 0.70, 0.90, 0.90])
    K = RatQuad(w=0.6, l=LogNormal(guess=0.3, σ=0.25), α=1)
    myGPI = GPI(Xd, Yd, K, explicit_basis=[0, 1], transform=Probit, Xscaling='range')
    # inference...
    print('Example 2: optimized value of the hyper-parameter:', myGPI.kernel.get_φ())
    xi = array([[0.10, 0.10], [0.50, 0.42]])
    yi = myGPI(xi)
    print('x = ', xi)
    print('y = ', yi)
    # across all (x1, x2) for visual effect...
    Ni = (30, 32)
    xi_1 = linspace(-0.2, 1.2, Ni[0])
    xi_2 = linspace(-0.3, 1.0, Ni[1])
    Xi_1, Xi_2 = meshgrid(xi_1, xi_2, indexing='ij')
    Xi = array([Xi_1.reshape(-1), Xi_2.reshape(-1)]).T
    Yi, Yistd = myGPI(Xi, infer_std=True)
    # plot it all...
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(Xi_1, Xi_2, Yi.reshape(Ni),
                    alpha=0.75, linewidth=0.5, cmap=mpl.colormaps['jet'], rstride=1, cstride=1)
    ax.plot_surface(Xi_1, Xi_2, (Yi + Yistd[0]).reshape(Ni),
                    alpha=0.25, linewidth=0.25, color='black', rstride=1, cstride=1)
    ax.plot_surface(Xi_1, Xi_2, (Yi - Yistd[1]).reshape(Ni),
                    alpha=0.25, linewidth=0.25, color='black', rstride=1, cstride=1)
    ax.scatter(Xd[:, 0], Xd[:, 1], Yd, c='black', s=35)
    ax.set_zlim([0.0, 1.0])
    ax.set_title('Example #2  (probit interpolation in 2D w/ a linear basis)',
                fontsize=12)
    ax.set_xlabel('independent variable, X1', fontsize=12)
    ax.set_ylabel('independent variable, X2', fontsize=12)
    ax.set_zlabel('dependent variable, Y', fontsize=12)

    plt.show()
