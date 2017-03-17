# -*- coding: utf-8 -*-
"""
Docstring for the pyregress module - needs work.

For basic usage see the documentation in the GPP class.
This docstring covers more advanced topics.
Performance:
  Calculation time will greatly depend on which Blas/Lapack libs are used.
  Some default python/numpy/scipy packages are based on unoptimized libs
  (including linux repositories), but anaconda now provides optimized libs.
Reading the code and development:
  Notation used throughout the code:
    X => independent variables,
    Y => dependent variable,
    Z => transformed dependent variable,
    d => data values (observations),
    i => inferred values,
    s => sampled values,
    μ => Gaussian expected values (for various values),
    K => kernel values (prior covariance matrix),
    φ => hyper(unknown)-parameters of the kernel,
    R => distance (radius) in independent variable space,
    Σ => other covariance matricies,
    H => explicit basis functions evaluated at X,
    Θ => linear coefficients to the basis functions,
    p => derivative (prime) of a variable,
    L => lower diagonal of a Cholesky factorization.
"""
# Created Sep 2013
# @author: Sean T. Smith
__all__ = ['GPP', 'InputError', 'ValidationError']

from copy import deepcopy
# from termcolor import colored  # may not work on windows
from numpy import (ndarray, array, empty, zeros, ones, eye, diag, tile,
                   sum, std, amin, amax, maximum, count_nonzero, abs, sqrt, log)
from numpy import pi as π
from numpy.linalg.linalg import LinAlgError, svd
from numpy.random import randn
from scipy.linalg import cho_factor, cho_solve
from pyregress.kernels import *
from pyregress.transforms import *
from pyregress.multi_newton import *
from pyregress.rprop import rprop

HLOG2PI = 0.5*log(2*π)


class GPP:
    """
    Doctring for the GPP class - needs work.

    Examples
    --------
    >>> from numpy import array
    >>> from pyregress import GPP, Noise, SquareExp, RatQuad
    >>> Xd = array([[0.1], [0.3], [0.6]])
    >>> Yd = array([[0.0], [1.0], [0.5]])
    >>> myGPP = GPP( Xd, Yd, Noise(w=0.1) + SquareExp(w=1.0, l=0.3) )
    >>> print myGPP( np.array([[0.2]]) )
    [[ 0.52641732]]

    >>> Xd = array([[0.00, 0.00], [0.50,-0.10], [1.00, 0.00],
    ...             [0.15, 0.50], [0.85, 0.50], [0.50, 0.85]])
    >>> Yd = array([[0.10], [0.30], [0.60], [0.70], [0.90], [0.90]])
    >>> myGPP = GPP(Xd, Yd, RatQuad(w=0.6, l=0.3, alpha=1.0),
    ...             explicit_basis=[0, 1], transform='Probit')
    >>> print myGPP( np.array([[0.10, 0.10], [0.50, 0.42]]) )
    [[ 0.22770558]
     [ 0.78029862]]
    """
    def __init__(self, Xd, Yd, Cov, Xscaling=None,
                 Ymean=None, explicit_basis=None, transform=None,
                 optimize_hp=True):
        """
        Create a GPP object and prepare for inference.

        Arguments
        ---------
        Xd:  array-2D,
            independent-variable observed values. First dimension is for
            multiple observations, second dimension for multiple variables.
        Yd:  array-1D [or column-shaped 2D],
            dependent-variable observed values - same length as the first
            dimension of Xd.
        Cov:  Kernel object,
            prior covariance kernel. Options include: Noise, SquareExp,
            GammaExp, RatQuad, or the sum of any of these.
        Xscaling:  string or array-1D (optional),
            pre-scaling of the independent variables (kernel anisotropy).
            Range scaling: 'range'; standard deviation scaling: 'std'; and
            manual scaling: array (same length as the second dimension of Xd).
        Ymean:  function (optional),
            prior mean of the dependent variable at Xd & Xi. It must accept
            input in form of Xd, and must provide output the same shape as Yd.
            If omitted, a prior mean of zero is assumed.
        explicit_basis:  list of ints (optional),
            explicit basis functions are specified by any combination of the
            integers: 0, 1, 2 - each corresponding to its polynomial order.
        transform:  string or BaseTransform object (optional)
            specify a dependent variable transformation with the name of a
            BaseTransform class (as a string) or a BaseTransform object.
            Options include: Logarithm, Logit, Probit, or ProbitBeta.

        Raises
        ------
        InputError:
            an exception is thrown for incompatible format of any inputs.
        """

        # Independent variables
        if Xd.ndim == 1:
            self.Xd = Xd.reshape((-1, 1))
        elif Xd.ndim == 2:
            self.Xd = Xd
        else:
            raise InputError("GPP argument Xd must be a 2D array.", Xd)
        self.Nd, self.Nx = Xd.shape
        if Xscaling is None:
            self.xscale = ones(self.Nx)
        elif Xscaling == 'range':
            self.xscale = Xd.max(0) - Xd.min(0)
        elif Xscaling == 'std':
            self.xscale = std(Xd, axis=0)
        elif Xscaling.shape == (self.Nx,):
            self.xscale = Xscaling
        else:
            raise InputError("GPP argument Xscaling must be one of: " +
                             "False, True, 'range', 'std', or 1D array " +
                             "(same length as the 2nd dim. of Xd)", Xscaling)

        # Dependent variable
        if Yd.shape[0] != self.Nd:
            raise InputError("GPP argument Yd must have the same length as " +
                             "the 1st dim. of Xd.", Yd)
        self.Yd = Yd.reshape((-1, 1)).copy()
        if transform is None:
            self.trans = None
        elif isinstance(transform, str):
            self.trans = eval(transform + '(self.Yd)')
            self.Yd = self.trans(self.Yd)
        elif isinstance(transform, BaseTransform):
            self.trans = transform
            self.Yd = self.trans(self.Yd)
        else:
            raise InputError("GPP argument transform must be BaseTransform " +
                             "class (string of name) or object.", transform)
        self.μ_prior = Ymean
        if self.μ_prior is not None:
            if self.trans is None:
                self.Yd -= self.μ_prior(Xd).reshape((-1, 1))
            else:
                self.Yd -= self.trans(self.μ_prior(Xd).reshape(-1, 1))
        self.basis = explicit_basis
        if self.basis is not None:
            self.Nθ, self.Hd = self._basis(Xd)
        # Kernel (prior covariance)
        self.kernel = Cov
        if not isinstance(Cov, Kernel):
            raise InputError("GPP argument Cov must be a Kernel object.", Cov)

        # Do as many calculations as possible in preparation for the inference
        # -- Create a separate function for the following? --
        self.Rdd = self._radius(self.Xd, self.Xd)
        if (self.kernel.Nφ > 0) and optimize_hp:
            self.maximize_hyper_posterior(optimize_hp)
        self.Kdd = self.kernel(self.Rdd, block_diag=True)
        self.LKdd = cho_factor_gen(self.Kdd)
        self.invKdd_Yd = cho_solve_gen(self.LKdd, self.Yd)
        if self.basis is not None:
            self.invKdd_Hd = cho_solve_gen(self.LKdd, self.Hd)
            LΣθ = cho_factor_gen(self.Hd.T @ self.invKdd_Hd)
            self.Σθ = cho_solve_gen(LΣθ, eye(self.Nθ))
            self.Θ = cho_solve_gen(LΣθ, self.Hd.T @ self.invKdd_Yd)
            HdΘ = self.Hd @ self.Θ
            self.invKdd_HdΘ = cho_solve_gen(self.LKdd, HdΘ)

    def __call__(self, Xi, infer_std=False, untransform=True, sum_terms=True,
                 exclude_mean=False, grad=False):
        return self.inference(Xi, infer_std, untransform, sum_terms,
                              exclude_mean, grad)

    def _basis(self, X, grad=False):
        """Calculate the basis functions given independent variables."""
        if not (isinstance(self.basis, list) and
                all([[0, 1, 2].count(entry) == 1 for entry in self.basis])):
            # TODO: also check if there is less data than degrees of freedom.
            raise InputError("GPP argument explicit_basis must be a list " +
                             "with: 0, 1, and/or 2.", self.basis)
        # TODO: implement an interface for user defined basis functions.
        # elif isinstance(self.basis, basis_callable):

        N = X.shape[0]
        Nθ = sum(self.Nx**array(self.basis))
        H = empty((N, Nθ))
        j = 0
        if self.basis.count(0):
            H[:, j] = 1.0
            j += 1
        if self.basis.count(1):
            H[:, j:j+self.Nx] = X
            j += self.Nx
        if self.basis.count(2):
            for ix in range(self.Nx):
                for jx in range(self.Nx):
                    H[:, j] = X[:, ix] * X[:, jx]
                    j += 1
        if not grad:
            return Nθ, H

        Hp = zeros((N, self.Nx, Nθ))
        j = 0
        if self.basis.count(0):
            j += 1
        if self.basis.count(1):
            for ix in range(self.Nx):
                Hp[:, ix, j] = 1.0
                j += 1
        if self.basis.count(2):
            for ix in range(self.Nx):
                for jx in range(self.Nx):
                    Hp[:, ix, j] += X[:, jx]
                    Hp[:, jx, j] += X[:, ix]
                    j += 1
        if not grad == 'Hess':
            return Nθ, H, Hp

        Hpp = zeros((N, self.Nx, self.Nx, Nθ))
        j = 0
        if self.basis.count(0):
            j += 1
        if self.basis.count(1):
            j += self.Nx
        if self.basis.count(2):
            for ix in range(self.Nx):
                Hpp[:, ix, ix, j] = 2.0
                j += self.Nx + 1
        return Nθ, H, Hp, Hpp

    def _radius(self, X, Y):
        """Calculate the distance matrix (radius)."""
        # Previously used: cdist(X, Y, 'seuclidean',V=self.xscale),
        # which required: from scipy.spatial.distance import cdist.
        Nx, Ny = X.shape[0], Y.shape[0]
        Rk = empty((Nx, Ny, self.Nx))
        for k in range(self.Nx):
            Rk[:, :, k] = tile(X[:, [k]], (1, Ny)) - tile(Y[:, [k]].T, (Nx, 1))
            if isinstance(self.xscale, ndarray):
                Rk[:, :, k] /= self.xscale[k]
        return Rk

    def hyper_posterior(self, params=None, grad=True):
        """
        Negative log of the hyper-parameter posterior & its gradient.

        Arguments
        ---------
        params:  array-1D,
            hyper parameters in an array for the minimization routine.
        φ_mapped:  array-1D,
            hyper parameter values that map to self.Kernel.
        grad:  bool or string (optional),
            when grad is True, must return lnP_grad,
            when grad is 'Hess', must also return lnP_hess.

        Returns
        -------
        lnP_neg:  float,
            negative log of the hyper-parameter posterior.
        lnP_grad:  array-1D (optional - depending on argument grad),
            gradient of lnP_neg with respect to each hyper-parameter.
        lnP_hess:  array-2D (optional - depending on argument grad),
            Hessian matrix (2nd derivatives) of lnP_neg.
        """
        Nd, Nφ = self.Nd, self.kernel.Nφ
        if not grad:
            K = self.kernel(self.Rdd, block_diag=True)
            lnprior = self.kernel._ln_priors(params)
        elif grad != 'Hess':
            K, Kp = self.kernel(self.Rdd, grad_hp=grad, block_diag=True)
            lnprior, dlnprior = self.kernel._ln_priors(params, grad=grad)
        else:
            K, Kp, Kpp = self.kernel(self.Rdd, grad_hp=grad, block_diag=True)
            lnprior, dlnprior, d2lnprior = \
                self.kernel._ln_priors(params, grad=grad)
        try:
            LK = cho_factor(K)
        except LinAlgError as e:
            print("GPP method hyper_posterior failed to factor the " +
                  "data kernel. This is most often an indication that the " +
                  "minimization routine is not converging.")
            print('Current hyper-parameter values: ')
            print(repr(params))
            raise e
        α = cho_solve(LK, self.Yd)
        lnP_neg = (float(self.Nd) * HLOG2PI + sum(log(diag(LK[0]))) +
                   0.5*self.Yd.T @ α - lnprior)

        if self.basis is not None:
            Nθ = self.Nθ
            β = cho_solve(LK, self.Hd)
            try:
                LΣθ = cho_factor(self.Hd.T @ β)
            except:
                print('debug')
            Σθ = cho_solve(LΣθ, eye(self.Nθ))
            Θ = cho_solve(LΣθ, self.Hd.T @ α)
            βΘ = β @ Θ
            lnP_neg -= (float(self.Nθ)*HLOG2PI - sum(log(diag(LΣθ[0]))) +
                        0.5 * Θ.T @ self.Hd.T @ βΘ)
        if not grad:
            return lnP_neg

        # grad == True or 'Hess':
        invK = cho_solve(LK, eye(Nd))
        invK_αα = invK - α @ α.T
        lnP_grad = empty(Nφ)
        for j in range(Nφ):
            lnP_grad[j] = 0.5*sum(invK_αα.T * Kp[:, :, j]) - 1.0*dlnprior[j]
        if self.basis is not None:
            Δ2 = βΘ.T - 2.0*α.T
            for j in range(Nφ):
                βKpβ = β.T @ Kp[:, :, j] @ β
                lnP_grad[j] -= 0.5*(sum(βKpβ.T * Σθ) + Δ2 @ Kp[:, :, j] @ βΘ)
        if grad != 'Hess':
            return lnP_neg, lnP_grad

        # grad == 'Hess':
        invK_Kp, ααKp = empty((Nd, Nd, Nφ)), empty((Nd, Nd, Nφ))
        for j in range(Nφ):
            invK_Kp[:, :, j] = invK @ Kp[:, :, j]
            αKp = α.T @ Kp[:, :, j]
            ααKp[:, :, j] = 2.0 * α @ αKp
        lnP_hess = empty((Nφ, Nφ))
        for j in range(Nφ):
            for i in range(j + 1):
                lnP_hess[i, j] = 0.5*sum(
                    invK_αα.T * Kpp[:, :, i, j] -
                    invK_Kp[:, :, i].T * invK_Kp[:, :, j] +
                    ααKp[:, :, i].T * invK_Kp[:, :, j]) - d2lnprior[i, j]
                lnP_hess[j, i] = lnP_hess[i, j]
        if self.basis is not None:
            Δ1 = βΘ.T - α.T
            βSβ_2invK = β @ Σθ @ β.T - 2.0 * invK
            βKp, ΘβKp = empty((Nθ, Nd, Nφ)), empty((Nd, Nφ))
            Δ1Kpβ, Δ2Kp = empty((Nθ, Nφ)), empty((Nd, Nφ))
            for j in range(Nφ):
                βKp[:, :, j] = β.T @ Kp[:, :, j]
                ΘβKp[:, j] = βΘ.T @ Kp[:, :, j]
                Δ1Kpβ[:, j] = Δ1 @ Kp[:, :, j] @ β
                Δ2Kp[:, j] = Δ2 @ Kp[:, :, j]
            for j in range(Nφ):
                for i in range(Nφ):
                    big_mess = (β.T @ Kpp[:, :, i, j] @ β +
                                βKp[:, :, i] @ βSβ_2invK @ βKp[:, :, j].T)
                    lnP_hess[i, j] -= 0.5*(sum(big_mess.T * Σθ) +
                                           Δ2 @ Kpp[:, :, i, j] @ βΘ -
                                           2.0*Δ2Kp[:, i] @ invK @ ΘβKp[:, j].T +
                                           2.0*Δ1Kpβ[:, i] @ Σθ @ Δ1Kpβ[:, j].T)
        return lnP_neg, lnP_grad, lnP_hess

    def maximize_hyper_posterior(self, optimize_φ):
        """
        Find the maximum of the hyper-parameter posterior.

        Arguments
        ---------
       optimize_φ - specify if printing of hyper-parameters is desired
        """

        # Setup hyper-parameters & map values from a single array
        all_hyper, bounds = self.kernel._map_hyper()
        lo, hi = [], []
        [(lo.append(bounds[i+i]), hi.append(bounds[2*i+1]))
            for i in range(int(len(bounds)/2))]

        # Perform minimization
       # if optimize_φ == 'print':
       #     MD_Newton(self.hyper_posterior, all_hyper,
       #               options={'tol': 1e-6, 'maxiter': 200, 'bounds': (lo, hi),
       #                        'repress text': False})
       # else:
       #     MD_Newton(self.hyper_posterior, all_hyper,
       #               options={'tol': 1e-6, 'maxiter': 200, 'bounds': (lo, hi),
       #                        'repress text': True})
        all_hyper = rprop(self.hyper_posterior, all_hyper)

        all_hyper, bounds = self.kernel._map_hyper(all_hyper, unmap=True)
        return self, all_hyper

    def inference(self, Xi, infer_std=False, untransform=True, sum_terms=True,
                  exclude_mean=False, grad=False):
        """
        Make inferences (interpolation or regression) at specified locations.
        Limited to a single value of each hyper-parameters.
        This method is invoked when the GPP object is called as a function.

        Arguments
        ---------
        Xi:  array-2D,
            independent variables - where to make inferences. The first
            dimension is for multiple inferences, and second dimension must
            match the second dimension of the argurment Xd from __init__.
        infer_std:  bool or 'covar' (optional),
            if True, return the inferred standard deviation;
            if 'covar', return the full posterior covariance matrix.
        untransform:  bool (optional),
            if False, any inverse transformation is suppressed.
        sum_terms:  bool, int or list of ints (optional),
            if int or list of ints, then use only this subset of terms of the
            sum kernel, by index (Cov in GPP.__init__ must be a KernelSum).
            If True, all terms are included.
        exclude_mean:  bool (optional),
            if False include prior mean and basis functions, otherwise don't.
        grad:  bool or 'Hess' (optional),
            if True or 'Hess' return the gradient of the dependent variable,
            if 'Hess' also return the second derivatives.

        Returns
        -------
        μ_post:  array-2D,  TODO: change to a 1D-array?
            inferred mean at each location in the argument Xi.
        Σ_post: array-2D or list (optional - depending on infer_std),
            inferred standard deviation or full covariance
            (for any inverse transformation, both the positive and negative
            standard deviations are returned - in that order).

        Raises
        ------
        InputError:
            an exception is thrown for incompatible format of any inputs.

        Note
        ----
        If μ_prior was specified for GPP class object, this function
            will also be applied to Xi data.
        """

        # TODO: calculation of the posterior mean of gradient and Hessian.

        # Independent variables
        if Xi.ndim == 1:
            Xi = Xi.reshape((-1, 1))
        if Xi.ndim != 2 or Xi.shape[1] != self.Nx:
            raise InputError("GPP object argument Xi must be a 2D array " +
                             "(2nd dimension must match that of Xd.)", Xi)

        # Mixed i-d kernel & inference of posterior mean
        Rid = self._radius(Xi, self.Xd)

        if grad is False:
            Kid = self.kernel(Rid, block_diag=False, sum_terms=sum_terms)
        elif grad is True:
            Kid, Kid_grad = self.kernel(Rid, block_diag=False,
                                        sum_terms=sum_terms, grad_r=grad)
        else:
            Kid, Kid_grad, Kid_hess = self.kernel(Rid, block_diag=False,
                                                  sum_terms=sum_terms,
                                                  grad_r=grad)

        if self.basis is None or exclude_mean:
            μ_post = Kid @ self.invKdd_Yd
        else:
            μ_post = Kid @ (self.invKdd_Yd - self.invKdd_HdΘ)
            if grad is False:
                Nθ, Hi = self._basis(Xi)
            elif grad is True:
                Nθ, Hi, Hpi = self._basis(Xi, grad=grad)
            else:
                Nθ, Hi, Hpi, Hppi = self._basis(Xi, grad=grad)

            μ_post += Hi @ self.Θ

        if grad is True or grad is 'Hess':
            μ_post_grad = empty((Rid.shape[0], Rid.shape[2]))
            if self.basis is None or exclude_mean:
                for i in range(Rid.shape[2]):
                    μ_post_grad[:, i] = \
                        (Kid_grad[:, :, i] @ self.invKdd_Yd).reshape(-1)
            else:
                for i in range(Rid.shape[2]):
                    μ_post_grad[:, i] = (Kid_grad[:, :, i] @ (self.invKdd_Yd -
                                              self.invKdd_HdΘ)).reshape(-1)
                μ_post_grad[:, :] += \
                    (Hpi @ self.Θ).reshape(μ_post_grad.shape)

        if grad is 'Hess':
            μ_post_hess = empty((Rid.shape[0], Rid.shape[2], Rid.shape[2]))
            if self.basis is None or exclude_mean:
                for i in range(Rid.shape[2]):
                    for j in range(Rid.shape[2]):
                        μ_post_hess[:, i, j] = \
                            (Kid_hess[:, :, i, j] @ self.invKdd_Yd).reshape(-1)
            else:
                for i in range(Rid.shape[2]):
                    for j in range(Rid.shape[2]):
                        μ_post_hess[:, i, j] = \
                            (Kid_hess[:, :, i, j] @ (self.invKdd_Yd -
                                                     self.invKdd_HdΘ)).reshape(-1)
                μ_post_hess[:, :, :] += \
                    (Hppi @ self.Θ).reshape(μ_post_hess.shape)

        # Dependent variable
        if self.μ_prior is not None and not exclude_mean:
            μi = self.μ_prior(Xi).reshape((-1, 1))
            if self.trans is None or not untransform:
                μ_post = μ_post.reshape(μi.shape) + μi
            else:
                μ_post = μ_post.reshape(μi.shape) + self.trans(μi)

        # Inference of posterior covariance
        if infer_std:
            Rii = self._radius(Xi, Xi)
            Kii = self.kernel(Rii, block_diag=True, sum_terms=sum_terms)
            Σ_post = Kii - Kid @ cho_solve_gen(self.LKdd, Kid.T)
            if self.basis is not None:
                A = Hi - Kid @ self.invKdd_Hd
                Σ_post += A @ (self.Σθ @ A.T)
            σ2_post = maximum(0.0, diag(Σ_post))
            σ_post = sqrt(σ2_post).reshape((-1, 1))

        # Inverse transformation of the dependent variable
        if self.trans is not None and untransform:
            if infer_std:
                σ_post = [self.trans(μ_post - σ_post, inverse=True),
                          self.trans(μ_post + σ_post, inverse=True)]
                μ_post = self.trans(μ_post, inverse=True)
                σ_post = [σ_post[0] - μ_post, μ_post - σ_post[1]]
            else:
                μ_post = self.trans(μ_post, inverse=True)

        if grad is True:
            μ_post = μ_post, μ_post_grad
        if grad is 'Hess':
            μ_post = μ_post, μ_post_grad, μ_post_hess

        if not infer_std:
            return μ_post
        elif infer_std == 'covar':
            return μ_post, Σ_post
        else:
            return μ_post, σ_post

    def sample(self, Xs, Nsamples=1, sum_terms=True, exclude_mean=False,
               grad=False):
        """
        Sample the Gassian process at specified locations.

        Arguments
        ---------
        Xs:  array-2D,
            independent variables - where to sample. First dimension is for
            multiple inferences, and second dimension must match the second
            dimension of the argument Xd from GPP.__init__.
        Nsamples: int (optional),
            allows the calculation of multiple samples at once.
        sum_terms:  bool, int or list of ints (optional),
            if int or list of ints, then use only this subset of terms of the
            sum kernel, by index (Cov in GPP.__init__ must be a KernelSum).
            For regression, standard use includes all terms except the noise.
            If True, all terms are included.
        exclude_mean:  bool (optional),
            if False include prior mean and basis functions, otherwise don't.

        Returns
        -------
        Ys:  array-2D,
            sample value at each location in the argument Xs.

        Raises
        ------
        InputError:
            an exception is thrown for incompatible format of any inputs.
        """
        Nx = Xs.shape[0]
        Ys_post, Cov = self.inference(Xs, infer_std='covar',
                                      sum_terms=sum_terms,
                                      exclude_mean=exclude_mean,
                                      grad=grad)
        if grad is True:
            Ys_post, Ys_post_grad = Ys_post
        if grad is 'Hess':
            Ys_post, Ys_post_grad, Ys_post_hess = Ys_post

        Z = randn(Nx, Nsamples)
        U, S, V = svd(Cov)
        sig = U @ diag(sqrt(S))
        Ys = empty((Nx, Nsamples))
        for i in range(Nsamples):
            Ys[:, i] = sig @ Z[:, i]
            Ys[:, i] += Ys_post[:, 0]
        if self.trans is not None:
            Ys = self.trans(Ys, inverse=True)
        if grad is False:
            return Ys
        if grad is True:
            return Ys, Ys_post_grad
        if grad is 'Hess':
            return Ys, Ys_post_grad, Ys_post_hess

    def loo(self, return_data=False, plot_results=False):
        """
        Perform a leave-one-out cross-validation analysis on the data
        following the procedure outlined by Sacks and Welch at SAMSI 2010.

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
            an exception is thrown when any standardized residues are
            greater in magnitude than three.
        """
        Xd_red, Yd_red = empty((self.Nd-1, self.Nx)), empty((self.Nd-1, 1))
        Cov_copy = deepcopy(self.kernel)
        Yd_pred, Yd_std = empty(self.Nd), empty(self.Nd)
        for i in range(self.Nd):
            Xd_red[:i, :], Xd_red[i:, :] = self.Xd[:i, :], self.Xd[i+1:, :]
            Yd_red[:i, :], Yd_red[i:, :] = self.Yd[:i, :], self.Yd[i+1:, :]
            tmpGP = GPP(Xd_red, Yd_red, Cov_copy, Xscaling=self.xscale,
                        Ymean=self.μ_prior, explicit_basis=self.basis,
                        transform=self.trans)
            tmp_out = tmpGP(self.Xd[i, :].reshape(1, -1), infer_std=True)
            Yd_pred[i], Yd_std[i] = tmp_out[0][0], tmp_out[1][0]
        std_res = (self.Yd[:, 0] - Yd_pred) / Yd_std
        if plot_results:
            from matplotlib.pyplot import figure, plot, xlabel, ylabel
            figure()
            plot(std_res, 'o')
            plot([0, self.Nd + 1], [-2.0, -2.0],
                 color='orange', linestyle='--', linewidth=2.0)
            plot([0, self.Nd + 1], [+2.0, +2.0],
                 color='orange', linestyle='--', linewidth=2.0)
            plot([0, self.Nd + 1], [-3.0, -3.0],
                 color='red', linestyle='-', linewidth=2.0)
            plot([0, self.Nd + 1], [+3.0, +3.0],
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
            raise ValidationError("GPP object failed its cross validation -" +
                                  " of %d data points, %d had std. resid." +
                                  " values greater than 3.0",
                                  self.Nd, N3, N2)
        if return_data:
            return Yd_pred, Yd_std, std_res
        else:
            return None


def cho_factor_gen(A, lower=False, **others):
    """Generalize scipy's cho_factor to handle arrays of length zero."""
    if A.size == 0:
        return empty(A.shape), lower
    else:
        try:
            return cho_factor(A, lower=lower, **others)
        except LinAlgError as e:
            print("GPP method __init__ failed to factor data kernel." +
                  "This often indicates that X has near duplicates or " +
                  "the noise kernel has too small of weight.")
            raise e


def cho_solve_gen(C, b, **others):
    """Generalize scipy's cho_solve to handle arrays of length zero."""
    if C[0].size == 0:
        return empty(b.shape)
    else:
        return cho_solve(C, b, **others)


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
                input argument that is the source of error. Provided so
                the value can be reported when the error is caught.
        """
        self.args = (msg,)
        self.input_argument = input_argument


class ValidationError(GPError):
    def __init__(self, msg, Nd=None, N3=None, N2=None):
        """
        Initialize a ValidationError when the interpolation or regression
        is failing its cross validation.

        Arguments
        ---------
            msg:  string,
                explanation of the error.
            Nd:  integer (optional),
                Number of cross validation data points.
            N3:  integer (optional),
                Number of points that have abs(std_err) > 3.0
            N2:  integer (optional),
                Number of points that have abs(std_err) > 2.0
        """
        self.args = (msg % (Nd, N3),)
        self.Nd = Nd
        self.N3 = N3
        self.N2 = N2


if __name__ == "__main__":
    from numpy import linspace, hstack, meshgrid, rot90
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from pyregress import *

    # TODO: Examples that provide verification!

    # Example 1:
    # Simple case, 1D with five data points and one regression point
    print('Example 1:')
    Xd1 = array([[0.1], [0.3], [.36], [0.65], [.57]])
    Yd1 = array([[0.0], [1.0], [1.2], [0.5], [.6]])
    xi1 = array([[0.2]])
    myGPP1 = GPP(Xd1, Yd1, Noise(w=0.1) + SquareExp(w=0.75, l=0.25))
    yi1, yi1_grad = myGPP1(xi1, grad=True, sum_terms=[1])
    print('Example 1:')
    print('x = ', xi1, ',  y = ', yi1)
    yi1_, yi1_grad_, yi1_hess_ = myGPP1(Xd1, grad='Hess', sum_terms=[1])

    # Example 2:
    # 2D with six data points and two regression points
    Xd2 = array([[0.00, 0.00], [0.50, -0.10], [1.00, 0.00],
                 [0.15, 0.50], [0.85, 0.50], [0.50, 0.85]])
    Yd2 = array([[0.10], [0.30], [0.60], [0.70], [0.90], [0.90]])
    K2 = RatQuad(w=0.6, l=LogNormal(guess=0.3, std=0.25), alpha=1.0)
    myGPP2 = GPP(Xd2, Yd2, K2, explicit_basis=[0, 1], transform='Probit')
    print('Example 2:')
    print('Optimized value of the hyper-parameters:', myGPP2.kernel.get_φ())
    xi2 = array([[0.1, 0.1], [0.5, 0.42]])
    yi2, yi2_grad = myGPP2(xi2, grad=True)
    print('x = ', xi2)
    print('y = ', yi2)

    # Figures to support the examples
    # fig. example 1
    Xi1 = linspace(0.0, 0.75, 200)
    Yi1, Yi1std = myGPP1(Xi1, infer_std=True, sum_terms=1)
    Yi1, Yi1std = Yi1.reshape(-1), Yi1std.reshape(-1)

    Xig1 = (xi1 + 0.025*array([-1.0, 1.0])).reshape(-1, 1)
    Yig1 = (yi1 + yi1_grad*0.025*array([-1.0, 1.0])).reshape(-1, 1)
    Xdg1 = Xd1 + 0.025*array([-1.0, 1.0])
    Ydg1 = yi1_ + yi1_grad_*0.025*array([-1.0, 1.0])


    fig1 = plt.figure(figsize=(5, 3), dpi=150)
    p1, = plt.plot(Xd1, Yd1, 'ko', label='Data')
    p2, = plt.plot(Xi1, Yi1, 'b-', linewidth=2.0, label='Inferred mean')
    plt.fill_between(Xi1, Yi1-Yi1std, Yi1+Yi1std, alpha=0.25)
    p3 = plt.Rectangle((0.0, 0.0), 1.0, 1.0, facecolor='blue',
                       alpha=0.25, label='Uncertainty (one std.)')
    p4, = plt.plot(xi1, yi1, 'ro', label='Example regression point')
    p5 = plt.plot(Xig1, Yig1, 'r-', linewidth=3.0, label='Inferred slope')
    p6 = plt.plot(Xdg1[0, :], Ydg1[0, :], 'r-', Xdg1[1, :], Ydg1[1, :], 'r-',
                  Xdg1[4, :], Ydg1[4, :], 'r-', linewidth=3.0)
    fig1.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)
    plt.title('Example 1', fontsize=16)
    plt.xlabel('Independent Variable, X', fontsize=12)
    plt.ylabel('Dependent Variable, Y', fontsize=12)
    plt.legend(loc='best', numpoints=1, prop={'size': 8})

    # fig. example 2
    Ni = (30, 30)
    xi_1 = linspace(-0.2, 1.2, Ni[0])
    xi_2 = linspace(-0.2, 1.0, Ni[1])
    Xi_1, Xi_2 = meshgrid(xi_1, xi_2, indexing='ij')
    Xi2 = hstack([Xi_1.reshape((-1, 1)), Xi_2.reshape((-1, 1))])
    Yi2, Yi2std = myGPP2(Xi2, infer_std=True)

    fig = plt.figure(figsize=(7, 5), dpi=150)
    ax = fig.gca(projection='3d')
    ax.plot_surface(Xi_1, Xi_2, Yi2.reshape(Ni), alpha=0.75,
                    linewidth=0.5, cmap=mpl.cm.jet, rstride=1, cstride=1)
    ax.plot_surface(Xi_1, Xi_2, (Yi2+Yi2std[0]).reshape(Ni), alpha=0.25,
                    linewidth=0.25, color='black', rstride=1, cstride=1)
    ax.plot_surface(Xi_1, Xi_2, (Yi2-Yi2std[1]).reshape(Ni), alpha=0.25,
                    linewidth=0.25, color='black', rstride=1, cstride=1)
    ax.scatter(Xd2[:, 0], Xd2[:, 1], Yd2, c='black', s=35)
    ax.set_zlim([0.0, 1.0])
    ax.set_title('Example 2', fontsize=16)
    ax.set_xlabel('Independent Variable, X1', fontsize=12)
    ax.set_ylabel('Independent Variable, X2', fontsize=12)
    ax.set_zlabel('Dependent Variable, Y', fontsize=12)

    fig3 = plt.figure(figsize=(5, 3), dpi=150)
    plt.pcolor(rot90(myGPP1.Kdd, 1))
    plt.yticks([.5, 1.5, 2.5, 3.5, 4.5], [5, 4, 3, 2, 1])
    plt.xticks([.5, 1.5, 2.5, 3.5, 4.5], [1, 2, 3, 4, 5])
    plt.xlim([0, 5])
    plt.ylim([0, 5])
    plt.colorbar()
    plt.title('Eg. 2 - Covariance Matrix')
    plt.show()
