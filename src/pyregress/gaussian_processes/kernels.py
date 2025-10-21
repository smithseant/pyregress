# -*- coding: utf-8 -*-
"""
The Kernels, as used, provide a straightforward implementation of the equations they represent.
However, the handling of kernel parameters is less straight forward due to the necessity to provide
flexibility to GPI.  An overview of the approach:
   -The Kernel object contains a dict, `param_vals`, which stores the current value of all
    parameters (known or unknown) in their untransformed state (as the kernel uses the value).
    Isotropic or 1-D lengthscales are stored as a scalar, while independent lengthscales are
    stored as a list.
   -The Kernel object also contains a dict of independent hyper-prior objects, `param_priors`.
    Each of these objects has an attribute for the initial guess, `guess`, as well as
    callables for the prior, `__call__`, parameter  transformation function, `transformation`,
    and (when applicable) the inverse transformation, `inv_trans`.

Created Sep 2013  @authors: Sean T. Smith & Benjamin B. Schroeder
"""
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
import re
from numpy import (empty, full, arange, eye, expand_dims, tile,
                   sum, abs, sqrt, exp, log, pi as π)
from numba import jit
from .hyper_params import HyperPrior

# TODO: Add a periodic kernel (..but flexible for dims. that are not periodic.)
# TODO: I would love to add heuristics so users are required to specify less.


@jit(nopython=True)
def radius(x, y, scale):
    """Calculate the distance matrix (radius)."""
    # Started with scipy.spatial.distance.cdist(X, Y, 'seuclidean', V=xscale);
    # Next, used numpy (with tile);
    # Currently prefer the simplicity of element operations with numba's jit.
    n_xpts, n_dims = x.shape
    n_ypts, n_dims = y.shape
    r = empty((n_xpts, n_ypts, n_dims))
    for i in range(n_xpts):
        for j in range(n_ypts):
           for k in range(n_dims):
                r[i, j, k] = (x[i, k] - y[j, k]) / scale[k]
    return r


class Kernel(metaclass=ABCMeta):
    """
    Provide methods & an interface for kernels in the GPI class.

    Specific kernels will need to inherit this baseclass and define
    `__init__`, `__call__` & `Kφ` methods in the derived class.
    """
    def __init__(self, **kwargs):
        self.param_vals = dict()
        self.param_priors = dict()
        self.n_φ = 0
        for key, val in kwargs.items():
            if not isinstance(val, Iterable):
                if not isinstance(val, HyperPrior):
                    self.param_vals[key] = val
                else:
                    self.param_vals[key] = val.guess
                    if not val in self.param_priors:
                        self.param_priors[val] = key
                    elif not isinstance(self.param_priors[val], list):
                        self.param_priors[val] = [self.param_priors[val], key]
                    else:
                        self.param_priors[val].append(key)
                    self.n_φ += 1
            else:
                self.param_vals[key] = val
                self.param_priors[key] = [None] * len(val)
                for i in range(len(val)):
                    if isinstance(val[i], HyperPrior):
                        self.param_vals[key][i] = val[i].guess
                        self.param_priors[key][i] = val[i]
                        self.n_φ += 1

    def __add__(self, other):
        """Overload '+' so Kernel objects can be added."""
        if not isinstance(other, KernelSum):
            # Neither term is a KernelSum object, so create one.
            return KernelSum(self, other)
        else:
            # Combine with the existing KernelSum object.
            return other.__add__(self, self_on_right=True)

    def __mul__(self, other):
        """Overload '*' so Kernel objects can be multiplied."""
        if not isinstance(other, KernelProd):
            # Neither term is a KernelProd object, so create one.
            return KernelProd(self, other)
        else:
            # Combine with the existing KernelProd object.
            return other.__mul__(self, self_on_right=True)

    def iter_φ(self):
        """
        Provide an iterator for each prior & its parameter(s), φ, in order. Two complications
        prevent this from simply being done inline: 1st, length-scales can be nested in lists
        (use an `if` & a `for` loop); and 2nd, CombiningKernel nest an entire dict for each term
        (and must overload this method.)
        """
        for val in self.param_priors.values():
            if not isinstance(val, Iterable):
                yield val
            else:
                for el in [e for e in val if e]:
                    yield el

    def get_φ(self, trans=True):
        """
        Return the values of the current hyper parameters from `param_vals`.
        Arguments
        ---------
        trans: bool (optional),
            return the hyper-parameters in their transformed space (or as used in their kernel).
        Returns
        -------
        all_φ:  array-1D,
            current values, for each hyper parameter, from `param_vals`.
        """
        φ = empty(self.n_φ)
        iφ = 0
        for key, val in self.param_priors.items():
            if not isinstance(val, Iterable):
                if not trans or not val.transformation:
                    φ[iφ] = self.param_vals[key]
                else:
                    φ[iφ] = self.param_priors[key].transformation(self.param_vals[key])
                iφ += 1
            else:
                for i in range(len(val)):
                    if self.param_priors[key][i]:
                        if trans:
                            φ[iφ] = self.param_priors[key][i].trans(self.param_vals[key][i])
                        else:
                            φ[iφ] = self.param_vals[key][i]
                        iφ += 1
        return φ

    def ln_priors(self, φ=None, ret_grad=False, trans=False):
        """
        Calculate log of prior distributions for hyper-parameters.

        Arguments
        ---------
        φ: array-1D (optional),
            array of hyper-parameter values. When no values are provided, use the initial guess.
        ret_grad: bool (optional),
            when ret_grad is True also return dlnprior.
        trans: book (optional),
            indicate whether the input values of φ have been transformed and whether the gradients
            are wrt the transformed space.

        Returns
        -------
        lnprior: scalar value
            summation of values of log prior probabilities evaluated at values provided by params
        dlnprior: array-1D
            array of gradients of log prior probabilities evaluated at values provided by params
        """
        if φ is None:
            φ = [None] * self.n_φ
        lnprior = 0.0
        iφ = 0
        if not ret_grad:
            for key, val in self.param_priors.items():
                if not isinstance(val, Iterable):
                    lnprior += val(φ[iφ], trans=trans)
                    iφ += 1
                else:
                    for i in range(len(val)):
                        if self.param_priors[key][i]:
                            lnprior += val[i](φ[iφ], trans=trans)
                            iφ += 1
            return lnprior
        else:
            dlnprior = empty(self.n_φ)
            for key, val in self.param_priors.items():
                if not isinstance(val, Iterable):
                    lnP, dlnP = val(φ[iφ], ret_grad=ret_grad, trans=trans)
                    lnprior += lnP
                    dlnprior[iφ] = dlnP
                    iφ += 1
                else:
                    for i in range(len(val)):
                        if self.param_priors[key][i]:
                            lnP, dlnP = val[i](φ[iφ], ret_grad=ret_grad, trans=trans)
                            lnprior += lnP
                            dlnprior[iφ] = dlnP
                            iφ += 1
            return lnprior, dlnprior

    def φ2param(self, φ, trans=True, set=True):
        p = dict()
        iφ = 0
        for key, val in self.param_vals.items():
            if key not in self.param_priors:
                p[key] = val
            elif not isinstance(self.param_priors[key], Iterable):
                if not trans:
                    p[key] = φ[iφ]
                else:
                    p[key] = self.param_priors[key].invtr(φ[iφ])
                iφ += 1
            else:
                p[key] = [None] * len(self.param_priors[key])
                for i in range(len(self.param_priors[key])):
                    if not self.param_priors[key][i]:
                        p[key][i] = val[i]
                    else:
                        if not trans:
                            p[key][i] = φ[iφ]
                        else:
                            p[key][i] = self.param_priors[key][i].invtr(φ[iφ])
                        iφ += 1
        if set:
            self.param_vals = p
            return None
        else:
            return p

    @abstractmethod
    def __call__(self, R, i_grad=False, ii_grad=False, scale=None, **kwargs):
        """
        Calculate and return kernel values given the radius array.

        Arguments
        ---------
        R: array-3D,
            directional distance matrix (distance between combination of points in each direction).
        i_grad: bool (optional),
            whether the returned kernel also includes gradient terms for the first index with
            respect to each combination of first & third indices of `R`.
        ii_grad: bool (optional),
            whether the returned kernel also includes gradient terms for both indices with respect
            to each combination of the second & third indices of `R` (assumes `R` is symmetric).
        scale: array,
            scaling of the dimensions of x when calculating `R` (required when `i_grad is True`).
        kwargs: any additional options (opt_name=opt_value),
            specific options for specific kernels, otherwise ignored.

        Returns
        -------
        K: array-2D,
            kernel values - shape must match argument R.
        Kgrad: array-3D (optional - depending on argument grad),
            partial of kernel (first two dimensions) with respect to each
            hyper parameter (third dimension).
        """
        return

    @abstractmethod
    def Kφ(self, φ, R, ret_grad=False, trans=False, **kwargs):
        """
        Calculate and return kernel values given a vector of hyper parameters and the radius array.

        Arguments
        ---------
        φ: array-1D,
            array of hyper parameters (potentially transformed),
        R: array-3D,
            directional distance matrix (distance between combination of points in each direction).
        ret_grad: bool (optional),
            indicate whether to return the gradients with respect to the transformed hyper
            parameters as Kgrad,
        trans: bool (optional),
            indicate whether the input values of φ have been transformed and whether the gradients
            are wrt the transformed space,
        kwargs: any additional options (opt_name=opt_value),
            specific options for specific kernels, otherwise ignored.

        Returns
        -------
        K: array-2D,
            kernel values - shape must match argument R,
        Kgrad: array-3D (optional - depending on argument ret_grad),
            partial of kernel (first two dimensions) with respect to each hyper param. (3rd dim.).
        """
        return


def subclass_name(kern):
    return re.search(r"kernels.(\w+)", repr(type(kern))).group(1)


class CombiningKernel(Kernel):
    """
    This is a super class of KernelSum & KernelProd, created to avoid repetition of these methods.
    """
    def __init__(self, k1, k2):
        self.terms = [k1, k2]
        self.n_φ = k1.n_φ + k2.n_φ

    @property
    def param_vals(self):
        param_vals = dict()
        for kern in self.terms:
            param_vals[subclass_name(kern)] = kern.param_vals
        return param_vals

    @property
    def φdist(self):
        φ = dict()
        for kern in self.terms:
            φ[subclass_name(kern)] = kern.φdist
        return φ

    def iter_φ(self):
        for kern in self.terms:
            for φdist in kern.iter_φ():
                yield φdist

    def get_φ(self, trans=True):
        φ = empty(self.n_φ)
        iφ = 0
        for kern in self.terms:
            φ[iφ:(iφ + kern.n_φ)] = kern.get_φ(trans)
            iφ += kern.n_φ
        return φ

    def setparam_vals(self, φ, trans=True, set=True):
        param_vals = dict()
        iφ = 0
        for kern in self.terms:
            param_vals[subclass_name(kern)] = kern.setparam_vals(φ[iφ:(iφ + kern.n_φ)], trans, set)
            iφ += kern.n_φ
        if set:
            return None
        else:
            return param_vals

    def ln_priors(self, φ=None, ret_grad=False, trans=False):
        if φ is None:
            φ = [None] * self.n_φ
        lnprior = 0.0
        iφ = 0
        if not ret_grad:
            for kern in self.terms:
                lnprior += kern.ln_priors(φ[iφ:(iφ + kern.n_φ)], ret_grad, trans)
                iφ += kern.n_φ
            return lnprior
        else:
            dlnprior = empty(self.n_φ)
            for kern in self.terms:
                lnp, dlnp = kern.ln_priors(φ[iφ:(iφ + kern.n_φ)], ret_grad, trans)
                lnprior += lnp
                dlnprior[iφ:(iφ + kern.n_φ)] = dlnp
                iφ += kern.n_φ
            return lnprior, dlnprior


class KernelSum(CombiningKernel):
    """Provide a class that lists kernels to be added at evaluation."""
    def __add__(self, other, self_on_right=False):
        if isinstance(other, KernelSum):
            self.terms += other.terms
        elif isinstance(other, Kernel):
            if self_on_right:
                self.terms = [other] + self.terms
            else:
                self.terms += [other]
        self.n_φ += other.n_φ
        return self

    def __call__(self, R, sum_terms='all', **kwargs):
        if sum_terms == 'all':
            terms = self.terms
        elif sum_terms == 'noisefree':
            terms = [t for t in self.terms if not isinstance(t, Noise)]
        elif type(sum_terms) is list:
            terms = [self.terms[i] for i in sum_terms]
        elif type(sum_terms) is int:
            terms = [self.terms[sum_terms]]
        K = terms[0](R, **kwargs)
        for kern in terms[1:]:
            K += kern(R, **kwargs)
        return K


    def Kφ(self, φ, R, ret_grad=False, trans=False, **kwargs):
        ni, nj, n_xdims = R.shape
        if 'sum_terms' not in kwargs or kwargs['sum_terms'] == 'all':
            terms = self.terms
        elif kwargs['sum_terms'] == 'noiseless':
            terms = [t for t in self.terms if not isinstance(t, Noise)]
        elif type(kwargs['sum_terms']) is list:
            terms = [self.terms[i] for i in kwargs['sum_terms']]
        elif type(kwargs['sum_terms']) is int:
            terms = [self.terms[kwargs['sum_terms']]]
        if not ret_grad:
            K = full((ni, nj), 0, dtype='float64')
            iφ = 0
            for kern in terms:
                n_φ = kern.n_φ
                K += kern.Kφ(φ[iφ:(iφ + n_φ)], R, trans=trans, **kwargs)
                iφ += n_φ
            return K
        else:
            K = full((ni, nj), 0, dtype='float64')
            Kgrad = full((ni, nj, self.n_φ), 0, dtype='float64')
            iφ = 0
            for kern in terms:
                n_φ = kern.n_φ
                Kt, Kgrad[:, :, iφ:(iφ + n_φ)] = kern.Kφ(φ[iφ:(iφ + n_φ)], R, ret_grad, trans, **kwargs)
                K += Kt
                iφ += n_φ
            return K, Kgrad


class KernelProd(CombiningKernel):
    """Provide a class that lists kernels to be multiplied at evaluation."""
    def __mul__(self, other, self_on_right=False):
        if isinstance(other, KernelProd):
            self.terms += other.terms
        elif isinstance(other, Kernel):
            self.terms += [other]
        else:
            # TODO: Throw a not-implemented error!
            pass
        self.n_φ += other.n_φ
        return self

    def __call__(self, R, i_grad=False, ii_grad=False, **kwargs):
        if i_grad or ii_grad:
            raise InputError("Kernel product does not currently support radial gradients.")
        K = self.terms[0](R, **kwargs)
        for kern in self.terms[1:]:
            K *= kern(R, **kwargs)
        return K

    def Kφ(self, φ, R, ret_grad=False, trans=False, **kwargs):
        ni, nj, n_xdims = R.shape
        if not ret_grad:
            K = full((ni, nj), 1, dtype='float64')
            iφ = 0
            for kern in self.terms:
                n_φ = kern.n_φ
                K *= kern.Kφ(φ[iφ:iφ+n_φ], R, trans=trans, **kwargs)
                iφ += n_φ
            return K
        else:
            K = full(R.shape[1:], 1, dtype='float64')
            Kgrad = full((ni, nj, self.n_φ), 1, dtype='float64')
            iφ = 0
            for kern in self.terms:
                n_φ = kern.n_φ
                Kt, Kgt = kern.Kφ(φ[iφ:(iφ + n_φ)], R, ret_grad, trans, **kwargs)
                K *= Kt
                irange = range(iφ, iφ + n_φ)
                iother = range(0, iφ) + range(iφ + n_φ, self.n_φ)
                Kgrad[:, :, irange] *= Kgt
                Kgrad[:, :, iother] *= tile(expand_dims(Kt, 2), (1, 1, len(iother)))
                iφ += n_φ
            return K, Kgrad


class Noise(Kernel):
    r"""
    White noise kernel object.
    ..math::
        K(R, data; w) = w^2 * I, or a zero matrix based on the data, with the weight parameter, w,
        and a flag indicating inclusion or not.
    White noise is discontinuous.
    """
    def __init__(self, w):
        super().__init__(w=w)

    def __call__(self, R, i_grad=False, ii_grad=False, **kwargs):
        ni, nj, n_xdims = R.shape
        K = self.param_vals['w']**2 * eye(ni, nj)
        if not i_grad or ii_grad:
            return K
        else:
            raise InputError("Noise Kernel is not differentiable, separate kernels before "
                             "differentiation")

    def Kφ(self, φ, R, ret_grad=False, trans=False, **kwargs):
        ni, nj, n_xdims = R.shape
        p = self.setparam_vals(φ, trans=trans, set=False)
        w = p['w']
        w2 = w**2
        K0 = eye(ni, nj)
        if not ret_grad:
            return w2 * K0
        else:
            Kgrad = empty((ni, nj, self.n_φ))
            if 'w' in self.param_priors:
                if not trans:
                    Kgrad[:, :, 0] = 2 * w  * K0
                else:
                    Kgrad[:, :, 0] = 2 * w2 * K0
            return w2 * K0, Kgrad


class SquareExp(Kernel):
    r"""
    Squared-exponential kernel object.
    .. math::
        K(R; w, l) = w^2 * \exp( -1/2 *(R/l)^2 ),
    with the parameters of weight, w, and length, l. For multiple dimensions, the length can be a
    single value applied to all directions or it can be a list with a separate value in each
    direction. Squared-exponential is continuous and infinitely differentiable.
    """
    def __init__(self, w, l):
        super().__init__(w=w, l=l)

    def __call__(self, R, i_grad=False, ii_grad=False, s=None, **kwargs):
        ni, nj, n_xdims = R.shape
        w, ℓ = self.param_vals['w'], self.param_vals['l']
        if not isinstance(ℓ, Iterable):
            l = full(n_xdims, ℓ)
        else:
            l = ℓ
        K = empty((ni * (1 + (i_grad or ii_grad) * n_xdims), nj * (1 + ii_grad * n_xdims)))
        K[:ni, :nj] = w**2 * exp(-0.5 * ((R / l)**2).sum(axis=2))
        if i_grad or ii_grad:
            for k in range(n_xdims):
                lo, hi = ni * (k + 1), ni * (k + 2)
                K[lo:hi, :nj] = -R[:, :, k] / (s[k] * l[k]**2) * K[:ni, :nj]
            if ii_grad:
                K[:ni, nj:] = -K[ni:, :nj].T
                for ki in range(n_xdims):
                    i_lo, i_hi = ni * (ki + 1), ni * (ki + 2)
                    for kj in range(n_xdims):
                        δ = 1 if ki == kj else 0
                        j_lo, j_hi = nj * (kj + 1), nj * (kj + 2)
                        K[i_lo:i_hi, j_lo:j_hi] = ((δ - R[:, :, ki] / l[ki] * R[:, :, kj] / l[kj]) *
                                                   K[:ni, :nj] / (s[ki] * l[ki] * s[kj] * l[kj]))
        return K

    def Kφ(self, φ, R, ret_grad=False, trans=False, **kwargs):
        ni, nj, n_xdims = R.shape
        p = self.setparam_vals(φ, trans=trans, set=False)
        w, l_in = p['w'], p['l']
        if not isinstance(l_in, Iterable):
            l = full(n_xdims, l_in)
        else:
            l = l_in
        Rl2 = ((R / l)**2).sum(axis=2)
        K = w**2 * exp(-0.5 * Rl2)
        if not ret_grad:
            return K
        else:
            Kgrad = empty((ni, nj, self.n_φ,))
            iφ = 0
            if 'w' in self.param_priors:
                if not trans:
                    Kgrad[:, :, iφ] = 2 * K
                else:
                    Kgrad[:, :, iφ] = 2 * K
                iφ += 1
            if 'l' in self.param_priors:
                ld = self.param_priors['l']
                if not isinstance(ld, Iterable):
                    if not trans:
                        Kgrad[:, :, iφ] = Rl2 / l_in * K
                    else:
                        Kgrad[:, :, iφ] = Rl2 * K
                else:
                    for i in range(len(ld)):
                        if ld[i]:
                            if not trans:
                                Kgrad[:, :, iφ] = R[i, :, :]**2 / l[i]**3 * K
                            else:
                                Kgrad[:, :, iφ] = (R[i, :, :] / l[i])**2 * K
                            iφ += 1
            return K, Kgrad


class GammaExp(Kernel):
    r"""
    Gamma-exponential kernel object.
    .. math::
        K(R; w, l, γ) = w^2 \exp( -(R/l)^γ ),
    with the parameters of weight, w, length, l, and power norm, γ. For multiple dimensions, the
    length can be a single value applied to all directions or a list with a separate value in each
    direction. Gamma-exponential is continuous, and it is smooth only when γ=2.
    """
    def __init__(self, w, l, γ):
        super().__init__(w=w, l=l, γ=γ)

    def __call__(self, R, i_grad=False, ii_grad=False, **kwargs):
        ni, nj, n_xdims = R.shape
        if i_grad or ii_grad:
            raise InputError("Gamma Exponential Kernel is not generally differentiable, need to "
                             "separate kernels if differentiation is desired")
        w, l, γ = self.param_vals['w'], self.param_vals['l'], self.param_vals['γ']
        if not isinstance(l, Iterable):
            Rl = abs(R / l)
        else:
            Rl = empty(R.shape)
            for k in range(R.shape[0]):
                Rl[:, :, k] = abs(R[k, :, :] / l[k])
        Rlγ = sum(Rl**γ, 2)
        w2 = w**2
        K0 = exp(-Rlγ)
        return w2 * K0

    def Kφ(self, φ, R, ret_grad=False, trans=False, **kwargs):
        p = self.setparam_vals(φ, trans=trans, set=False)
        w, l, γ = p['w'], p['l'], p['γ']
        if not isinstance(l, Iterable):
            Rl = abs(R / l)
        else:
            Rl = empty(R.shape)
            for k in range(R.shape[0]):
                Rl[k, :, :] = abs(R[k, :, :] / l[k])
        Rlγ = sum(Rl**γ, 2)
        w2 = w**2
        K0 = exp(-Rlγ)
        if not ret_grad:
            return w2 * K0
        else:
            Kgrad = empty(R.shape[1:] + (self.n_φ,))
            iφ = 0
            if 'w' in self.param_priors:
                if not trans:
                    Kgrad[:, :, iφ] = 2 * w  * K0
                else:
                    Kgrad[:, :, iφ] = 2 * w2 * K0
                iφ += 1
            if 'l' in self.param_priors:
                ld  = self.param_priors['l']
                if not isinstance(ld, Iterable):
                    if not trans:
                        Kgrad[:, :, iφ] = γ * w2 * Rlγ / l * K0
                    else:
                        Kgrad[:, :, iφ] = γ * w2 * Rlγ * K0
                    iφ += 1
                else:
                    for i in range(len(ld)):
                        if ld[i]:
                            if not trans:
                                Kgrad[:, :, iφ] = γ * w2 * Rl[i, :, :]**γ / l[i] * K0
                            else:
                                Kgrad[:, :, iφ] = γ * w2 * Rl[i, :, :]**γ * K0
                            iφ += 1
            if 'γ' in self.param_priors:
                tmp1 = full(R.shape, 0, dtype='float64')
                tmp1[Rl > 0] = Rl[Rl > 0]**γ * log(Rl[Rl > 0])
                γ_tmp = sum(tmp1, 2)
                if not trans:
                    Kgrad[:, :, iφ] = -w2 * γ_tmp * K0
                else:
                    γtr = self.param_priors['γ'].trans(γ)
                    c = self.param_priors['γ'].c
                    dγdγtr = c * exp(-γtr**2 / 2) / sqrt(2 * π)
                    Kgrad[:, :, iφ] = -w2 * γ_tmp * K0 * dγdγtr
            return w2 * K0, Kgrad


class RatQuad(Kernel):
    r"""
    Rational-quadratic kernel object.
    .. math::
        K(R; w, l, \alpha) = w^2*( 1 + \frac{R^2}{2*\alpha*l^2} )^{-\alpha},
    with the parameters of weight, w, length, l, and length-variance parameter, α. The length can
    be a single value applied to all directions or a list with a separate value in each direction.
    Rational quadratic is SE over a gamma distribution of length scales with a mean of α*l^2 and
    variance of α*l^4.
    """
    def __init__(self, w, l, α):
        super().__init__(w=w, l=l, α=α)

    def __call__(self, R, i_grad=False, ii_grad=False, s=None, **kwargs):
        ni, nj, n_xdims = R.shape
        w, l_in, α = self.param_vals['w'], self.param_vals['l'], self.param_vals['α']
        if not isinstance(l_in, Iterable):
            l = full(n_xdims, l_in, dtype='float64')
        else:
            l = l_in
        K = empty((ni * (1 + (i_grad or ii_grad) * n_xdims), nj * (1 + ii_grad * n_xdims)))
        base = 1 + ((R / ℓ)**2).sum(axis=2) / (2 * α)
        K[:ni, :nj] =  w**2 * base**(-α)
        if i_grad or ii_grad:
            for k in range(n_xdims):
                lo, hi = ni * (k + 1), ni * (k + 2)
                K[lo:hi, :nj] = -w**2 * R[:, :, k] / (s[k] * l[k]**2) * base**(-(α + 1))
            if ii_grad:
                K[:ni, nj:] = -K[ni:, :nj].T
                for ki in range(n_xdims):
                    i_lo, i_hi = ni * (ki + 1), ni * (ki + 2)
                    for kj in range(n_xdims):
                        δ = 1 if ki == kj else 0
                        j_lo, j_hi = nj * (kj + 1), nj * (kj + 2)
                        K[i_lo:i_hi, j_lo:j_hi] = (w**2 / (s[ki] * l[ki] * s[kj] * l[kj])
                          (δ - (α + 1) / α * (R[:, :, ki] / l[ki]) * (R[:, :, kj] / l[kj]) / base) *
                          base**(-(α + 1)))
        return K

    def Kφ(self, φ, R, ret_grad=False, trans=False, **kwargs):
        ni, nj, n_xdims = R.shape
        p = self.setparam_vals(φ, trans=trans, set=False)
        w, l_in, α = p['w'], p['l'], p['α']
        if not isinstance(l_in, Iterable):
            l = full(n_xdims, l_in, dtype='float64')
        else:
            l = l_in
        R2l2 = ((R / ℓ)**2).sum(axis=2)
        base = 1 + R2l2 / (2 * α)
        K =  w**2 * base**(-α)

        if not ret_grad:
            return K
        else:
            Kgrad = empty((ni, nj, self.n_φ,))
            iφ = 0
            if 'w' in self.param_priors:
                if not trans:
                    Kgrad[:, :, iφ] = 2 * w  * base**(-α)
                else:
                    Kgrad[:, :, iφ] = 2 * K
                iφ += 1
            if 'l' in self.param_priors:
                ld = self.param_priors['l']
                if not isinstance(ld, Iterable):
                    if not trans:
                        Kgrad[:, :, iφ] = R2l2 / (l_in * base) * K
                    else:
                        Kgrad[:, :, iφ] = R2l2 / base * K
                    iφ += 1
                else:
                    for i in range(len(ld)):
                        if ld[i]:
                            if not trans:
                                Kgrad[:, :, iφ] = R[i, :, :]**2 / (l[i]**3 * base) * K
                            else:
                                Kgrad[:, :, iφ] = R[i, :, :]**2 / (l[i]**2 * base) * K
                            iφ += 1
            if 'α' in self.param_priors:
                α_tmp = 1 - 1 / base - log(base)
                if not trans:
                    Kgrad[:, :, iφ] = α_tmp * K
                else:
                    Kgrad[:, :, iφ] = α_tmp * α * K
            return K, Kgrad


class KernelError(Exception):
    """Base class for exceptions in the kernels module."""
    pass


class InputError(KernelError):  # -- not a ValueError? --
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
