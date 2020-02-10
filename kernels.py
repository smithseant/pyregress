# -*- coding: utf-8 -*-
"""
The Kernels themselves are rather straight forward.
The kernel parameters are not so much due to the flexibility offered
by GPI. Some comments about the handling of parameters as well as
hyper-parameters (unknown parameters):
   -The Kernel object will contain a dict (p) which stores the
    current value of all parameters - known or unknown.
   -Isotropic or 1-D lengthscales can be stored as a scalar,
    while independent lengthscales will be stored as a list.
   -The Kernel object will contain a dict (φdist) for the hyper
    parameters objects. The list of lengthscales for independent
    lengthscales can have None valued placeholders for known values.
   -The HyperParam object will contain the value of the initial
    guess (guess) as well as callables for the prior (__call__),
    prior of the transformed φ (transformed), parameter transformation
    (trans), and parameter inverse transformation (invtr).

Created Sep 2013  @authors: Sean T. Smith & Benjamin B. Schroeder
"""

__all__ = ['Kernel', 'Noise', 'SquareExp', 'GammaExp', 'RatQuad', 'KernelError']

from abc import ABCMeta, abstractmethod
from collections import Iterable
from re import search
from numpy import (empty, zeros, ones, eye, expand_dims, tile,
                   sum, abs, sqrt, exp, log, pi as π)
from pyregress.hyper_params import *

# TODO: Add a periodic kernel (..but flexible for dims. that are not periodic.)
# TODO: I would love to add heuristics so users are required to specify less.

class Kernel(metaclass=ABCMeta):
    """
    Provide methods & an interface for kernels in the GPI class.

    Specific kernels will need to inherit this baseclass and define
    __init__, __call__ & Kφ methods in the derived class.
    """
    def __init__(self, **kwargs):
        self.p = {}
        self.φdist = {}
        self.Nφ = 0
        for key, val in kwargs.items():
            if not isinstance(val, Iterable):
                if not isinstance(val, HyperPrior):
                    self.p[key] = val
                else:
                    self.Nφ += 1
                    self.p[key] = val.guess
                    self.φdist[key] = val
            else:
                self.p[key] = val
                self.φdist[key] = [None] * len(val)
                for i in range(len(val)):
                    if isinstance(val[i], HyperPrior):
                        self.Nφ += 1
                        self.φdist[key][i] = val[i]
                        self.p[key][i] = val[i].guess

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

    def get_φ(self, trans=True):
        """
        Return the values of the current hyper parameters.
        Arguments
        ---------
        trans: bool (optional),
            indicate whether to transform the hyper-parameters.
        Returns
        -------
        all_φ:  array-1D,
            current values, for each hyper parameter, from p (after running
            minimization, these should be optimized values).
        """
        # TODO: Currently relying on the dicts to maintain order (python 3.6)!
        φout = empty(self.Nφ)
        iφ = 0
        for key, val in self.φdist.items():
            if not isinstance(val, Iterable):
                if not trans:
                    φout[iφ] = self.p[key]
                else:
                    φout[iφ] = self.φdist[key].trans(self.p[key])
                iφ += 1
            else:
                for i in range(len(val)):
                    if self.φdist[key][i]:
                        if trans:
                            φout[iφ] = self.φdist[key][i].trans(self.p[key][i])
                        else:
                            φout[iφ] = self.p[key][i]
                        iφ += 1
        return φout

    def iter_φdist(self):
        """
        Provide an iterator for each prior φ in order. Two complications
        prevent this from simply being done inline: 1st, length-scales can
        be nested in lists (use an if & a for loop); and 2nd, CombiningKernel
        nest an entire dict for each term (overload this method.)
        """
        # TODO: Currently relying on the dicts to maintain order (python 3.6)!
        for val in self.φdist.values():
            if not isinstance(val, Iterable):
                yield val
            else:
                for el in [e for e in val if e]:
                    yield el

    def update_p(self, φ, trans=True, set=True):
        # TODO: Currently relying on the dicts to maintain order (python 3.6)!
        p = {}
        iφ = 0
        for key, val in self.p.items():
            if key not in self.φdist:
                p[key] = val
            elif not isinstance(self.φdist[key], Iterable):
                if not trans:
                    p[key] = φ[iφ]
                else:
                    p[key] = self.φdist[key].invtr(φ[iφ])
                iφ += 1
            else:
                p[key] = [None] * len(self.φdist[key])
                for i in range(len(self.φdist[key])):
                    if not self.φdist[key][i]:
                        p[key][i] = val[i]
                    else:
                        if not trans:
                            p[key][i] = φ[iφ]
                        else:
                            p[key][i] = self.φdist[key][i].invtr(φ[iφ])
                        iφ += 1
        if set:
            self.p = p
            return None
        else:
            return p

    def ln_priors(self, φ=None, grad=False, trans=False):
        """
        Calculate log of prior distributions for hyper-parameters.

        Arguments
        ---------
        φ: array-1D (optional),
            array of hyper-parameter values. When no values are provided,
            use the initial guess.
        grad: bool (optional),
            when grad is True also return dlnprior.
        trans: book (optional),
            indicate whether the input values of φ have been transformed
            and whether the gradients are wrt the transformed space.

        Returns
        -------
        lnprior: scalar value
            summation of values of log prior probabilities evaluated at
            values provided by params
        dlnprior: array-1D
            array of gradients of log prior probabilities evaluated at
            values provided by params
        """
        if φ is None:
            φ = [None] * self.Nφ
        lnprior = 0.0
        iφ = 0
        if not grad:
            for key, val in self.φdist.items():
                if not isinstance(val, Iterable):
                    lnprior += val(φ[iφ], trans=trans)
                    iφ += 1
                else:
                    for i in range(len(val)):
                        if self.φdist[key][i]:
                            lnprior += val[i](φ[iφ], trans=trans)
                            iφ += 1
            return lnprior
        else:
            dlnprior = empty(self.Nφ)
            for key, val in self.φdist.items():
                if not isinstance(val, Iterable):
                    lnP, dlnP = val(φ[iφ], grad=grad, trans=trans)
                    lnprior += lnP
                    dlnprior[iφ] = dlnP
                    iφ += 1
                else:
                    for i in range(len(val)):
                        if self.φdist[key][i]:
                            lnP, dlnP = val[i](φ[iφ], grad=grad, trans=trans)
                            lnprior += lnP
                            dlnprior[iφ] = dlnP
                            iφ += 1
            return lnprior, dlnprior

    @abstractmethod
    def __call__(self, Rk, grad=False, **kwargs):
        """
        Calculate and return kernel values given the radius array.

        Arguments
        ---------
        Rk: array-3D,
            directional radius matrix (difference between points).
        grad: bool (optional),
            gradients with respect to radius, when True also return Kgrad.
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
    def Kφ(self, φ, Rk, grad=False, trans=False, **kwargs):
        """
        Calculate and return kernel values given a vector of hyper
        parameters and the radius array.

        Arguments
        ---------
        φ: array-1D,
            array of hyper parameters (potentially transformed),
        Rk: array-3D,
            directional radius matrix (difference between points),
        grad: bool (optional),
            indicate whether to return the gradients with respect to the
            transformed hyper parameters as Kgrad,
        trans: bool (optional),
            indicate whether the input values of φ have been transformed
            and whether the gradients are wrt the transformed space,
        kwargs: any additional options (opt_name=opt_value),
            specific options for specific kernels, otherwise ignored.

        Returns
        -------
        K: array-2D,
            kernel values - shape must match argument R,
        Kgrad: array-3D (optional - depending on argument grad),
            partial of kernel (first two dimensions) with respect to each
            hyper parameter (third dimension).
        """
        return


def subclass(kern):
    return search("kernels.(\w+)", repr(type(kern))).group(1)


class CombiningKernel(Kernel):
    """
    This is just a super class of KernelSum & KernelProd created so these
    methods don't need to be repeated.
    """
    def __init__(self, k1, k2):
        self.terms = [k1, k2]
        self.Nφ = k1.Nφ + k2.Nφ

    @property
    def p(self):
        _p = {}
        for kern in self.terms:
            _p[subclass(kern)] = kern.p
        return _p

    @property
    def φdist(self):
        _φ = {}
        for kern in self.terms:
            _φ[subclass(kern)] = kern.φdist
        return _φ

    def iter_φdist(self):
        # TODO: Currently relying on the dicts to maintain order (python 3.6)!
        for kern in self.terms:
            for φdist in kern.iter_φdist():
                yield φdist

    def get_φ(self, trans=True):
        # TODO: Currently relying on the dicts to maintain order (python 3.6)!
        φ = empty(self.Nφ)
        iφ = 0
        for kern in self.terms:
            φ[iφ:iφ+kern.Nφ] = kern.get_φ(trans)
            iφ += kern.Nφ
        return φ

    def update_p(self, φ, trans=True, set=True):
        # TODO: Currently relying on the dicts to maintain order (python 3.6)!
        p = {}
        iφ = 0
        for kern in self.terms:
            p[subclass(kern)] = kern.update_p(φ[iφ:iφ+kern.Nφ], trans, set)
            iφ += kern.Nφ
        if set:
            return None
        else:
            return p

    def ln_priors(self, φ=None, grad=False, trans=False):
        if φ is None:
            φ = [None] * self.Nφ
        lnprior = 0.0
        iφ = 0
        if not grad:
            for kern in self.terms:
                lnprior += kern.ln_priors(φ[iφ:iφ+kern.Nφ], grad, trans)
                iφ += kern.Nφ
            return lnprior
        else:
            dlnprior = empty(self.Nφ)
            for kern in self.terms:
                lnp, dlnp = kern.ln_priors(φ[iφ:iφ+kern.Nφ], grad, trans)
                lnprior += lnp
                dlnprior[iφ:iφ+kern.Nφ] = dlnp
                iφ += kern.Nφ
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
        self.Nφ += other.Nφ
        return self

    def __call__(self, Rk, grad=False, **kwargs):
        if 'sum_terms' not in kwargs or kwargs['sum_terms'] == 'all':
            terms = self.terms
        elif kwargs['sum_terms'] == 'underlying':
            terms = [t for t in self.terms if not isinstance(t, Noise)]
        elif type(kwargs['sum_terms']) is list:
            terms = [self.terms[i] for i in kwargs['sum_terms']]
        elif type(kwargs['sum_terms']) is int:
            terms = [self.terms[kwargs['sum_terms']]]
        if not grad:
            K = zeros(Rk.shape[:2])
            for kern in terms:
                K += kern(Rk, **kwargs)
            return K
        else:
            K = zeros(Rk.shape[:2])
            Kgrad = zeros(Rk.shape)
            for kern in terms:
                Kt, Ktgrad = kern(Rk, grad=grad, **kwargs)
                K += Kt
                Kgrad += Ktgrad
            return K, Kgrad


    def Kφ(self, φ, Rk, grad=False, trans=False, **kwargs):
        if 'sum_terms' not in kwargs or kwargs['sum_terms'] == 'all':
            terms = self.terms
        elif kwargs['sum_terms'] == 'underlying':
            terms = [t for t in self.terms if not isinstance(t, Noise)]
        elif type(kwargs['sum_terms']) is list:
            terms = [self.terms[i] for i in kwargs['sum_terms']]
        elif type(kwargs['sum_terms']) is int:
            terms = [self.terms[kwargs['sum_terms']]]
        if not grad:
            K = zeros(Rk.shape[:2])
            iφ = 0
            for kern in terms:
                Nφ = kern.Nφ
                K += kern.Kφ(φ[iφ:iφ+Nφ], Rk, trans=trans, **kwargs)
                iφ += Nφ
            return K
        else:
            K = zeros(Rk.shape[:2])
            Kgrad = zeros((Rk.shape[0], Rk.shape[1], self.Nφ))
            iφ = 0
            for kern in terms:
                Nφ = kern.Nφ
                Kt, Kgrad[:, :, iφ:iφ+Nφ] = kern.Kφ(φ[iφ:iφ+Nφ], Rk, grad,
                                                    trans, **kwargs)
                K += Kt
                iφ += Nφ
            return K, Kgrad


class KernelProd(CombiningKernel):
    """Provide a class that lists kernels to be multiplied at evaluation."""
    def __mul__(self, other, self_on_right=False):
        if isinstance(other, KernelProd):
            self.terms += other.terms
        elif isinstance(other, Kernel):
            self.terms += [other]
        else:
            # TODO: Throw an error!
            pass
        self.Nφ += other.Nφ
        return self

    def __call__(self, Rk, grad=False, **kwargs):
        if grad is not False:
            raise InputError("Kernel product does not currently support" +
                             " radial gradients.")
        K = ones(Rk.shape[:2])
        for kern in self.terms:
            K *= kern(Rk, **kwargs)
        return K

    def Kφ(self, φ, Rk, grad=False, trans=False, **kwargs):
        if not grad:
            K = ones(Rk.shape[:2])
            iφ = 0
            for kern in self.terms:
                Nφ = kern.Nφ
                K *= kern.Kφ(φ[iφ:iφ+Nφ], Rk, trans=trans, **kwargs)
                iφ += Nφ
            return K
        else:
            K = ones(Rk.shape[:2])
            Kgrad = ones((Rk.shape[0], Rk.shape[1], self.Nφ))
            iφ = 0
            for kern in self.terms:
                Nφ = kern.Nφ
                Kt, Kgt = kern.Kφ(φ[iφ:iφ+Nφ], Rk, grad, trans, **kwargs)
                K *= Kt
                irange = range(iφ, iφ+Nφ)
                iother = range(0, iφ) + range(iφ+Nφ, self.Nφ)
                Kgrad[:, :, irange] *= Kgt
                Kgrad[:, :, iother] *= tile(expand_dims(Kt, 2),
                                            (1, 1, len(iother)))
                iφ += Nφ
            return K, Kgrad


class Noise(Kernel):
    r"""
    White noise kernel object.
    ..math::
        K(R, data; w) = w^2 * I, or a zero matrix based on the data,
    with the weight parameter, w, and a flag indicating inclusion or not.
    White noise is discontinuous.
    """
    def __init__(self, w):
        super().__init__(w=w)

    def __call__(self, Rk, grad=False, **kwargs):
        w = self.p['w']
        w2 = w**2
        K0 = eye(Rk.shape[0], Rk.shape[1])
        if not grad:
            # K = w2 * K0
            return w2 * K0
        else:
            raise InputError("Noise Kernel is not differentiable, need" +
                             " to separate kernels for differentiation")

    def Kφ(self, φ, Rk, grad=False, trans=False, **kwargs):
        p = self.update_p(φ, trans=trans, set=False)
        w = p['w']
        w2 = w**2
        K0 = eye(Rk.shape[0], Rk.shape[1])
        if not grad:
            return w2 * K0
        else:
            Kgrad = empty((Rk.shape[0], Rk.shape[1], self.Nφ))
            if 'w' in self.φdist:
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
    with the parameters of weight, w, and length, l. For multiple
    dimensions, the length can be a single value applied to all directions
    or it can be a list with a separate value in each direction.
    Squared-exponential is continuous and infinitely differentiable.
    """
    def __init__(self, w, l):
        super().__init__(w=w, l=l)

    def __call__(self, Rk, grad=False, **kwargs):
        w, l = self.p['w'], self.p['l']
        if not isinstance(l, Iterable):
            Rl2 = sum(Rk**2, 2) / l**2
        else:
            Rl2 = zeros(Rk.shape[:2])
            for k in range(Rk.shape[2]):
                Rl2 += (Rk[:, :, k] / l[k])**2
        w2 = w**2
        K0 = exp(-0.5 * Rl2)
        if not grad:
            return w2 * K0
        else:
            Kgrad = empty(Rk.shape)
            for i in range(Rk.shape[2]):
                if isinstance(l, Iterable):
                    Kgrad[:, :, i] = -w2 * Rk[:, :, i] / l[i]**2 * K0
                else:
                    Kgrad[:, :, i] = -w2 * Rk[:, :, i] / l**2 * K0
            return w2 * K0, Kgrad

    def Kφ(self, φ, Rk, grad=False, trans=False, **kwargs):
        p = self.update_p(φ, trans=trans, set=False)
        w, l = p['w'], p['l']
        if not isinstance(l, Iterable):
            Rl2 = sum(Rk**2, 2) / l**2
        else:
            Rl2 = zeros(Rk.shape[:2])
            for k in range(Rk.shape[2]):
                Rl2 += (Rk[:, :, k] / l[k])**2
        w2 = w**2
        K0 = exp(-0.5 * Rl2)
        if not grad:
            return w2 * K0
        else:
            Kgrad = empty(Rk.shape[:2] + (self.Nφ,))
            iφ = 0
            if 'w' in self.φdist:
                if not trans:
                    Kgrad[:, :, iφ] = 2 * w  * K0
                else:
                    Kgrad[:, :, iφ] = 2 * w2 * K0
                iφ += 1
            if 'l' in self.φdist:
                ld = self.φdist['l']
                if not isinstance(ld, Iterable):
                    if not trans:
                        Kgrad[:, :, iφ] = w2 * Rl2 / l * K0
                    else:
                        Kgrad[:, :, iφ] = w2 * Rl2 * K0
                else:
                    for i in range(len(ld)):
                        if ld[i]:
                            if not trans:
                                Kgrad[:, :, iφ] = w2 * Rk[:, :, i]**2 / l[i]**3 * K0
                            else:
                                Kgrad[:, :, iφ] = w2 * (Rk[:, :, i] / l[i])**2 * K0
                            iφ += 1
            return w2 * K0, Kgrad


class GammaExp(Kernel):
    r"""
    Gamma-exponential kernel object.
    .. math::
        K(R; w, l, γ) = w^2 \exp( -(R/l)^γ ),
    with the parameters of weight, w, length, l, and power norm, γ.
    For multiple dimensions, the length can be a single value applied to
    all directions or a list with a separate value in each direction.
    Gamma-exponential is continuous, and it is smooth only when γ=2.
    """
    def __init__(self, w, l, γ):
        super().__init__(w=w, l=l, γ=γ)

    def __call__(self, Rk, grad=False, **kwargs):
        w, l, γ = self.p['w'], self.p['l'], self.p['γ']
        if not isinstance(l, Iterable):
            Rl = abs(Rk / l)
        else:
            Rl = empty(Rk.shape)
            for k in range(Rk.shape[2]):
                Rl[:, :, k] = abs(Rk[:, :, k] / l[k])
        Rlγ = sum(Rl**γ, 2)
        w2 = w**2
        K0 = exp(-Rlγ)
        if not grad:
            return w2 * K0
        else:
            raise InputError("Gamma Exponential Kernel is not generally" +
                             "differentiable, need to separate kernels if " +
                             "differentiation is desired")

    def Kφ(self, φ, Rk, grad=False, trans=False, **kwargs):
        p = self.update_p(φ, trans=trans, set=False)
        w, l, γ = p['w'], p['l'], p['γ']
        if not isinstance(l, Iterable):
            Rl = abs(Rk / l)
        else:
            Rl = empty(Rk.shape)
            for k in range(Rk.shape[2]):
                Rl[:, :, k] = abs(Rk[:, :, k] / l[k])
        Rlγ = sum(Rl**γ, 2)
        w2 = w**2
        K0 = exp(-Rlγ)
        if not grad:
            return w2 * K0
        else:
            Kgrad = empty(Rk.shape[:2] + (self.Nφ,))
            iφ = 0
            if 'w' in self.φdist:
                if not trans:
                    Kgrad[:, :, iφ] = 2 * w  * K0
                else:
                    Kgrad[:, :, iφ] = 2 * w2 * K0
                iφ += 1
            if 'l' in self.φdist:
                ld  = self.φdist['l']
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
                                Kgrad[:, :, iφ] = γ * w2 * Rl[:, :, i]**γ / l[i] * K0
                            else:
                                Kgrad[:, :, iφ] = γ * w2 * Rl[:, :, i]**γ * K0
                            iφ += 1
            if 'γ' in self.φdist:
                tmp1 = zeros(Rk.shape)
                tmp1[Rl > 0] = Rl[Rl > 0]**γ * log(Rl[Rl > 0])
                γ_tmp = sum(tmp1, 2)
                if not trans:
                    Kgrad[:, :, iφ] = -w2 * γ_tmp * K0
                else:
                    γtr = self.φdist['γ'].trans(γ)
                    c = self.φdist['γ'].c
                    dγdγtr = c * exp(-γtr**2 / 2) / sqrt(2 * π)
                    Kgrad[:, :, iφ] = -w2 * γ_tmp * K0 * dγdγtr
            return w2 * K0, Kgrad


class RatQuad(Kernel):
    r"""
    Rational-quadratic kernel object.
    .. math::
        K(R; w, l, \alpha) = w^2*( 1 + \frac{R^2}{2*\alpha*l^2} )^{-\alpha},
    with the parameters of weight, w, length, l, and length-variance
    parameter, α. The length can be a single value applied to all
    directions or a list with a separate value in each direction.
    Rational quadratic is SE over a gamma distribution of length scales
    with a mean of α*l^2 and variance of α*l^4.
    """
    def __init__(self, w, l, α):
        super().__init__(w=w, l=l, α=α)

    def __call__(self, Rk, grad=False, **kwargs):
        w, l, α = self.p['w'], self.p['l'], self.p['α']
        if not isinstance(l, list):
            R2l2 = sum(Rk**2, 2) / l**2
        else:
            R2l2 = zeros(Rk.shape[:2])
            for k in range(Rk.shape[2]):
                R2l2 += (Rk[:, :, k] / l[k])**2
        w2 = w**2
        base = 1 + R2l2 / (2 * α)
        K0 = base**(-α)
        if not grad:
            return w2*K0
        else:
            Kgrad = empty(Rk.shape)
            for i in range(Rk.shape[2]):
                if isinstance(l, list):
                    Kgrad[:, :, i] = -w2 * Rk[:, :, i] / l[i]**2 * base**(-α-1)
                else:
                    Kgrad[:, :, i] = -w2 * Rk[:, :, i] / l**2 * base**(-α-1)
            return w2*K0, Kgrad

    def Kφ(self, φ, Rk, grad=False, trans=False, **kwargs):
        p = self.update_p(φ, trans=trans, set=False)
        w, l, α = p['w'], p['l'], p['α']
        if not isinstance(l, list):
            R2l2 = sum(Rk**2, 2) / l**2
        else:
            R2l2 = zeros(Rk.shape[:2])
            for k in range(Rk.shape[2]):
                R2l2 += (Rk[:, :, k] / l[k])**2
        w2 = w**2
        base = 1 + R2l2 / (2 * α)
        K0 = base**(-α)
        if not grad:
            return w2*K0
        else:
            Kgrad = empty(Rk.shape[:2] + (self.Nφ,))
            iφ = 0
            if 'w' in self.φdist:
                if not trans:
                    Kgrad[:, :, iφ] = 2 * w  * K0
                else:
                    Kgrad[:, :, iφ] = 2 * w2 * K0
                iφ += 1
            if 'l' in self.φdist:
                ld = self.φdist['l']
                if not isinstance(ld, Iterable):
                    if not trans:
                        Kgrad[:, :, iφ] = w2 * R2l2 / (l * base) * K0
                    else:
                        Kgrad[:, :, iφ] = w2 * R2l2 / base * K0
                    iφ += 1
                else:
                    for i in range(len(ld)):
                        if ld[i]:
                            if not trans:
                                Kgrad[:, :, iφ] = w2 * Rk[:, :, i]**2 / (l[i]**3 * base) * K0
                            else:
                                Kgrad[:, :, iφ] = w2 * Rk[:, :, i]**2 / (l[i]**2 * base) * K0
                            iφ += 1
            if 'α' in self.φdist:
                α_tmp = 1 - 1 / base - log(base)
                if not trans:
                    Kgrad[:, :, iφ] = w2 * α_tmp * K0
                else:
                    Kgrad[:, :, iφ] = w2 * α_tmp * α * K0
            return w2*K0, Kgrad


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
                input argument that is the source of error. Provided so
                the value can be reported when the error is caught.
        """
        self.args = (msg,)
        self.input_argument = input_argument
