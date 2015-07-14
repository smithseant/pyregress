# -*- coding: utf-8 -*-
"""
Docstring for the kernels module - needs to be written
"""
# Created Sep 2013
# @author: Sean T. Smith

__all__ = ['Kernel', 'KernelSum', 'KernelProd', 'Noise', 'RatQuad',
           'SquareExp', 'GammaExp']

from abc import ABCMeta, abstractmethod
from collections import OrderedDict as odict
from numbers import Number
from numpy import (empty, zeros, ones, eye, sum, abs,
                   ix_, expand_dims, tile, hstack)
from scipy import exp, log
from pyregress.hyper_params import *

# TODO: Add periodic, but it would require general handling of multiple Rs.


class Kernel(metaclass=ABCMeta):
    """
    Provide methods & an interface for kernels in the GPP class.

    User-defined kernels will need to inherit this baseclass and define
    both __init__ and __call__ methods in the derived class. This base
    class provides the abstract interface for the __call__ method and
    provides the methods: __init__, declare_hyper, and map_hyper.
    """

    def __init__(self, params, params_spec):
        """
        Create a Kernel object.

        Arguments
        ---------
        params: dict,
            the specified kernel parameters. The length parameter value
            can be a list for different lengths in multiple dimensions.
            Uncertain values should be represented by HyperPrior objects.
        params_spec: dict,
            All required kernel parameters with the domain of each as a
            tuple (min, max). When unbounded, use None for min and/or max.
        """
        # TODO: throw an error if the parameters don't match the spec.
        self.Np = len(params_spec)
        self.Nhp = len([i for i in params.values()
                        if isinstance(i, HyperPrior)])
        if 'l' in params and isinstance(params['l'], list):
            self.Np += len(params['l']) - 1
            self.Nhp += len([i for i in params['l']
                             if isinstance(i, HyperPrior)])
        self.p = odict([(key, None) for key in params_spec.keys()])
        self.p_bounds = [val for val in params_spec.values()]
        self.hp, self.hp_id = [], []
        for key in params_spec.keys():
            val = params[key]
            if isinstance(val, Number):
                self.p[key] = val
            elif isinstance(val, HyperPrior):
                self.hp += [val]
                self.hp_id += [key]
                self.p[key] = val.guess
            elif isinstance(val, list):
                self.p[key] = [None]*len(val)
                for i in range(len(val)):
                    if isinstance(val[i], Number):
                        self.p[key][i] = val[i]
                    elif isinstance(val[i], HyperPrior):
                        self.hp += [val[i]]
                        self.hp_id += [i]
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

    def _map_hyper(self, hp_mapped=None, bounds_mapped=None, unmap=False):
        """Replace hyper-parameter values with pointers to a 1D array."""
        if hp_mapped is None:
            hp_mapped = empty(self.Nhp)
            bounds_mapped = empty(0)
        if unmap is True:
            bounds_mapped = []
        if isinstance(self, KernelSum) or isinstance(self, KernelProd):
            i = 0
            for kern in self.terms:
                hp_mapped[i:i+kern.Nhp], bounds_mapped = \
                    kern._map_hyper(hp_mapped[i: i+kern.Nhp],
                                    bounds_mapped, unmap=unmap)
                i += kern.Nhp
        else:
            for (hp, i) in zip(self.hp_id, range(self.Nhp)):
                if not isinstance(hp, int):
                    if unmap is True:
                        self.hp[i].guess = self.p[hp]
                        hp_mapped[i] = self.hp[i].guess
                    else:
                        hp_mapped[i] = self.hp[i].guess
                        self.p[hp] = hp_mapped[i:i+1]
                    bounds_mapped = hstack((bounds_mapped, (self.p_bounds[i])))
                else:
                    if unmap is True:
                        self.hp[i].guess = self.p['l'][hp]
                        hp_mapped[i] = self.hp[i].guess
                    else:
                        hp_mapped[i] = self.hp[i].guess
                        self.p['l'][hp] = hp_mapped[i:i+1]
#                    if hp != 0:
                    bounds_mapped = hstack((bounds_mapped,
                                            (self.p_bounds[0])))
#                    else:
#                        bounds_mapped = hstack((bounds_mapped,
#                                                (self.p_bounds[i])))

        # return (self, hp_mapped)
        return hp_mapped, bounds_mapped

    def _ln_priors(self, params=None, grad=False):
        """
        Calculate log of prior distributions for hyper-parameters.

        Arguments
        ---------
        params: array-1D
            array of hyper-parameter values.
        grad: bool (optional),
            when grad is True also return dlnprior, and when grad is 'Hess'
            also return d2lnpdf.

        Returns
        -------
        lnprior: scalar value
            summation of values of log prior probabilities evaluated at
            values provided by params
        dlnprior: array-1D
            array of gradients of log prior probabilities evaluated at
            values provided by params
        d2lnprior: array-2D
            matrix where diagonal is 2nd derivatives of log prior probabilities
            evaluated at values provided by params
        """
        if params is None:
            params = zeros(self.Nhp)
            i = 0
            if isinstance(self, KernelSum) or isinstance(self, KernelProd):
                for kern in self.terms:
                    for hp in kern.hp:
                        params[i] = hp.guess
                        i += 1
            else:
                for hp in self.hp:
                    params[i] = hp.guess
                    i += 1

        lnprior = 0.0
        if not grad:
            if isinstance(self, KernelSum) or isinstance(self, KernelProd):
                i = 0
                for kern in self.terms:
                    for f_prior in kern.hp:
                        lnprior += f_prior(params[i])
                        i += 1
            else:
                for f_prior, i in zip(self.hp, range(self.Nhp)):
                    lnprior += f_prior(params[i])
            return lnprior

        elif grad is True:
            dlnprior = empty(self.Nhp)
            if isinstance(self, KernelSum) or isinstance(self, KernelProd):
                i = 0
                for kern in self.terms:
                    for f_prior in kern.hp:
                        (lnp, dlnp) = f_prior(params[i], grad)
                        lnprior += lnp
                        dlnprior[i] = dlnp
                        i += 1
            else:
                for f_prior, i in zip(self.hp, range(self.Nhp)):
                    lnp, dlnp = f_prior(params[i], grad)
                    lnprior += lnp
                    dlnprior[i] = dlnp
            return lnprior, dlnprior

        elif grad == 'Hess':
            dlnprior = empty(self.Nhp)
            d2lnprior = zeros((self.Nhp, self.Nhp))
            if isinstance(self, KernelSum) or isinstance(self, KernelProd):
                i = 0
                for kern in self.terms:
                    for f_prior in kern.hp:
                        lnp, dlnp, d2lnp = f_prior(params[i], grad)
                        lnprior += lnp
                        dlnprior[i] = dlnp
                        d2lnprior[i, i] = d2lnp
                        i += 1
            else:
                for f_prior, i in zip(self.hp, range(self.Nhp)):
                    lnp, dlnp, d2lnp = f_prior(params[i], grad)
                    lnprior += lnp
                    dlnprior[i] = dlnp
                    d2lnprior[i, i] = d2lnp
            return lnprior, dlnprior, d2lnprior

    def get_hp(self):
        """
        Return the current hyper parameter values.

        Returns
        -------
        all_hp:  array-1D,
            current values, for each hyper parameter, from HyperPrior.guess
            (after minimization is run these should be optimized values).
        """
        all_hp = zeros(self.Nhp)
        i = 0
        if isinstance(self, KernelSum) or isinstance(self, KernelProd):
            for kern in self.terms:
                for hp in kern.hp:
                    all_hp[i] = hp.guess
                    i += 1
        else:
            for hp in self.hp:
                all_hp[i] = hp.guess
                i += 1
        return all_hp

    @abstractmethod
    def __call__(self, Rk, grad_hp=False, grad_r=False, **kwargs):
        """
        Calculate and return kernel values given the radius array.

        Arguments
        ---------
        Rk: array-3D,
            directional radius matrix (difference between points).
        grad_hp: bool (optional),
            gradients with respect to hyper parameters,
            when grad_hp is True also return Kgrad, and when grad_hp is 'Hess'
            also return Khess.
        grad_r: bool (optional),
            gradients with respect to radius,
            when grad_r is True also return Kgrad, and when grad_r is 'Hess'
            also return Khess.
        kwargs: any additional options (opt_name=opt_value),
            specific options for specific kernels, otherwise ignored.

        Returns
        -------
        K: array-2D,
            kernel values - shape must match argument R.
        Kgrad: array-3D (optional - depending on argument grad),
            partial of kernel (first two dimensions) with respect to each
            hyper parameter (third dimension).
        Khess: array-4D (optional - depending on argument grad),
            second derivative for all combinations of two hyper parameters.
        """
        return


class KernelSum(Kernel):
    """Provide a class that lists kernels to be added at evaluation."""
    def __init__(self, k1, k2):
        self.terms = [k1, k2]
        self.Np, self.Nhp = k1.Np + k2.Np, k1.Nhp + k2.Nhp

    def __add__(self, other, self_on_right=False):
        if isinstance(other, KernelSum):
            self.terms += other.terms
        elif isinstance(other, Kernel):
            if self_on_right:
                self.terms = [other] + self.terms
            else:
                self.terms += [other]
        else:
            # TODO: throw an error!
            pass
        self.Np += other.Np
        self.Nhp += other.Nhp
        return self

    def __call__(self, Rk, grad_hp=False, grad_r=False, **kwargs):
        if 'sum_terms' in kwargs and not kwargs['sum_terms'] is True:
            if type(kwargs['sum_terms']) is list:
                terms = [self.terms[i] for i in kwargs['sum_terms']]
            elif type(kwargs['sum_terms']) is int:
                terms = [self.terms[kwargs['sum_terms']]]
        else:
            terms = self.terms
        if (not grad_hp) and (not grad_r):
            K = zeros(Rk.shape[:2])
            for kern in terms:
                K += kern(Rk, **kwargs)
            return K
        if grad_hp is not False:
            if grad_hp != 'Hess':
                K = zeros(Rk.shape[:2])
                Kgrad = zeros((Rk.shape[0], Rk.shape[1], self.Nhp))
                h = 0
                for kern in terms:
                    K_t, Kgrad[:, :, h:h+kern.Nhp] = kern(Rk, grad_hp=grad_hp,
                                                          **kwargs)
                    K += K_t
                    h += kern.Nhp
                return K, Kgrad
            else:
                K = zeros(Rk.shape[:2])
                Kgrad = zeros((Rk.shape[0], Rk.shape[1], self.Nhp))
                Khess = zeros((Rk.shape[0], Rk.shape[1], self.Nhp, self.Nhp))
                h = 0
                for kern in terms:
                    hn = h + kern.Nhp
                    Kt, Kgrad[:, :, h:hn], Khess[:, :, h:hn, h:hn] = \
                        kern(Rk, grad_hp=grad_hp, **kwargs)
                    K += Kt
                    h = hn
                return K, Kgrad, Khess
        if grad_r is not False:
            if grad_r != 'Hess':
                K = zeros(Rk.shape[:2])
                Kgrad = zeros(Rk.shape)
                h = 0
                for kern in terms:
                    Kt, Ktgrad = kern(Rk, grad_r=grad_r, **kwargs)
                    K += Kt
                    Kgrad += Ktgrad
                return K, Kgrad
            else:
                K = zeros(Rk.shape[:2])
                Kgrad = zeros(Rk.shape)
                Khess = zeros((Rk.shape[0], Rk.shape[1], Rk.shape[2],
                               Rk.shape[2]))
                h = 0
                for kern in terms:
                    Kt, Ktgrad, Kthess = kern(Rk, grad_r=grad_r, **kwargs)
                    K += Kt
                    Kgrad += Ktgrad
                    Khess += Kthess
                return K, Kgrad, Khess


class KernelProd(Kernel):
    """Provide a class that lists kernels to be multiplied at evaluation."""
    def __init__(self, k1, k2):
        self.terms = [k1, k2]
        self.Np, self.Nhp = k1.Np + k2.Np, k1.Nhp + k2.Nhp

    def __mul__(self, other, self_on_right=False):
        if isinstance(other, KernelProd):
            self.terms += other.terms
        elif isinstance(other, Kernel):
            self.terms += [other]
        else:
            # TODO: throw an error!
            pass
        self.Np += other.Np
        self.Nhp += other.Nhp
        return self

    def __call__(self, Rk, grad_hp=False, grad_r=False, **kwargs):
        if grad_r is not False:
            raise InputError("Kernel products do not currently support" +
                             " radial gradients.  If desired, feel free" +
                             " to implement")
        if not grad_hp:
            K = ones(Rk.shape[:2])
            for kern in self.terms:
                K *= kern(Rk, **kwargs)
            return K
        elif grad_hp != 'Hess':
            K = ones(Rk.shape[:2])
            Kgrad = ones((Rk.shape[0], Rk.shape[1], self.Nhp))
            h = 0
            for kern in self.terms:
                Kt, Kgt = kern(Rk, grad_hp=grad_hp, **kwargs)
                K *= Kt
                irange = range(h, h+kern.Nhp)
                iother = range(0, h) + range(h+kern.Nhp, self.Nhp)
                Kgrad[:, :, irange] *= Kgt
                Kgrad[:, :, iother] *= tile(expand_dims(Kt, 2),
                                            (1, 1, len(iother)))
                h += kern.Nhp
            return K, Kgrad
        else:
            K = ones(Rk.shape[:2])
            Kgrad = ones((Rk.shape[0], Rk.shape[1], self.Nhp))
            Khess = ones((Rk.shape[0], Rk.shape[1], self.Nhp, self.Nhp))
            h = 0
            for kern in self.terms:
                Kt, Kgt, Kht = kern(Rk, grad_hp=grad_hp, **kwargs)
                K *= Kt
                hn = h + kern.Nhp
                irange = range(h, hn)
                iother = range(0, h) + range(hn, self.Nhp)
                Kgrad[:, :, irange] *= Kgt
                Kgrad[:, :, iother] *= tile(expand_dims(Kt, 2),
                                            (1, 1, len(iother)))
                Khess[ix_(range(0, Rk.shape[0]), range(0, Rk.shape[1]),
                          irange, irange)] *= Kht
                Khess[ix_(range(0, Rk.shape[0]), range(0, Rk.shape[1]),
                          irange, iother)] *= tile(expand_dims(Kgt, 3),
                                                   (1, 1, 1, len(iother)))
                Khess[ix_(range(0, Rk.shape[0]), range(0, Rk.shape[1]),
                          iother, irange)] *= tile(expand_dims(Kgt, 2),
                                                   (1, 1, len(iother), 1))
                Khess[ix_(range(0, Rk.shape[0]), range(0, Rk.shape[1]),
                      iother, iother)] *= \
                          tile(expand_dims(expand_dims(Kt, 2), 3),
                               (1, 1, len(iother), len(iother)))
                h += kern.Nhp
            return K, Kgrad, Khess


class Noise(Kernel):
    r"""
    White noise kernel object.
    ..math::
        K(R, data; w) = w^2 * I, or a zero matrix based on the data,
    with the weight parameter, w, and a flag indicating inclusion or not.
    White noise is discontinuous.
    """
    def __init__(self, **params):
        p_bounds = odict([('w', (0.0, None))])
        super(Noise, self).__init__(params, p_bounds)

    def __call__(self, Rk, grad_hp=False, grad_r=False, **kwargs):
        w = self.p['w']
        w2 = w**2
        if 'block_diag' in kwargs and kwargs['block_diag'] == True:
                K0 = eye(Rk.shape[0], Rk.shape[1])
        else:
            K0 = zeros(Rk.shape[:2])
        if (not grad_hp) and (not grad_r):
            # K = w2*K0
            return w2*K0
        if grad_hp is not False:
            Kgrad = empty((Rk.shape[0], Rk.shape[1], self.Nhp))
            if 'w' in self.hp_id:
                # dK/dw
                Kgrad[:, :, 0] = 2.0*w*K0
            if grad_hp != 'Hess':
                return w2*K0, Kgrad
            Khess = empty((Rk.shape[0], Rk.shape[1], self.Np, self.Np))
            if 'w' in self.hp_id:
                # d^2K/dw^2
                Khess[:, :, 0, 0] = 2.0*K0
            return w2*K0, Kgrad, Khess

        if grad_r is not False:
            raise InputError("Noise Kernel is not differentiable, need" +
                             " to separate kernels if differntiation is" +
                             " desired")


class SquareExp(Kernel):
    r"""
    Squared-exponential kernel object.
    .. math::
        K(R; w, l) = w^2*\exp( -1/2 *(R/l)^2 ),
    with the parameters of weight, w, and length, l. For multiple
    dimensions, the length can be a single value applied to all directions
    or it can be a list with a separate value in each direction.
    Squared-exponential is continuous and infinitely differentiable.
    """
    def __init__(self, **params):
        p_bounds = odict([('w', (1E-6, None)), ('l', (1E-6, None))])
        super(SquareExp, self).__init__(params, p_bounds)

    def __call__(self, Rk, grad_hp=False, grad_r=False, **kwargs):
        w, l = self.p['w'], self.p['l']
        if not isinstance(l, list):
            R2l2 = sum(Rk**2, 2)/l**2
        else:
            R2l2 = zeros(Rk.shape[:2])
            for k in range(Rk.shape[2]):
                R2l2 += (Rk[:, :, k]/l[k])**2
        w2 = w**2
        K0 = exp(-0.5*R2l2)
        if (not grad_hp) and (not grad_r):
            # K = w2*K0
            return w2*K0
        if grad_hp is not False:
            # First derivatives:
            Kgrad = empty((Rk.shape[0], Rk.shape[1], self.Nhp))
            for (i, h) in zip(range(self.Nhp), self.hp_id):
                if h == 'w':
                    # dK/dw
                    Kgrad[:, :, i] = 2.0*w*K0
                elif h == 'l':
                    # dK/dl
                    Kgrad[:, :, i] = w2*R2l2/l*K0
                elif isinstance(h, int):
                    # dK/dl_h
                    Kgrad[:, :, i] = w2*Rk[:, :, h]**2/l[h]**3*K0
            if grad_hp != 'Hess':
                return w2*K0, Kgrad
            # Second derivatives:
            Khess = empty((Rk.shape[0], Rk.shape[1], self.Nhp, self.Nhp))
            for i, h1 in zip(range(self.Nhp), self.hp_id):
                for j, h2 in zip(range(i, self.Nhp+1), self.hp_id[i:]):
                    if h1 == 'w' and h2 == 'w':
                        # d^2K/dw^2
                        Khess[:, :, i, j] = 2.0*K0
                    elif h1 == 'w' and h2 == 'l':
                        # d^2K/dwdl
                        Khess[:, :, i, j] = 2.0*w*R2l2/l*K0
                        Khess[:, :, j, i] = Khess[:, :, i, j]
                    elif h1 == 'w' and isinstance(h2, int):
                        # d^2K/dwdl_i
                        Khess[:, :, i, j] = 2.0*w*Rk[:, :, h2]**2/l[h2]**3*K0
                        Khess[:, :, j, i] = Khess[:, :, i, j]
                    elif h1 == 'l' and h2 == 'l':
                        # d^2K/dl^2
                        Khess[:, :, i, j] = w2*R2l2/l**2*(R2l2-3.0)*K0
                    elif (isinstance(h1, int) and isinstance(h2, int) and
                          h1 == h2):
                        # d^2K/dl_i^2
                        Khess[:, :, i, j] = (w2*Rk[:, :, h1]**2/l[h1]**4 *
                                             ((Rk[:, :, h1]/l[h1])**2 -
                                             3.0)*K0)
                    elif isinstance(h1, int) and isinstance(h2, int):
                        # d^2K/dl_i dl_j
                        Khess[:, :, i, j] = (w2*(Rk[:, :, h1] *
                                             Rk[:, :, h2])**2 /
                                             (l[h1]*l[h2])**3*K0)
                        Khess[:, :, j, i] = Khess[:, :, i, j]
            return w2*K0, Kgrad, Khess

        if grad_r is not False:
            # First derivatives:
            Kgrad = empty(Rk.shape)
            for i in range(Rk.shape[2]):
                if isinstance(l, list):
                    Kgrad[:, :, i] = -w2*Rk[:, :, i]/l[i]**2 * K0
                else:
                    Kgrad[:, :, i] = -w2*Rk[:, :, i]/l**2 * K0
            if grad_r != 'Hess':
                return w2*K0, Kgrad
            # Second derivatives:
            Khess = empty((Rk.shape[0], Rk.shape[1], Rk.shape[2], Rk.shape[2]))
            for i in range(Rk.shape[2]):
                for j in range(Rk.shape[2]):
                    if i == j:
                        if isinstance(l, list):
                            Khess[:, :, i, j] = (w2/l[i]**2 *
                                                 (Rk[:, :, i]**2/l[i]**2 -
                                                  1.0)*K0)
                        else:
                            Khess[:, :, i, j] = (w2/l**2 *
                                                 (Rk[:, :, i]**2/l**2 -
                                                  1.0) * K0)
                    else:
                        if isinstance(l, list):
                            Khess[:, :, i, j] = (w2*Rk[:, :, i]*Rk[:, :, j] /
                                                 (l[i]**2 * l[j]**2) * K0)
                        else:
                            Khess[:, :, i, j] = (w2*Rk[:, :, i]*Rk[:, :, j] /
                                                 (l**4)*K0)
            return w2*K0, Kgrad, Khess


class GammaExp(Kernel):
    r"""
    Gamma-exponential kernel object.
    .. math::
        K(R; w, l, gamma) = w^2*\exp( -(R/l)^{\gamma} ),
    with the parameters of weight, w, length, l, and power norm, gamma.
    For multiple dimensions, the length can be a single value applied to
    all directions or a list with a separate value in each direction.
    Gamma-exponential is continuous, and when gamma=2 it is smooth.
    """
    def __init__(self, **params):
        p_bounds = odict([('w', (0.0, None)), ('l', (0.0, None)),
                          ('gamma', (0.0, 2.0))])
        super(GammaExp, self).__init__(params, p_bounds)

    def __call__(self, Rk, grad_hp=False, grad_r=False, **kwargs):
        w, l, g = self.p['w'], self.p['l'], self.p['gamma']
        if not isinstance(l, list):
            Rl = abs(Rk/l)
        else:
            Rl = empty(Rk.shape)
            for k in range(Rk.shape[2]):
                Rl[:, :, k] = abs(Rk[:, :, k]/l[k])
        Rglg = sum(Rl**g, 2)
        w2 = w**2
        K0 = exp(-Rglg)
        if (not grad_hp) and (not grad_r):
            # K = w2*K0
            return w2*K0
        if grad_hp is not False:
            # First derivatives:
            Kgrad = empty((Rk.shape[0], Rk.shape[1], self.Nhp))
            for i, h in zip(range(self.Nhp), self.hp_id):
                if h == 'w':
                    # dK/dw
                    Kgrad[:, :, i] = 2.0*w*K0
                elif h == 'l':
                    # dK/dl
                    Kgrad[:, :, i] = g*w2*Rglg / l*K0
                elif isinstance(h, int):
                    # dK/dl_h
                    Kgrad[:, :, i] = g*w2*Rl[:, :, h]**g / l[h]*K0
                elif h == 'gamma':
                    # dK/dgamma
                    tmp1 = zeros(Rk.shape)
                    tmp1[Rl > 0] = Rl[Rl > 0]**g * log(Rl[Rl > 0.0])
                    gamma_tmp = sum(tmp1, 2)
                    Kgrad[:, :, i] = -w2*gamma_tmp*K0
            if grad_hp != 'Hess':
                return w2*K0, Kgrad
            # Second derivatives:
            Khess = empty((Rk.shape[0], Rk.shape[1], self.Nhp, self.Nhp))
            for i, h1 in zip(range(self.Nhp), self.hp_id):
                for j, h2 in zip(range(i, self.Nhp), self.hp_id[i:]):
                    if h1 == 'w' and h2 == 'w':
                        # d^2K/dw^2
                        Khess[:, :, i, j] = 2.0*K0
                    elif h1 == 'w' and h2 == 'l':
                        # d^2K/dwdl
                        Khess[:, :, i, j] = 2.0*g*w*Rglg/l*K0
                        Khess[:, :, j, i] = Khess[:, :, i, j]
                    elif h1 == 'w' and isinstance(h2, int):
                        # d^2K/dwdl_h
                        Khess[:, :, i, j] = 2.0*g*w*Rl[:, :, h2]**g/l[h2]*K0
                        Khess[:, :, j, i] = Khess[:, :, i, j]
                    elif h1 == 'w' and h2 == 'gamma':
                        # d^2K/dwdgamma
                        Khess[:, :, i, j] = -2.0*w*gamma_tmp*K0
                        Khess[:, :, j, i] = Khess[:, :, i, j]
                    elif h1 == 'l' and h2 == 'l':
                        # d^2K/dl^2
                        Khess[:, :, i, j] = g*w2*Rglg/l**2*(g*Rglg-(g+1.0))*K0
                    elif h1 == 'l' and h2 == 'gamma':
                        # d^2K/dldgamma
                        Khess[:, :, i, j] = (w2/l*(Rglg + g*(1.0-Rglg) *
                                             gamma_tmp) * K0)
                        Khess[:, :, j, i] = Khess[:, :, i, j]
                    elif isinstance(h1, int) and isinstance(h2, int) and h1 == h2:
                        # d^2K/dl_h^2
                        Khess[:, :, i, j] = (g*w2*Rl[:, :, h1]**g/l[h1]**2 *
                                             (g*Rl[:, :, h1]**g-(g+1.0))*K0)
                    elif isinstance(h1, int) and isinstance(h2, int):
                        # d^2K/dl_h1 dl_h2
                        Khess[:, :, i, j] = (g**2*w2*(Rl[:, :, h1] *
                                             Rl[:, :, h2])**g /
                                             (l[h1]*l[h2])*K0)
                        Khess[:, :, j, i] = Khess[:, :, i, j]
                    elif isinstance(h1, int) and h2 == 'gamma':
                        # d^2K/dl_h1 dgamma
                        Khess[:, :, i, j] = (w2 * (g*tmp1[:, :, h1]/l[h1] +
                                             Rl[:, :, h1]**g/l[h1] *
                                             (1.0-g*gamma_tmp))*K0)
                        Khess[:, :, j, i] = Khess[:, :, i, j]
                    elif h1 == 'gamma' and h2 == 'gamma':
                        # d^2K/dgamma^2
                        tmp2 = zeros(Rk.shape)
                        tmp2[Rl > 0] = tmp1[Rl > 0] * log(Rl[Rl > 0.0])
                        Khess[:, :, i, j] = w2*(gamma_tmp**2 - sum(tmp2, 2))*K0
            return w2*K0, Kgrad, Khess
        if grad_r is not False:
            raise InputError("Gamma Exponential Kernel is not " +
                             "differentiable, need to separate kernels if " +
                             "differentiation is desired")


class RatQuad(Kernel):
    r"""
    Rational-quadratic kernel object.
    .. math::
        K(R; w, l, alpha) = w^2*( 1 + \frac{R^2}{2*\alpha*l^2} )^{-\alpha},
    with the parameters of weight, w, length, l, and length-variance
    parameter, alpha. The length can be a single value applied to all
    directions or a list with a separate value in each direction.
    Rational quadratic is SE over a gamma distribution of length scales
    with a mean of alpha*l^2 and variance of alpha*l^4.
    """
    def __init__(self, **params):
        p_bounds = odict([('w', (0.0, None)), ('l', (0.0, None)),
                          ('alpha', (0.0, None))])
        super(RatQuad, self).__init__(params, p_bounds)

    def __call__(self, Rk, grad_hp=False, grad_r=False, **kwargs):
        w, l, a = self.p['w'], self.p['l'], self.p['alpha']
        if not isinstance(l, list):
            R2l2 = sum(Rk**2, 2)/l**2
        else:
            R2l2 = zeros(Rk.shape[:2])
            for k in range(Rk.shape[2]):
                R2l2 += (Rk[:, :, k]/l[k])**2
        w2 = w**2
        all_tmp = 1.0 + R2l2/(2.0*a)
        K0 = all_tmp**(-a)
        if (not grad_hp) and (not grad_r):
            return w2*K0
        if grad_hp is not False:
            # First derivatives:
            Kgrad = empty((Rk.shape[0], Rk.shape[1], self.Nhp))
            for i, h in zip(range(self.Nhp), self.hp_id):
                if h == 'w':
                    # dK/dw
                    Kgrad[:, :, i] = 2.0*w*K0
                elif h == 'l':
                    # dK/dl
                    Kgrad[:, :, i] = w2*R2l2/(l*all_tmp)*K0
                elif isinstance(h, int):
                    # dK/dl_h
                    Kgrad[:, :, i] = w2*Rk[:, :, h]**2/(l[h]**3*all_tmp)*K0
                elif h == 'alpha':
                    # dK/dalpha
                    alpha_tmp = (all_tmp-1)/all_tmp - log(all_tmp)
                    Kgrad[:, :, i] = w2*alpha_tmp*K0
            if grad_hp != 'Hess':
                return w2*K0, Kgrad
            # Second derivatives:
            Khess = empty((Rk.shape[0], Rk.shape[1], self.Nhp, self.Nhp))
            for i, h1 in zip(range(self.Nhp), self.hp_id):
                for j, h2 in zip(range(i, self.Nhp), self.hp_id[i:]):
                    if h1 == 'w' and h2 == 'w':
                        # d^2K/dw^2
                        Khess[:, :, i, j] = 2.0*K0
                    elif h1 == 'w' and h2 == 'l':
                        # d^2K/dwdl
                        Khess[:, :, i, j] = 2.0*w*R2l2/(l*all_tmp)*K0
                        Khess[:, :, j, i] = Khess[:, :, i, j]
                    elif h1 == 'w' and isinstance(h2, int):
                        # d^2K/dwdl_h
                        Khess[:, :, i, j] = (2.0*w*Rk[:, :, h]**2 /
                                             (l[h]**3*all_tmp) * K0)
                        Khess[:, :, j, i] = Khess[:, :, i, j]
                    elif h1 == 'w' and h2 == 'alpha':
                        # d^2K/dwdalpha
                        Khess[:, :, i, j] = 2.0*w*alpha_tmp*K0
                        Khess[:, :, j, i] = Khess[:, :, i, j]
                    elif h1 == 'l' and h2 == 'l':
                        # d^2K/dl^2
                        Khess[:, :, i, j] = (w2*R2l2/l**2*((a+1.0)/a*R2l2 /
                                             all_tmp - 3.0)/all_tmp * K0)
                    elif h1 == 'l' and h2 == 'alpha':
                        # d^2K/dldalpha
                        Khess[:, :, i, j] = (w2*R2l2/l*((a+1.0)/a*(all_tmp-1.0) /
                                             all_tmp - log(all_tmp)) /
                                             all_tmp * K0)
                        Khess[:, :, j, i] = Khess[:, :, i, j]
                    elif isinstance(h1, int) and isinstance(h2, int) and h1 == h2:
                        # d^2K/dl_h^2
                        Khess[:, :, i, j] = (w2*Rk[:, :, h1]**2/l[h1]**4 *
                                             ((a+1.0)/a*(Rk[:, :, h1] /
                                              l[h1])**2/all_tmp - 3.0) /
                                             all_tmp * K0)
                    elif isinstance(h1, int) and isinstance(h2, int):
                        # d^2K/dl_h1 dl_h2
                        Khess[:, :, i, j] = ((a + 1.0)/a*w2*(Rk[:, :, h1] *
                                             Rk[:, :, h2]/all_tmp)**2 /
                                             (l[h1]*l[h2])**3 * K0)
                        Khess[:, :, j, i] = Khess[:, :, i, j]
                    elif isinstance(h1, int) and h2 == 'alpha':
                        # d^2K/dl_h dalpha
                        Khess[:, :, i, j] = (w2*Rk[:, :, h1]**2/l[h1]**3 *
                                             ((a+1.0)/a*(all_tmp-1.0)/all_tmp -
                                             log(all_tmp))/all_tmp*K0)
                        Khess[:, :, j, i] = Khess[:, :, i, j]
                    elif h1 == 'alpha' and h2 == 'alpha':
                        # d^2K/dalpha^2
                        Khess[:, :, i, j] = (w2*(R2l2**2/(4.0*a**3*all_tmp**2) +
                                             alpha_tmp**2)*K0)
            return w2*K0, Kgrad, Khess
        if grad_r is not False:
            # First derivatives:
            Kgrad = empty(Rk.shape)
            for i in range(Rk.shape[2]):
                if isinstance(l, list):
                    Kgrad[:, :, i] = (-w2*Rk[:, :, i]/l[i]**2 *
                                      all_tmp**(-(a+1.0)))
                else:
                    Kgrad[:, :, i] = (-w2*Rk[:, :, i]/l**2 *
                                      all_tmp**(-(a+1.0)))
            if grad_r != 'Hess':
                return w2*K0, Kgrad
            Khess = empty((Rk.shape[0], Rk.shape[1], Rk.shape[2], Rk.shape[2]))
            for i in range(Rk.shape[2]):
                for j in range(Rk.shape[2]):
                    if i == j:
                        if isinstance(l, list):
                            Khess[:, :, i, j] = (w2/l[i]**2*(((a+1.0)/a *
                                                 Rk[:, :, i]**2/l[i]**2) /
                                                 (1.0 + R2l2/(2.0*a)) - 1.0) *
                                                 all_tmp**(-(a+1.0)))
                        else:
                            Khess[:, :, i, j] = (w2/l**2*(((a+1.0)/a *
                                                 Rk[:, :, i]**2/l**2) /
                                                 (1.0 + R2l2/(2.0*a)) - 1.0) *
                                                 all_tmp**(-(a+1.0)))
                    else:
                        if isinstance(l, list):
                            Khess[:, :, i, j] = ((a+1.0)/a*(w2*Rk[:, :, i] *
                                                 Rk[:, :, j]/(l[i]**2*l[j]**2) *
                                                 all_tmp**(-(a+2.0))))
                        else:
                            Khess[:, :, i, j] = ((a+1.0)/a * (w2*Rk[:, :, i] *
                                                 Rk[:, :, j]/l**4 *
                                                 all_tmp**(-(a+2.0))))
            return w2*K0, Kgrad, Khess


class Error(Exception):
    """Base class for exceptions in the kernels module."""
    pass


class InputError(Error):  # -- not a ValueError? --
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
