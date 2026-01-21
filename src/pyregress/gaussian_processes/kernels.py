# -*- coding: utf-8 -*-
"""
The Kernels, as used, provide a straightforward implementation of the equations they represent.
However, handling of kernel parameters is less straightforward due to the necessity to provide
flexibility to GPI.  The first half of this file is infrastructure setup, while only only the
second half is specification of the kernels.  Here is an overview of the approach:
  - Each prior object has a `dimensionality` attribute w/ value of "univariate" or "multivariate":
    - when "univariate", `guess` & `trans` are attributes of the prior directly;
    - when "multivariate",  the prior object has an `unknowns` dict attribute and each parameter
      in `unknowns` has a nested dict value containing its `guess` & `trans`.
    Also, the prior objects each have a callable attribute for the log of the prior, `ln_pdf`.
  - The Kernel object has a dict attribute, `θ`, that contains all parameter values in their
    untransformed state (as each value is used by its kernel).  The 1-D or isotropic lengthscales
    are stored as a scalar, while n-D independent lengthscales are stored as a list.
  - The uncertain hyper-parameters, `φ` (in memory as a 1-D array), are simply a subset of `θ`
    (w/ a transformation applied) — so a dict mapping, `map_θ2φ`, is constructed as an attribute
    of the kernel object to cleanly access the subset & transform it (when getting & setting `φ`).
  - Two more mappings are constructed as attributes of the kernel object:
    - `map_prior2φs` maps each prior to the indices in `φ` used when calling it (for `ln_prior`).
    - `map_prior2θs` maps each prior to the associated values in `θ` (for `_finalize`).

Created Sep 2013  @authors: Sean T. Smith & Benjamin B. Schroeder
"""
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from copy import deepcopy
import re
from numpy import (empty, full, eye, expand_dims, tile,
                   sum, abs, sqrt, exp, log, pi as π)
from numba import jit
from .hyper_params import HyperPrior

# TODO: Add a periodic distance & kernel (..but flexible for dims. that are not periodic.)
# TODO: I would love to add heuristics so users are required to specify less.


@jit(nopython=True)
def radius(x, y, scale):
    """Calculate the distance matrix (radius)."""
    # Originally used scipy.spatial.distance.cdist(X, Y, 'seuclidean', V=xscale);
    # Next, moved to numpy (using tile);
    # Currently prefer the simplicity of manually performing element operations & jitting w/ numba.
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
    Provide methods & an interface for kernels which support uncertain parameters (w/ `HyperPrior`)
    and which are ready to be used in the `GPI` class.

    Specific kernels will need to inherit this baseclass and define
    `__init__`, `__call__` & `Kφ` methods in the derived class.
    """
    def __init__(self, **kwargs):
        kernel_id = self.kernel_id
        self.θ = dict()
        self.map_θ2φ = dict()
        self.map_prior2φs = dict()
        self.map_prior2θs = dict()
        i_φ = 0
        for param_name, param in kwargs.items():
            for k, param_prior in enumerate(param) if isinstance(param, list) else [(None, param)]:
                if not isinstance(param_prior, HyperPrior):
                    self.set_θ_value(param_name, k, param_prior)
                else:
                    if param_prior.dimensionality == "univariate":
                        self.set_θ_value(param_name, k, param_prior.guess)
                        self.map_θ2φ[(param_name, k)] = (param_prior.trans, i_φ)
                        self.map_prior2φs[param_prior] = [i_φ]
                        self.map_prior2θs[param_prior] = (kernel_id, param_name, k)
                    elif param_prior.dimensionality == "multivariate":
                        prior_param_dict = param_prior.unknowns[(kernel_id[0], param_name, k)]
                        self.set_θ_value(param_name, k, prior_param_dict['guess'])
                        self.map_θ2φ[(param_name, k)] = (prior_param_dict['trans'], i_φ)
                        if param_prior not in self.map_prior2θs:
                            self.map_prior2φs[param_prior] = [i_φ]
                            self.map_prior2θs[param_prior] = [(kernel_id, param_name, k)]
                        else:
                            self.map_prior2φs[param_prior].append(i_φ)
                            self.map_prior2θs[param_prior].append((kernel_id, param_name, k))
                    i_φ += 1
        self.n_φ = i_φ

    def _finalize(self, Xd, **kwargs):
        my_id = self.kernel_id
        for prior, θ_refs in self.map_prior2θs.items():
            if prior.depends_on_Xd:
                prior._finalize(Xd, self.θ_flat, **kwargs)
                if prior.dimensionality == "univariate":
                    kernel_id, param_name, k = θ_refs
                    kern = self if kernel_id == my_id else self.terms[kernel_id]
                    kern.set_θ_value(param_name, k, prior.guess)
                elif prior.dimensionality == "multivariate":
                    for kernel_id, param_name, k in θ_refs:
                        kern = self if kernel_id == my_id else self.terms[kernel_id]
                        param_spec= prior.unknowns[(kernel_id[0], param_name, k)]
                        kern.set_θ_value(param_name, k, param_spec['guess'])

    def set_θ_value(self, param_name, k, θ_val):
        if k is None:
            self.θ[param_name] = θ_val
        elif param_name not in self.θ:
            self.θ[param_name] = [θ_val]
        elif k == len(self.θ[param_name]):
            self.θ[param_name].append(θ_val)
        else:
            self.θ[param_name][k] = θ_val

    @property
    def θ_flat(self):
        kernel_id = self.kernel_id[0]
        flatten = dict()
        for param_name, param in self.θ.items():
            for k, param_prior in enumerate(param) if isinstance(param, list) else [(None, param)]:
                flatten[(kernel_id, param_name, k)] = param_prior
        return flatten

    @property
    def φ(self):
        """
        Return a 1D array with the current values of the unknown hyper-parameters from `θ` with
        any transformations applied.
        """
        φ_vals = empty(self.n_φ)
        for (param_name, k), (trans, i) in self.map_θ2φ.items():
            θ_val = self.θ[param_name] if k is None else self.θ[param_name][k]
            φ_vals[i] = θ_val if trans is None else trans['forward'](θ_val)
        return φ_vals

    @φ.setter
    def φ(self, φ_vals):
        """
        Given a 1D array of transformed values for the unknown hyper-parameters, set untransformed
        values into their appropriate locations in `θ`.
        """
        for (param_name, k), (trans, i) in self.map_θ2φ.items():
            θ_val = φ_vals[i] if trans is None else trans['inverse'](φ_vals[i])
            self.set_θ_value(param_name, k, θ_val)
    
    def merge_φ2θ(self, φ_vals):
        θ_merged = deepcopy(self.θ)
        for (param_name, k), (trans, i) in self.map_θ2φ.items():
            θ_val = φ_vals[i] if trans is None else trans['inverse'](φ_vals[i])
            if k is None:
                θ_merged[param_name] = θ_val
            else:
                θ_merged[param_name][k] = θ_val
        return θ_merged

    def __add__(self, other):
        """Overload `+` so Kernel objects can be added."""
        if not isinstance(other, KernelSum):
            return KernelSum(self, other)  # Neither term is a `KernelSum` object, so create one.
        else:
            return other.__add__(self, self_on_right=True)  # Combine w/ the existing `KernelSum`.
    def __mul__(self, other):
        """Overload `*` so Kernel objects can be multiplied."""
        if not isinstance(other, KernelProd):
            return KernelProd(self, other)  # Neither term is a `KernelProd` object, so create one.
        else:
            return other.__mul__(self, self_on_right=True)  # Combine w/ the existing `KernelProd`.

    def ln_priors(self, φ, ret_grad=False):
        """
        Calculate log of prior distributions for hyper-parameters.

        Arguments
        ---------
        φ: array-1D,
            array of uncertain hyper-parameter values — transformed according to their own
            definitions, and ordered according to `self.map_φ2prior`.
        ret_grad: bool (optional),
            when ret_grad is True also return dlnprior.

        Returns
        -------
        lnprior: scalar value
            summation of values of log prior probabilities evaluated at values provided by params
        dlnprior: array-1D
            array of gradients of log prior probabilities evaluated at values provided by params
        """
        lnprior = 0
        if not ret_grad:
            for prior, prior_indices in self.map_prior2φs.items():
                lnprior += prior.ln_pdf(φ[prior_indices])
            return lnprior
        else:
            dlnprior = empty(self.n_φ)
            for prior, prior_indices in self.map_prior2φs.items():
                lnP, dlnP = prior.ln_pdf(φ[prior_indices], ret_grad=ret_grad)
                lnprior += lnP
                dlnprior[prior_indices] = dlnP
            return lnprior, dlnprior

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
    def Kφ(self, φ_vals, R, ret_grad=False, **kwargs):
        """
        Calculate kernel values given a temporary vector of hyper parameters and a radius array.
        (Beyond providing a necessarily unique call signature, this will often be a unique
         implementation of the equations compared to `kernel.__call__` — mainly because the
         gradients are with respect to the parameters rather than the abscissas.)

        Arguments
        ---------
        φ_vals: array-1D,
            array of hyper parameter values (transformed according to each variable's own setup),
        R: array-3D,
            directional distance matrix (distance between combination of points in each direction).
        ret_grad: bool (optional),
            indicate whether to return the gradients with respect to the transformed hyper
            parameters as Kgrad,
        kwargs: any additional options (opt_name=opt_value),
            specific options for specific kernels, otherwise ignored.

        Returns
        -------
        K: array-2D,
            kernel values - shape must match argument R,
        Kgrad: array-3D (optional - depending on argument ret_grad),
            partial of kernel (first two dimensions) with respect to each hyper param. (3rd dim.).
        """

    @property
    def kernel_id(self):
        """Return a string repr. of the kernel type."""
        return (re.search(r"kernels.(\w+)", repr(type(self))).group(1), id(self))


class CombiningKernel(Kernel):
    """
    This is a super class of KernelSum & KernelProd, created to avoid repetition of these methods.
    """
    def __init__(self, k1, k2):
        self.terms = {k1.kernel_id:k1, k2.kernel_id:k2}
        self.n_φ = k1.n_φ + k2.n_φ
        self.map_prior2φs = dict()
        self.map_prior2θs = dict()
        i_φ = 0
        for term in self.terms.values():
            for (prior, ind), θ_refs in zip(term.map_prior2φs.items(), term.map_prior2θs.values()):
                if prior not in self.map_prior2φs:
                    self.map_prior2φs[prior] = [i + i_φ for i in ind]
                    self.map_prior2θs[prior] = θ_refs
                else:
                    self.map_prior2φs[prior].extend([i + i_φ for i in ind])
                    self.map_prior2θs[prior].extend(θ_refs)
            i_φ += term.n_φ

    @property
    def θ(self):
        θ_combined = dict()
        for term_id, term in self.terms.items():
            θ_combined[term_id] = term.θ
        return θ_combined

    @property
    def θ_flat(self):
        θ_combined = dict()
        for term in self.terms.values():
            for key, val in term.θ_flat.items():
                θ_combined[key] = val
        return θ_combined

    @property
    def φ(self):
        φ_val = empty(self.n_φ)
        i_φ = 0
        for term in self.terms.values():
            φ_val[i_φ:(i_φ + term.n_φ)] = term.φ
            i_φ += term.n_φ
        return φ_val
    
    @φ.setter
    def φ(self, φ_vals):
        i_φ = 0
        for term in self.terms.values():
            term.φ = φ_vals[i_φ:(i_φ + term.n_φ)]
            i_φ += term.n_φ


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
            terms = list(self.terms.values())
        elif sum_terms == 'noisefree':
            terms = [kern for id, kern in self.terms.items() if id[0] != "Noise"]
        elif type(sum_terms) is list:
            terms = [kern for i, kern in enumerate(self.terms.values()) if i in sum_terms]
        elif type(sum_terms) is int:
            terms = [list(self.terms.values())[sum_terms]]
        K = 0
        for kern in terms:
            K += kern(R, **kwargs)
        return K

    def Kφ(self, φ_vals, R, ret_grad=False, **kwargs):
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
            for kern in terms.values():
                n_φ = kern.n_φ
                K += kern.Kφ(φ_vals[iφ:(iφ + n_φ)], R, **kwargs)
                iφ += n_φ
            return K
        else:
            K = full((ni, nj), 0, dtype='float64')
            Kgrad = full((ni, nj, self.n_φ), 0, dtype='float64')
            iφ = 0
            for kern in terms.values():
                n_φ = kern.n_φ
                Kt, Kgrad_t = kern.Kφ(φ_vals[iφ:(iφ + n_φ)], R, ret_grad, **kwargs)
                K += Kt
                Kgrad[:, :, iφ:(iφ + n_φ)] = Kgrad_t
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
            raise NotImplementedError("Products of kernels are only partially implemented.")
        self.n_φ += other.n_φ
        return self

    def __call__(self, R, i_grad=False, ii_grad=False, **kwargs):
        if i_grad or ii_grad:
            raise InputError("Kernel product does not currently support radial gradients.")
        K = self.terms[0](R, **kwargs)
        for kern in self.terms[1:]:
            K *= kern(R, **kwargs)
        return K

    def Kφ(self, φ_vals, R, ret_grad=False, **kwargs):
        ni, nj, n_xdims = R.shape
        if not ret_grad:
            K = full((ni, nj), 1, dtype='float64')
            iφ = 0
            for kern in self.terms:
                n_φ = kern.n_φ
                K *= kern.Kφ(φ_vals[iφ:iφ+n_φ], R, **kwargs)
                iφ += n_φ
            return K
        else:
            K = full(R.shape[1:], 1, dtype='float64')
            Kgrad = full((ni, nj, self.n_φ), 1, dtype='float64')
            iφ = 0
            for kern in self.terms:
                n_φ = kern.n_φ
                Kt, Kgt = kern.Kφ(φ_vals[iφ:(iφ + n_φ)], R, ret_grad, **kwargs)
                K *= Kt
                irange = range(iφ, iφ + n_φ)
                iother = list(range(0, iφ)) + list(range(iφ + n_φ, self.n_φ))
                Kgrad[:, :, irange] *= Kgt
                Kgrad[:, :, iother] *= tile(expand_dims(Kt, 2), (1, 1, len(iother)))
                iφ += n_φ
            return K, Kgrad


class Noise(Kernel):
    r"""
    White noise kernel object.
    ..math::
        K(R, data; σ) = σ^2 * I, or a zero matrix based on the data, with the prior std parameter,
        σ, and a flag indicating inclusion or not.
    White noise is discontinuous.
    """
    def __init__(self, σ):
        super().__init__(σ=σ)

    def __call__(self, R, i_grad=False, ii_grad=False, **kwargs):
        ni, nj, n_xdims = R.shape
        K = self.θ['σ']**2 * eye(ni, nj)
        if not i_grad or ii_grad:
            return K
        else:
            raise InputError("Noise Kernel is not differentiable, separate kernels before "
                             "differentiation.")

    def Kφ(self, φ_vals, R, ret_grad=False, **kwargs):
        ni, nj, n_xdims = R.shape
        θ = self.merge_φ2θ(φ_vals)
        σ = θ['σ']
        σ2 = σ**2
        K0 = eye(ni, nj)
        K = σ2 * K0
        if not ret_grad:
            return K
        else:
            Kgrad = empty((ni, nj, self.n_φ))
            if ("σ", None) in self.map_θ2φ:
                trans = self.map_θ2φ[('σ', None)][0]
                if trans is None:
                    Kgrad[:, :, 0] = 2 * σ  * K0
                elif trans['forward'] == log:
                    Kgrad[:, :, 0] = 2 * K
            return K, Kgrad


class SquareExp(Kernel):
    r"""
    Squared-exponential kernel object.
    .. math::
        K(R; σ, l) = σ^2 * \exp( -1/2 *(R/l)^2 ),
    with the parameters of prior std, σ, and length, l. For multiple dimensions, the length can be
    a single value applied to all directions or it can be a list with a separate value in each
    direction. Squared-exponential is continuous and infinitely differentiable.
    """
    def __init__(self, σ, l):
        super().__init__(σ=σ, l=l)

    def __call__(self, R, i_grad=False, ii_grad=False, s=None, **kwargs):
        ni, nj, n_xdims = R.shape
        σ, l_in = self.θ['σ'], self.θ['l']
        if not isinstance(l_in, Iterable):
            l = full(n_xdims, l_in)
        else:
            l = l_in
        K = empty((ni * (1 + (i_grad or ii_grad) * n_xdims), nj * (1 + ii_grad * n_xdims)))
        K[:ni, :nj] = σ**2 * exp(-0.5 * ((R / l)**2).sum(axis=2))
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

    def Kφ(self, φ_vals, R, ret_grad=False, **kwargs):
        ni, nj, n_xdims = R.shape
        θ = self.merge_φ2θ(φ_vals)
        σ, l_in = θ['σ'], θ['l']
        σ2 = σ**2
        if not isinstance(l_in, Iterable):
            l = full(n_xdims, l_in)
        else:
            l = l_in
        Rl2 = ((R / l)**2).sum(axis=2)
        K0 = exp(-0.5 * Rl2)
        K = σ2 * K0
        if not ret_grad:
            return K
        else:
            Kgrad = empty((ni, nj, self.n_φ,))
            iφ = 0
            if ('σ', None) in self.map_θ2φ:
                trans = self.map_θ2φ[('σ', None)][0]
                if trans is None:
                    Kgrad[:, :, iφ] = 2 * σ * K0
                elif trans['forward'] == log:
                    Kgrad[:, :, iφ] = 2 * K
                iφ += 1
            if ('l', None) in self.map_θ2φ:
                trans = self.map_θ2φ[('l', None)][0]
                if trans is None:
                    Kgrad[:, :, iφ] = Rl2 / l_in * K
                elif trans['forward'] == log:
                    Kgrad[:, :, iφ] = Rl2 * K
            else:
                for k in range(n_xdims):
                    if ('l', k) in self.map_θ2φ:
                        trans = self.map_θ2φ[('l', k)][0]
                        if trans is None:
                            Kgrad[:, :, iφ] = R[k, :, :]**2 / l[k]**3 * K
                        elif trans['forward'] == log:
                            Kgrad[:, :, iφ] = (R[k, :, :] / l[k])**2 * K
                        iφ += 1
            return K, Kgrad


class GammaExp(Kernel):
    r"""
    Gamma-exponential kernel object.
    .. math::
        K(R; σ, l, γ) = σ^2 \exp( -(R/l)^γ ),
    with the parameters of prior std, σ, length, l, and power norm, γ. For multiple dimensions, the
    length can be a single value applied to all directions or a list with a separate value in each
    direction. Gamma-exponential is continuous, and it is smooth only when γ=2.
    """
    def __init__(self, σ, l, γ):
        super().__init__(σ=σ, l=l, γ=γ)

    def __call__(self, R, i_grad=False, ii_grad=False, **kwargs):
        ni, nj, n_xdims = R.shape
        if i_grad or ii_grad:
            raise InputError("Gamma Exponential Kernel is not generally differentiable, need to "
                             "separate kernels if differentiation is desired")
        σ, l, γ = self.θ['σ'], self.θ['l'], self.θ['γ']
        if not isinstance(l, Iterable):
            Rl = abs(R / l)
        else:
            Rl = empty(R.shape)
            for k in range(n_xdims):
                Rl[:, :, k] = abs(R[k, :, :] / l[k])
        Rlγ = sum(Rl**γ, 2)
        σ2 = σ**2
        K0 = exp(-Rlγ)
        return σ2 * K0

    def Kφ(self, φ_vals, R, ret_grad=False, trans=False, **kwargs):
        ni, nj, n_xdims = R.shape
        θ = self.merge_φ2θ(φ_vals)
        σ, l_in, γ = θ['σ'], θ['l'], θ['γ']
        σ2 = σ**2
        if not isinstance(l_in, Iterable):
            R_l = abs(R / l_in)
        else:
            R_l = empty(R.shape)
            for k in range(n_xdims):
                R_l[k, :, :] = abs(R[k, :, :] / l_in[k])
        R_lγ = sum(R_l**γ, 2)
        K0 = exp(-R_lγ)
        K = σ2 * K0
        if not ret_grad:
            return K
        else:
            Kgrad = empty((ni, nj, self.n_φ))
            iφ = 0
            if ('σ', None) in self.map_θ2φ:
                trans = self.map_θ2φ[('σ', None)][0]
                if trans is None:
                    Kgrad[:, :, iφ] = 2 * σ  * K0
                elif trans['forward'] == log:
                    Kgrad[:, :, iφ] = 2 * K
                iφ += 1
            if ('l', None) in self.map_θ2φ:
                trans = self.map_θ2φ[('l', None)][0]
                if trans is None:
                    Kgrad[:, :, iφ] = γ * R_lγ / l_in * K
                elif trans['forward'] == log:
                    Kgrad[:, :, iφ] = γ * R_lγ * K
                iφ += 1
            else:
                for k in range(n_xdims):
                    if ('l', k) in self.map_θ2φ:
                        trans = self.map_θ2φ[('l', k)][0]
                        if trans is None:
                            Kgrad[:, :, iφ] = γ * R_l[k, :, :]**γ / l_in[k] * K
                        else:
                            Kgrad[:, :, iφ] = γ * R_l[k, :, :]**γ * K
                        iφ += 1
            if ('γ', None) in self.map_θ2φ:
                trans = self.map_θ2φ[('γ', None)][0]
                tmp1 = full(R.shape, 0, dtype='float64')
                tmp1[R_l > 0] = R_l[R_l > 0]**γ * log(R_l[R_l > 0])
                γ_tmp = sum(tmp1, 2)
                if trans is None:
                    Kgrad[:, :, iφ] = -γ_tmp * K
                elif trans['forward'] == log:
                    γtr = trans['forward'](γ)
                    c = self.map_prior2θs['γ'].c
                    dγdγtr = c * exp(-γtr**2 / 2) / sqrt(2 * π)
                    Kgrad[:, :, iφ] = -γ_tmp * K * dγdγtr
            return K, Kgrad


class RatQuad(Kernel):
    r"""
    Rational-quadratic kernel object.
    .. math::
        K(R; σ, l, \alpha) = σ^2*( 1 + \frac{R^2}{2*\alpha*l^2} )^{-\alpha},
    with the parameters of prior std, σ, length, l, and length-variance parameter, α. The length
    can be a single value applied to all directions or a list with a separate value in each
    direction.  Rational quadratic is SE over a gamma distribution of length scales with a mean of
    α*l^2 and variance of α*l^4.
    """
    def __init__(self, σ, l, α):
        super().__init__(σ=σ, l=l, α=α)

    def __call__(self, R, i_grad=False, ii_grad=False, s=None, **kwargs):
        ni, nj, n_xdims = R.shape
        σ, l_in, α = self.θ['σ'], self.θ['l'], self.θ['α']
        if not isinstance(l_in, Iterable):
            l = full(n_xdims, l_in, dtype='float64')
        else:
            l = l_in
        K = empty((ni * (1 + (i_grad or ii_grad) * n_xdims), nj * (1 + ii_grad * n_xdims)))
        base = 1 + ((R / ℓ)**2).sum(axis=2) / (2 * α)
        K[:ni, :nj] =  σ**2 * base**(-α)
        if i_grad or ii_grad:
            for k in range(n_xdims):
                lo, hi = ni * (k + 1), ni * (k + 2)
                K[lo:hi, :nj] = -σ**2 * R[:, :, k] / (s[k] * l[k]**2) * base**(-(α + 1))
            if ii_grad:
                K[:ni, nj:] = -K[ni:, :nj].T
                for ki in range(n_xdims):
                    i_lo, i_hi = ni * (ki + 1), ni * (ki + 2)
                    for kj in range(n_xdims):
                        δ = 1 if ki == kj else 0
                        j_lo, j_hi = nj * (kj + 1), nj * (kj + 2)
                        K[i_lo:i_hi, j_lo:j_hi] = (σ**2 / (s[ki] * l[ki] * s[kj] * l[kj])
                          (δ - (α + 1) / α * (R[:, :, ki] / l[ki]) * (R[:, :, kj] / l[kj]) / base) *
                          base**(-(α + 1)))
        return K

    def Kφ(self, φ_vals, R, ret_grad=False, trans=False, **kwargs):
        ni, nj, n_xdims = R.shape
        θ = self.merge_φ2θ(φ_vals)
        σ, l_in, α = θ['σ'], θ['l'], θ['α']
        if not isinstance(l_in, Iterable):
            l = full(n_xdims, l_in, dtype='float64')
        else:
            l = l_in
        R2l2 = ((R / l)**2).sum(axis=2)
        base = 1 + R2l2 / (2 * α)
        K =  σ**2 * base**(-α)

        if not ret_grad:
            return K
        else:
            Kgrad = empty((ni, nj, self.n_φ,))
            iφ = 0
            if ('σ', None) in self.map_θ2φ:
                trans = self.map_θ2φ[('σ', None)][0]
                if trans is None:
                    Kgrad[:, :, iφ] = 2 * σ  * base**(-α)
                elif trans['forward'] == log:
                    Kgrad[:, :, iφ] = 2 * K
                iφ += 1
            if ('l', None) in self.map_θ2φ:
                trans = self.map_θ2φ[('l', None)][0]
                if trans is None:
                    Kgrad[:, :, iφ] = R2l2 / (l_in * base) * K
                elif trans['forward'] == log:
                    Kgrad[:, :, iφ] = R2l2 / base * K
                iφ += 1
            else:
                for k in range(n_xdims):
                    if ('l', k) in self.map_θ2φ:
                        trans = self.map_θ2φ[('l', k)][0]
                        if trans is None:
                            Kgrad[:, :, iφ] = R[k, :, :]**2 / (l[k]**3 * base) * K
                        elif trans['forward'] == log:
                            Kgrad[:, :, iφ] = R[k, :, :]**2 / (l[k]**2 * base) * K
                        iφ += 1
            if ('α', None) in self.map_θ2φ:
                trans = self.map_θ2φ[('α', None)][0]
                α_tmp = 1 - 1 / base - log(base)
                if trans is None:
                    Kgrad[:, :, iφ] = α_tmp * K
                elif trans['forward'] == log:
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
