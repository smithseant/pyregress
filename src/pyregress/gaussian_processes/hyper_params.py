# -*- coding: utf-8 -*-
"""
The module for the prior distributions for the hyper-parameters in `pyregress`.
Provided prior distributions log(P) and, if requested, gradients (log(P) and dlog(P)/dθ).

Created Sep 2013  @authors: Sean T. Smith & Benjamin B. Schroeder
"""
from typing import Literal
from abc import ABC, abstractmethod
from numpy import ndarray, empty, full, where, logical_and, inf, sqrt, exp, log, pi as π
from scipy.special import gamma, beta, erf, erfinv

# TODO:  Consider inverse-cumulative transformations to `uniform-ize` onto the unit box for MLSL.

class HyperPrior(ABC):
    """Abstract base class for hyper-parameter prior-distributions."""
    dimensionality: Literal["univariate", "multivariate"] = "univariate"  # default of two options
    depends_on_Xd: bool = False  # default value
    def __init__(self, *args, guess=None, trans=None, **kwargs):
        """
        Arguments:
            guess - initial guess for hyper-parameter,
            trans - which transformation to use for this variable,
            **kwargs - additional distribution-specific key-word arguments
        """
        self.guess = guess
        if trans is None:
            self.trans = trans
        elif trans == "log":
            self.trans = dict(forward=log, inverse=exp)
        elif trans == "probit":
            lo, hi = kwargs['bounds'] if 'bounds' in kwargs else (0, 1)
            self.trans = dict(forward=lambda x: sqrt(2) * erfinv(2 * (x - lo) / (hi - lo) - 1),
                              inverse=lambda z: (hi - lo) * (erf(z / sqrt(2)) + 1) / 2 + lo)
        # Distributions w/ transformed variables must adapt or redirect `ln_pdf` accordingly.
    def _finalize(self, Xd):
        """
        In case the initialization of this distribution `depends_on_Xd`, perform the steps to
        finalize the initialization.
        Arguments:
            Xd:  array - 1D or 2D,
                independent-variable observed values. The first index is for multiple observations,
                the second index is for multiple variables (dimensions of the X space).  The second
                index may be omitted for 1D problems.
        """
        if not self.depends_on_Xd:
            return None
    @abstractmethod
    def ln_pdf(self, x, ret_grad=False):
        """
        Arguments:
            x - current hyper-parameter value,
            ret_grad - bool (optional), when `ret_grad is True` also return `dln_p`.
        Returns:
            ln_p - log. of hyper-parameter prior,
            dln_p - derivative of ln_p (optional, when `ret_grad is True`).
        """
        pass


class Uniform(HyperPrior):
    """
    Uniform/Constant hyper-parameter prior with flexibility regarding the bounding of the support:
        f = const if inside of bounds, else zero,
        ln(f) =  cost. if inside of bounds, else -inf,
    Potentially representing:
      - a Wide/Flat non-informative prior when `bounds = (-inf, inf)`,
      - a naive prior on an interval,
      - or a degenerate beta (when: α = β = 1) => `bounds = (0, 1)`).
    Transformations:
      - "probit" (finite bnds): z = sqrt(2) * erf^{-1}( 2 * (x - bnd[0]) / (bnd[1] - bnd[0]) - 1)
                  =>  z ~ Normal(0, 1),
    """
    def __init__(self, bounds=(0, 1), guess=0, trans=None):
        super().__init__(guess=guess, trans=trans, bounds=bounds)
        self.lo, self.hi = bounds
        self.lo = max(self.lo, -9e15)
        self.hi = min(self.hi, +9e15)
        if trans == "probit":
            self.transformed = Normal(guess=self.trans['forward'](guess), μ=0, σ=1)
            self.ln_pdf = self.transformed.ln_pdf
    def ln_pdf(self, x, ret_grad=False):
        if not isinstance(x, ndarray):
            ln_p = -log(self.hi - self.lo) if self.lo <= x <= self.hi else -inf
        else:
            ln_p = where(logical_and(self.lo <= x, x <= self.hi), -log(self.hi - self.lo), -inf)
        if not ret_grad:
            return ln_p
        else:
            if not isinstance(x, ndarray):
                dln_p = 0
            else:
                dln_p = full(x.shape, 0, dtype='float8')
            return ln_p, dln_p

class Jeffreys(HyperPrior):
    """
    Jefferys' distribution for hyper-parameter priors (non-informative) for a variable with a
    support on [0, inf) such as w & l:
        f(x) = 1 / x  (A degenerate lognormal in the case that σ approaches infinity),
        ln(f) = -ln(x)
    Transformations:
      - "log": z = log(x)  =>  z ~ Wide/Flat
    """
    def __init__(self, guess=1, trans="log", **kwargs):
        super().__init__(guess=guess, trans=trans)
        if trans == "log":
            self.transformed = Uniform(guess=self.trans['forward'](guess), bounds=(-inf, inf))
            self.ln_pdf = self.transformed.ln_pdf
    def ln_pdf(self, x, ret_grad=False):
        ln_p = -log(x)
        if not ret_grad:
            return ln_p
        else:
            dln_p = -1 / x
            return ln_p, dln_p

class Normal(HyperPrior):
    """
    Normal (a.k.a. Gaussian) hyper-parameter prior distribution w/ a support on (-inf, inf):
        f(x; μ, σ) = 1 / \sqrt(2 π σ^2) * \exp(-(x - μ)^2 / (2 σ^2)),
        ln(f; μ, σ) = -1/2 [ ln(2 π σ^2) + ([x - μ] / σ)^2 ].
    """
    def __init__(self, guess=0, μ=0, σ=1, trans=None, **kwargs):
        super().__init__(guess=guess, trans=trans)
        self.μ = μ
        self.σ = σ
    def ln_pdf(self, x, ret_grad=False, **kwargs):
        ln_p = -0.5 * (log(2 * π) + 2 * log(self.σ) + ((x - self.μ) / self.σ)**2)
        if not ret_grad:
            return ln_p
        else:
            dln_p = -(x-self.μ) / self.σ**2
            return ln_p, dln_p

class LogNormal(HyperPrior):
    """
    Log-normal distribution for hyper-parameter priors w/ a support on [0, inf):
        f(x; μ, σ) = 1 / (x \sqrt(2 π σ^2)) * \exp(-(ln(x) - μ)^2 / (2 σ^2)) , x > 0,
        ln(f; μ, σ) = -1/2 [ ln(2 π σ^2) + 2 log(x) + ([ln(x) - μ] / σ)^2 ] , x > 0.
    Transformations:
      - "log": z = log(x)  =>  z ~ Normal(μ, σ)
    """
    def __init__(self, guess, μ=0, σ=1, trans="log", **kwargs):
        super().__init__(guess=guess, trans=trans)
        self.μ = μ
        self.σ = σ
        if trans == "log":
            self.transformed = Normal(guess=self.trans['forward'](guess), μ=μ, σ=σ)
            self.ln_pdf = self.transformed.ln_pdf
    def ln_pdf(self, x, ret_grad=False):
        ln_p = (-log(self.σ * x) - 0.5 * log(2 * π) -
                0.5 * ((log(x) - self.μ) / self.σ)**2)
        if not ret_grad:
            return ln_p
        else:
            dln_p = -1 / x - (log(x) - self.μ) / (self.σ**2 * x)
            return ln_p, dln_p
# class LogNormal_Xd(LogNormal):
#     """
#     A log-normal distribution for hyper-parameter priors, specifically correlation-length params,
#     with `μ` & `σ` inferred from the locations of the data (`Xd`) — optionally inferring `guess`.
#     """
#     depends_on_Xd = True
#     def __init__(self, Xd, guess=None, trans=None, **kwargs):
#         super().__init__(guess=guess, trans=trans)
#         R = radius(Xd, Xd, full(Xd.shape[0], 1, dtype='float64'))
#         lnR = log(abs(R))
#         n, my_sum = 0, 0.0
#         for i in range(1, R.shape[0]):
#             for j in range(i):
#                 n += 1
#                 my_sum += lnR[i, j]
#         self.μ = my_sum / n
#         my_sum = 0.0
#         for i in range(1, R.shape[0]):
#             for j in range(i):
#                 my_sum += (lnR[i, j] - self.μ)**2
#         self.σ = my_sum / n
#         if guess:
#             self.guess = guess
#         else:
#             self.guess = exp(self.μ + self.σ**2 / 2)
#         if trans is None:
#             self.transformation = False
#         else:
#             self.transformation = log
#             self.inv_trans = exp
#             self.transformed = Normal(guess=self.transformation(self.guess), μ=self.μ, σ=self.σ)
#     def finialize(self, Xd):
#         return super().finialize(Xd)

class Gamma(HyperPrior):
    """
    Gamma hyper-parameter priors w/ a support on [0, inf):
        f(x ; α , θ) = x^(α - 1) * exp(-x / θ) / (θ^α * Γ(α)),
        ln(f; α, θ) = (α - 1) ln(x) - x / θ - α ln(θ) - lnΓ(α)
    """
    def __init__(self, mean, std, guess, trans=None):
        self.guess = guess
        self.α = (mean / std)**2
        self.θ = std**2 / mean
        self.denomenator = self.α * log(self.θ) + log(gamma(self.α))
    def ln_pdf(self, x, ret_grad=False):
        k = self.α - 1.
        ln_p = k * log(x) - x / self.θ - self.denomenator
        if not ret_grad:
            return ln_p
        else:
            dln_p = k / x - 1 / self.θ
            return ln_p, dln_p

class Beta(HyperPrior):
    """
    Beta hyper-parameter prior for parameters with a support on [0, c]:
        f(y; α, β) = const * y^(α - 1) * (1 - y)^(β - 1),  where y = x / c,
        ln(f; α, β) = (α - 1) ln(y) + (β - 1) ln(1 -y) where y = x / c.
    """
    def __init__(self, α, β, c=1, guess=0.5, trans=None):
        super().__init__(guess=guess, trans=trans)
        self.α = α
        self.β = β
        self.c = c
        self.lnnorm = log(c * beta(α, β))
    def ln_pdf(self, x, ret_grad=False):
        y = x / self.c
        a = self.α - 1
        b = self.β - 1
        ln_p = (a * log(y) + b * log(1-y)) - self.lnnorm
        if not ret_grad:
            return ln_p
        else:
            dln_p = (a / y - b / (1- y)) / self.c
            return ln_p, dln_p


class GroupedSingleton:
    """Define a pattern for a singleton that can manually be reset for multiple groupings."""
    _instances = []
    _current_group = None
    def __new__(cls, new_group=False, group=None, **kwargs):
        """
        Append to or create a new grouped-singleton instance:
        This method ensures that calls to `GroupedSingleton()` will append to the previous object
        (or an existing one in the specified `group`), or it will create a new object only when
        it is the first of this class or when `new_group` is specified.
        """
        if not cls._instances or new_group:
            # create a new group...
            obj = super().__new__(cls)
            cls._current_group = obj.group_id = len(cls._instances)
            cls._instances.append(obj)
            obj._group_setup()
        elif group is not None:
            # refer to the specified group...
            obj = cls._instances[group]
        else:
            # refer to the current group...
            obj = cls._instances[cls._current_group]
        return obj
    def __init_subclass__(cls, **kwargs):
        """Ensure each subclass has independent groupings by providing the necessary attributes."""
        super().__init_subclass__(**kwargs)
        cls._instances = []
        cls._current_group = 0
    def _group_setup(self):
        pass
    def __init__(self, guess=None, **kwargs):
        pass

class HyperPrior_MultiVar(HyperPrior, GroupedSingleton):
    """
    Base class for hyper-parameter multivariate prior-distributions that can be constructed one
    variable at a time.  Unlike the "univariate" default, this "multivariate" alternative sets
    `guess` and `transformation` by variable.
    """
    dimensionality = "multivariate"
    def __init__(self, kernel_name, param_name, dim=None, guess=None, trans="log", **kwargs):
        self.unknowns[(kernel_name, param_name, dim)] = dict(guess=guess)
        if trans is None:
            self.unknowns[(kernel_name, param_name, dim)]["trans"] = trans
        elif trans == "log":
            self.unknowns[(kernel_name, param_name, dim)]["trans"] = dict(forward=log, inverse=exp)
    def _group_setup(self):
        self.knowns = dict()
        self.unknowns = dict()

class JointlyRobust(HyperPrior_MultiVar):
    """
    Define a class for a jointly robust prior in the spirit of Gu, Bayesian Analysis (2019).  In
    this implementation, the untransformed hyper-parameters are in the form that they are used by
    the kernels.
    Unknown variates are initialized one at a time in the order they will appear in `φ`, and
    cannot be mixed with other priors. The `param_name` must be  "σ" or "l"; and for "l" the
    corresponding index within `Xd` must be passed as `dim`.  The standard `guess` is `None`.
    """
    depends_on_Xd = True
    def _finalize(self, Xd, known_θ, use="emulation"):
        """
        For a `HyperPrior` object that `depends_on_Xd`, the initialization must be finalized by
        calling this method and passing `Xd`.  If "multivariate", finalization must occur after
        each of the individual variates is initialized and kernels are combined.
        Known parameters are expected to be passed in `known_θ`.
        """
        n_data, n_xdims = Xd.shape
        for (kernel_name, param_name, k) in self.unknowns.keys():
            if not hasattr(self, 'corr_kernel') and kernel_name != "Noise":
                self.corr_kernel = kernel_name
        for (kernel_name, param_name, k), param_val in known_θ.items():
            if (kernel_name, param_name, k) not in self.unknowns:
                self.knowns[(kernel_name, param_name, k)] = param_val
            if not hasattr(self, 'corr_kernel') and kernel_name != "Noise":
                self.corr_kernel = kernel_name
        # recommended parameter values:
        if use == "calibration":
            self.a = 0.5 - n_xdims
        elif use == "emulation":
            self.a = 0.2
        self.b = 1
        self.cs = n_data**(-1 / n_xdims) * (Xd.max(axis=0) - Xd.min(axis=0))
        # recommended initial guesses:
        r = (n_xdims + 1) * self.b / (n_xdims + 1 + self.a)
        if (self.corr_kernel, "σ", None) in self.knowns:
            σ_guess = self.knowns[(self.corr_kernel, "σ", None)]
        elif (σ_guess:=self.unknowns[(self.corr_kernel, "σ", None)]["guess"]) is None:
            σ_guess = self.unknowns[(self.corr_kernel, "σ", None)]["guess"] = 1
        for (kernel_name, param_name, k), var_dict in self.unknowns.items():
            if var_dict["guess"] is None:
                if kernel_name == "Noise" and param_name == "σ":
                    var_dict["guess"] = σ_guess / sqrt(r)
                elif param_name == "l":
                    if k is None:
                        var_dict["guess"] = self.cs[0] * r
                    else:
                        var_dict["guess"] = self.cs[k] * r
        # one time prep.:
        self.const_terms = 0
        for (kernel_name, param_name, k), param_val in self.knowns.items():
            if param_name == "l":
                if k is None:
                    self.const_terms += self.cs[0] / param_val
                else:
                    self.const_terms += self.cs[k] / param_val

    def ln_pdf(self, φ, ret_grad=False, **kwargs):
        my_sum = self.const_terms
        ln_p = 0
        if ("Noise", "σ", None) in self.knowns:
            σn = self.knowns[("Noise", "σ", None)]
        if (self.corr_kernel, "σ", None) in self.knowns:
            σ = self.knowns[(self.corr_kernel, "σ", None)]
        for ((kernel_name, param_name, k), φ_spec), φ_i in zip(self.unknowns.items(), φ):
            if kernel_name == "Noise" and param_name == "σ":
                if φ_spec["trans"] is None:
                    σn = φ_i
                    ln_p += log(σn)
                elif φ_spec["trans"]["forward"] == log:
                    σn = φ_spec["trans"]["inverse"](φ_i)
                    ln_p += 2 * φ_i
            elif kernel_name == self.corr_kernel and param_name == "σ":
                if φ_spec["trans"] is None:
                    σ = φ_i
                    ln_p -= 3 * log(σ)
                elif φ_spec["trans"]["forward"] == log:
                    σ = φ_spec["trans"]["inverse"](φ_i)
                    ln_p -= 2 * φ_i
            elif param_name == "l":
                if φ_spec["trans"] is None:
                    lk = φ_i
                    ln_p -= 2 * log(lk)
                elif φ_spec["trans"]["forward"] == log:
                    lk = φ_spec["trans"]["inverse"](φ_i)
                    ln_p -= φ_i
                my_sum += self.cs[k] / lk
        my_sum += (σn / σ)**2
        ln_p += self.a * log(my_sum) - self.b * my_sum
        if not ret_grad:
            return ln_p
        else:
            dln_p = empty(len(self.unkonwns))
            b_aS = (self.b - self.a / my_sum)
            for i, (((kernel, param, k), φ_spec), φ_i) in enumerate(zip(self.unknowns.items(), φ)):
                if kernel == "Noise" and param == "σ":
                    if φ_spec["trans"] is None:
                        dln_p[i] = 1 / σn - 2 * b_aS * σn / σ**2
                    elif φ_spec["trans"]["forward"] == log:
                        dln_p[i] = 2 - 2 * b_aS * (σn / σ)**2
                elif kernel == self.corr_kernel and param == "σ":
                    if φ_spec["trans"] is None:
                        dln_p[i] = 2 * b_aS * σn**2 / σ**3 - 3 / σ
                    elif φ_spec["trans"]["forward"] == log:
                        dln_p[i] = 2 * b_aS * (σn / σ)**2 - 2
                elif kernel == self.corr_kernel and param == "l":
                    if φ_spec["trans"] is None:
                        dln_p[i] = b_aS * self.cs[k] / φ_i**2 - 2 / φ_i
                    elif φ_spec["trans"]["forward"] == log:
                        dln_p[i] = b_aS * self.cs[k] * exp(-φ) - 1
            return ln_p, dln_p
