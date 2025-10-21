# -*- coding: utf-8 -*-
"""
The module for the hyper-parameters

Includes additional features for `pyregress` including prior distributions for hyper-parameters and
   a class for derivative inputs. Currently only handles independent priors.

Provided prior distributions log(P) and, if requested, gradients (log(P) and dlog(P)).

Created Sep 2013  @authors: Sean T. Smith & Benjamin B. Schroeder
"""
from typing import Literal
from numpy import array, full, inf, sqrt, exp, log, pi as π
from scipy.special import gamma, beta, erf, erfinv
# from .kernels import radius

# Prior distribution options for the hyper-parameters

class HyperPrior:
    """Base class for hyper-parameter prior-distributions."""
    dimensionality: Literal["univariate", "multivariate"] = "univariate"  # default of two options
    depends_on_Xd: bool = False  # default value
    def __init__(self, *args, guess=1, trans=None, **kwargs):
        """
        Arguments:
            guess - initial guess for hyper-parameter,
            trans - which transformation to use for this variable,
            **kwargs - additional distribution-specific key-word arguments
        """
        self.guess = guess
        if trans is None:
            self.transformation = False
    def finialize(self, Xd):
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
    def __call__(self, x, ret_grad=False):
        """
        Arguments:
            x - current hyper-parameter value,
            ret_grad - bool (optional), when ret_grad is True also return dlnpdf.
        Returns:
            lnprior - ln of hyper-parameter prior,
            dlnprior - derivative of lnprior (optional, ret_grad=True).
        """
        pass

# class HyperPrior_Multivariate(HyperPrior)

class Constant(HyperPrior):
    """
    Const (wide/flat) hyper-parameter prior (non-informative) distribution for variables w/ a
    support on (-inf, inf):
       f(x) = const.
       ln(f) = const.
    (A degenerate normal distribution in the case that σ approaches infinity.)
    """
    dimensionality = "univariate"
    def __call__(self, x, ret_grad=False, **kwargs):
        if not ret_grad:
            return 1
        else:
            return 1, 0

class Normal(HyperPrior):
    """
    Normal (a.k.a. Gaussian) hyper-parameter prior distribution w/ a support on (-inf, inf):
        f(x; μ, σ) = 1 / \sqrt(2 π σ^2) * \exp(-(x - μ)^2 / (2 σ^2)),
        ln(f; μ, σ) = -1/2 [ ln(2 π σ^2) + ([x - μ] / σ)^2 ].
    """
    dimensionality = "univariate"
    def __init__(self, guess=0, μ=0, σ=1, trans=None, **kwargs):
        super().__init__(guess=guess, trans=trans)
        self.μ = μ
        self.σ = σ
    def __call__(self, x, ret_grad=False, **kwargs):
        lnpdf = -0.5 * (log(2 * π) + 2 * log(self.σ) + 0.5 * ((x - self.μ) / self.σ)**2)
        if not ret_grad:
            return lnpdf
        else:
            dlnpdf = -(x-self.μ) / self.σ**2
            return lnpdf, dlnpdf

class Jeffreys(HyperPrior):
    """
    Jefferys' distribution for hyper-parameter priors (non-informative) for a variable with a
    support on [0, inf) such as w & l:
        f(x) = 1 / x  (A degenerate lognormal in the case that σ approaches infinity),
        ln(f) = -ln(x)
    Transformations:
      - "log": z = log(x)  =>  z ~ Wide/Flat
    """
    dimensionality = "univariate"
    def __init__(self, guess=1, trans=None, **kwargs):
        super().__init__(guess=guess, trans=trans)
        if trans == "log":
            self.transformation = log
            self.inv_trans = exp
            self.transformed = Constant(guess=self.transformation(guess), **kwargs)
    def __call__(self, x, ret_grad=False):
        if not self.transformation:
            lnpdf = -log(x)
            if not ret_grad:
                return lnpdf
            else:
                dlnpdf = -1 / x
                return lnpdf, dlnpdf
        else:
            return self.transformed(x=x, ret_grad=ret_grad)

class LogNormal(HyperPrior):
    """
    Log-normal distribution for hyper-parameter priors w/ a support on [0, inf):
        f(x; μ, σ) = 1 / (x \sqrt(2 π σ^2)) * \exp(-(ln(x) - μ)^2 / (2 σ^2)) , x > 0,
        ln(f; μ, σ) = -1/2 [ ln(2 π σ^2) + 2 log(x) + ([ln(x) - μ] / σ)^2 ] , x > 0.
    Transformations:
      - "log": z = log(x)  =>  z ~ Normal(μ, σ)
    """
    dimensionality = "univariate"
    def __init__(self, guess, μ=0, σ=1, trans=None, **kwargs):  # TODO: After testing, change the default to `trans="log"`
        super().__init__(guess=guess, trans=trans)
        self.μ = μ
        self.σ = σ
        if trans == "log":
            self.transformation = log
            self.inv_trans = exp
            self.transformed = Normal(guess=self.transformation(guess), μ=μ, σ=σ, **kwargs)
    def __call__(self, x, ret_grad=False):
        if not self.transformation:
            lnpdf = (-log(self.σ * x) - 0.5 * log(2 * π) -
                    0.5 * ((log(x) - self.μ) / self.σ)**2)
            if not ret_grad: 
                return lnpdf
            else:
                dlnpdf = -1 / x - (log(x) - self.μ) / (self.σ**2 * x)
                return lnpdf, dlnpdf
        else:
            return self.transformed(x, ret_grad)
# class LogNormal_Xd(LogNormal):
#     """
#     A log-normal distribution for hyper-parameter priors, specifically correlation-length params,
#     with `μ` & `σ` inferred from the locations of the data (`Xd`) — optionally inferring `guess`.
#     """
#     dimensionality = "univariate"
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
    dimensionality = "univariate"
    def __init__(self, mean, std, guess=1, trans=None):
        self.guess = guess
        self.α = (mean / std)**2
        self.θ = std**2 / mean
        self.denomenator = self.α * log(self.θ) + log(gamma(self.α))
    def __call__(self, x, ret_grad=False):
        k = self.α - 1.
        lnpdf = k * log(x) - x / self.θ - self.denomenator
        if not ret_grad:
            return lnpdf
        else:
            dlnpdf = k / x - 1 / self.θ
            return lnpdf, dlnpdf

class Uniform(HyperPrior):
    """
    One sided uniform hyper-parameter prior for parameters with a support on [0, c]:
        f(y) = const. when 0 < y < 1, and 0 otherwise,  where y = x / c,
        ln(f) = const. when 0 < y < 1, and 0 otherwise,  where y = x / c,
        (A degenerate beta in the case that α = β = 1)
    Transformations:
      - "normalizing": z = sqrt(2) * erf^{-1}( 2 * x / c - 1)  =>  z ~ Normal(0, 1).
    """
    dimensionality = "univariate"
    def __init__(self, c=1, guess=0.5, trans=None, **kwargs):
        super().__init__(guess=guess, trans=trans)
        self.c = c
        self.lnnorm = log(c)
        if trans is None:
            self.transformation = False
        else:
            self.transformation = lambda u: sqrt(2) * erfinv(2 * u / c - 1)
            self.inv_tran = lambda z: c * (erf(z / sqrt(2)) + 1) / 2
            self.transformed = Normal(guess=self.transformation(guess), **kwargs)
    def __call__(self, x=None, ret_grad=False):
        if not self.transformation:
            y = x / self.c
            lnpdf = -self.lnnorm if 0 < y < 1 else 0
            if not ret_grad:
                return lnpdf
            else:
                dlnpdf = 0
                return lnpdf, dlnpdf
        else:
            return self.transformed(x=x, ret_grad=ret_grad)

class Rectangular(HyperPrior):
    """
    Uniform/Constant hyper-parameter prior with flexibility regarding the bounding of the support:
       f = const if inside of bounds, else zero,
       ln(f) =  cost. if inside of bounds, else zero,
    (Meant for use when optimizing likelihood.)
    """
    dimensionality = "univariate"
    def __init__(self, low_b=-inf, high_b=inf, guess=1, trans=None):
        super().__init__(guess=guess, trans=trans)
        self.low  = low_b
        self.high = high_b
    def __call__(self, x, ret_grad=False):
        in_bounds = self.low < x < self.high
        if in_bounds:
            val = 1
        else:
            val = -inf
            
        if not ret_grad:
            return array([val])
        else:
            return array([val]), array([0.0])

class Beta(HyperPrior):
    """
    Beta hyper-parameter prior for parameters with a support on [0, c]:
        f(y; α, β) = const * y^(α - 1) * (1 - y)^(β - 1),  where y = x / c,
        ln(f; α, β) = (α - 1) ln(y) + (β - 1) ln(1 -y) where y = x / c.
    """
    dimensionality = "univariate"
    def __init__(self, α, β, c=1, guess=0.5, trans=None):
        super().__init__(guess=guess, trans=trans)
        self.α = α
        self.β = β
        self.c = c
        self.lnnorm = log(c * beta(α, β))
    def __call__(self, x, ret_grad=False):
        y = x / self.c
        a = self.α - 1
        b = self.β - 1
        lnpdf = (a * log(y) + b * log(1-y)) - self.lnnorm
        if not ret_grad:
            return lnpdf
        else:
            dlnpdf = (a / y - b / (1- y)) / self.c
            return lnpdf, dlnpdf


class GroupedSingleton:
    """Define a pattern for a singleton that can manually be reset for multiple groupings."""
    _instances = []
    _current_group = None
    def __new__(cls, new_group=False, group=None, **kwargs):
        """
        Append to or create a grouped-singleton instance:
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
    @classmethod
    def _group_setup(self):
        pass
    def __init__(self, guess=1, **kwargs):
        pass

class HyperPrior_MultiVar(HyperPrior, GroupedSingleton):
    """
    Base class for hyper-parameter multivariate prior-distributions that can be constructed one
    variable at a time.
    """
    dimensionality = "multivariate"

class JointlyRobust(HyperPrior_MultiVar):
    depends_on_Xd = True
    def _group_setup(self):
        self.σn2_unknown = False
        self.σ2_unknown = False
        self.l_unknown = []
    def __init__(self, role, dim=None, **kwargs):
        if role == "σn2":
            self.σn2_unknown = True
        elif role == "σ2":
            self.σ2_unknown = True
        elif role == "l":
            self.l_unknown.append(dim)
    # TODO:  Provide the capability to combine two groups (needed for `Noise() + SquareExp()`)
    def _finalize(self, Xd, goal="emulation", **kwargs):
        n_data, n_xdims = Xd.shape
        # variable mappings:
        i = 0
        self.known = dict()
        # TODO:  The order of these parameters must correspond to the order in the kernel mapping.
        if self.σn2_unknown:
            self.ind_σn2 = i
            i += 1
        elif "σn2" in kwargs:
            self.known["σn2"] = kwargs["σn2"]
        if self.σ2_unknown:
            self.ind_σ2 = i
            i += 1
        else:
            self.known["σ2"]  = kwargs["σ2"]
        self.l_unknown.sort()
        if self.l_unknown:
            self.ind_l = list(range(i, i + len(self.l_unknown)))
        if "l" in kwargs:
            self.known["l"] = kwargs["l"]
            self.known["ind_l"] = kwargs["ind_l"]
        # recommended parameter values:
        if goal == "calibration":
            self.a = 0.5 - n_xdims
        elif goal == "emulation":
            self.a = 0.2
        self.b = 1
        x_range = (Xd.max(axis=0) - Xd.min(axis=0))
        self.c_unknown = n_data**(-1 / n_xdims) * x_range[self.l_unknown]
        self.c_known = n_data**(-1 / n_xdims) * x_range[self.known["ind_l"]]
    def __call__(self, x, ret_grad=False, **kwargs):
        # TODO:  Deal with transformations.
        my_sum = 0
        if self.l_unknown:
            my_sum += (self.c_unknown / x[self.ind_l]).sum()
        if "l" in self.known:
            my_sum += (self.c_known / self.known["l"]).sum()
        σ2 = x[self.ind_σ2] if self.σ2_unknown else self.known["σ2"]
        if self.σn2_unknown:
            my_sum += x[self.ind_σn2] / σ2
        elif "σn2" in self.known:
            my_sum += self.known["σ2"] / σ2
        lnpdf = self.a * log(my_sum) - self.b * my_sum
        if self.σ2_unknown:
            lnpdf -= log(σ2)  # Account for the Jeffrey's prior on σ2.
        if not ret_grad:
            return lnpdf
        else:
            raise NotImplementedError
        #     return lnpdf, dlnpdf
