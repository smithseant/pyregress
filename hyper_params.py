# -*- coding: utf-8 -*-
"""
The module for the hyper-parameters

Includes additional features for pyregress including prior distributions for
    hyper-parameters and a class for derivative inputs. Can only handle
    independent priors.

Provided prior distributions (log(P) and dlog(P))
    Currently includes constant, normal, Jeffreys', log-Normal, gamma,
    uniform, beta, and bounded constant distributions.

Created Sep 2013  @authors: Sean T. Smith & Benjamin B. Schroeder
"""
__all__ = ['HyperPrior', 'Constant', 'Normal', 'Jeffreys', 'LogNormal',
           'Gamma', 'Uniform', 'Beta', 'Bounded']

from numpy import array, inf, sqrt, exp, log, pi as π
from scipy.special import gamma, beta, erf, erfinv

# Prior distribution options for the hyper-parameters

class HyperPrior:
    """
    Base class for hyper-parameter prior-distribution classes. Subclasses
    provide different probability distributions and take the general form of:
    Init:

    """
    def __init__(self, *args, guess=1, **kwargs):
        """
        Arguments:
            guess - initial guess for hyper-parameter
            **kwargs - additional distribution-specific key-word arguments
        """
        self.guess = guess
    
    def __call__(self, x, grad=None):
        """
        Arguments:
            x - current hyper-parameter value,
            grad - bool (optional), when grad is True also return dlnpdf.
        Returns:
            lnprior - ln of hyper-parameter prior,
            dlnprior - derivative of lnprior (optional, grad=True).
        """
        pass


class Constant(HyperPrior):
    """
    Constant hyper-parameter prior (non-informative prior) for a variable
    with a support of (-inf, inf):
       f = const.
    (A degenerate normal in the case that σ approaches infinity.)
    """
    def __call__(self, x=None, grad=False, **kwargs):
        if not x:
            x = self.guess
        if not grad:
            return 1
        else:
            return 1, 0


class Normal(HyperPrior):
    """
    Normal (a.k.a. Gaussian) hyper-parameter prior
        f(x; μ, σ) = 1 / \sqrt(2 π σ^2) * \exp(-(x - μ)^2 / (2 σ^2)).
    """
    def __init__(self, guess=0, μ=0, σ=1, **kwargs):
        # TODO: Add a heuristic to calculate guess, μ & σ from Rk2.
        self.guess = guess
        self.μ = μ
        self.σ = σ

    def __call__(self, x=None, grad=False, **kwargs):
        if not x:
            x = self.guess
        lnpdf = -log(self.σ) - 0.5*log(2*π) - 0.5 * ((x - self.μ) / self.σ)**2
        if not grad:
            return lnpdf
        else:
            dlnpdf = -(x-self.μ) / self.σ**2
            return lnpdf, dlnpdf


class Jeffreys(HyperPrior):
    """
    Jefferys' distribution for hyper-parameter priors (non-informative prior)
    for a variable with a support of [0, inf) such as w & l:
        f(x) = 1 / x
    (A degenerate lognormal in the case that σ approaches infinity.)
    """
    def __init__(self, guess=1, **kwargs):
        # TODO: Add a heuristic to calculate guess, μ & σ from Rk2.
        super().__init__(self, guess=guess, **kwargs)
        self.trans = log
        self.invtr = exp
        trguess = self.trans(guess)
        self.transformed = Constant(guess=trguess, **kwargs)

    def __call__(self, x=None, grad=False, trans=False):
        if x is None:
            x = self.guess
        if trans:
            return self.transformed(x=x, grad=grad, trans=False)
        lnpdf = -log(x)
        if not grad:
            return lnpdf
        else:
            dlnpdf = -1 / x
            return lnpdf, dlnpdf


class LogNormal(HyperPrior):
    """
    Log Normal distribution class for hyper-parameter priors:
        f(x; μ, σ) = 1 / (x * σ * \sqrt(2 π)) *
                     \exp(-(ln(x) - μ)^2 / (2 σ^2)) , x > 0
    """
    def __init__(self, guess=1, μ=0, σ=1, **kwargs):
        # TODO: Add a heuristic to calculate guess, μ & σ from Rk2.
        self.guess = guess
        self.μ = μ
        self.σ = σ
        self.trans = log
        self.invtr = exp
        trguess = self.trans(guess)
        self.transformed = Normal(guess=trguess, μ=μ, σ=σ, **kwargs)

    def __call__(self, x=None, grad=False, trans=False):
        if not x:
            x = self.guess
        if trans:
            return self.transformed(x=x, grad=grad, trans=False)
        lnpdf = (-log(self.σ * x) - 0.5 * log(2 * π) -
                 0.5 * ((log(x) - self.μ) / self.σ)**2)
        if not grad: 
            return lnpdf
        else:
            dlnpdf = -1 / x - (log(x) - self.μ) / (self.σ**2 * x)
            return lnpdf, dlnpdf


class Gamma(HyperPrior):
    """
    Gamma hyper-parameter priors
        f(x ; k , θ) = x^(k - 1) * exp(-x / θ) / (θ^k * Γ(k))
    """

    def __init__(self, mean, std, guess=1):
        self.guess = guess
        self.k = (mean / std)**2
        self.θ = std**2 / mean
        self.denomenator = self.k * log(self.θ) + log(gamma(self.k))

    def __call__(self, x=None, grad=False):
        if x is None:
            x = self.guess
        k = self.k - 1.
        lnpdf = k * log(x) - x / self.θ - self.denomenator
        if not grad:
            return lnpdf
        else:
            dlnpdf = k / x - 1 / self.θ
            return lnpdf, dlnpdf


class Uniform(HyperPrior):
    """
    Uniform hyper-parameter prior for parameters with a support of [0, c]:
        f(y) = 1 when 0 < y < 1, and 0 otherwise,  where y = x / c.
    (A degenerate beta in the case that α = β = 1.)
    """
    def __init__(self, c=1, guess=0.5, **kwargs):
        self.guess = guess
        self.c = c
        self.lnnorm = log(c)
        self.trans = lambda u: sqrt(2) * erfinv(2 * u / c - 1)
        self.invtr = lambda z: c * (erf(z / sqrt(2)) + 1) / 2
        trguess = self.trans(guess)
        self.transformed = Normal(guess=trguess, μ=0, σ=1, **kwargs)

    def __call__(self, x=None, grad=False, trans=False):
        if not x:
            x = self.guess
        if trans:
            return self.transformed(x=x, grad=grad, trans=False)
        y = x / self.c
        lnpdf = -self.lnnorm if 0 < y < 1 else 0
        if not grad:
            return lnpdf
        else:
            dlnpdf = 0
            return lnpdf, dlnpdf


class Beta(HyperPrior):
    """
    Beta hyper-parameter prior for parameters with a support of [0, c]:
        f(y; α, β) = const * y^(α - 1) * (1 - y)^(β - 1),  where y = x / c.
    """
    def __init__(self, α, β, c=1, guess=0.5):
        self.guess = guess
        self.α = α
        self.β = β
        self.c = c
        self.lnnorm = log(c * beta(α, β))

    def __call__(self, x=None, grad=False):
        if not x:
            x = self.guess
        y = x / self.c
        a = self.α - 1
        b = self.β - 1
        lnpdf = (a * log(y) + b * log(1-y)) - self.lnnorm
        if not grad:
            return lnpdf
        else:
            dlnpdf = (a / y - b / (1- y)) / self.c
            return lnpdf, dlnpdf


class Bounded(HyperPrior):
    """
    Uniform/Constant hyper-parameter prior with flexibility regarding
    the bounding of the support:
       f = const if inside of bounds, else zero
    (Meant for use when optimizing likelihood.)
    """
       
    def __init__(self, low_b=-inf, high_b=inf, guess=1):
        self.low  = low_b
        self.high = high_b
        self.guess = guess

    def __call__(self, x=None, grad=False):
        if x is None:
            x = self.guess

        in_bounds = self.low < x < self.high
        if in_bounds:
            val = 1
        else:
            val = -inf
            
        if not grad:
            return array([val])
        else:
            return array([val]), array([0.0])
