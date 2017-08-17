# -*- coding: utf-8 -*-
"""
The module for the hyper-parameters

Includes additional features for pyregress including prior distributions for
    hyper-parameters and a class for derivative inputs

Provided prior distributions (log(P) and d_log(P))
    Currently includes constant, Jeffreys', log-Normal, beta, gamma, and
    bounded constant distributions.
"""
__all__ = ['HyperPrior', 'Constant', 'Jeffreys', 'LogNormal', 'Beta', 'Gamma', 'Bounded']

from numpy import array, sum, divide, inf, log, sqrt
from numpy import pi as π
from scipy.special import gamma

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
            *args - additional distribution-specific arguments
            guess - initial guess for hyper-parameter
            **kwargs - additional distribution-specific key-word arguments
        """
        self.guess = guess
    
    def __call__(self, x, grad=None):
        """
        Arguments:
            x - current hyper-parameter value
            grad - bool (optional), when grad is True also return dlnpdf,
                    and when grad is 'Hess' also return d2lnpdf.
        Returns:
            lnprior - ln of hyper-parameter prior
            dlnprior - derivative of lnprior (optional, grad=True)
            d2lnprior - 2nd derivative of lnprior (optional, grad='Hess')
        """
        pass


class Constant(HyperPrior):
    """
    Constant hyper-parameter prior
       f = const.
    """

    def __call__(self, x=None, grad=False):
        if x is None:
            x = self.guess
        if not grad:
            return array([1.0])
        if grad is True:
            return array([1.0]), array([0.0])
        if grad == 'Hess':
            return array([1.0]), array([0.0]), array([0.0])


class Jeffreys(HyperPrior):
    """
    Jefferys' distribution for hyper-parameter priors (Non informative prior):
        f(x) = 1/x
    """

    def __call__(self, x=None, grad=False):
        if x is None:
            x = self.guess
        lnpdf = -log(x)
        if not grad:
            return lnpdf
        dlnpdf = divide(-1, x)
        if grad is True:
            return lnpdf, dlnpdf
        if grad == 'Hess':
            d2lnpdf = divide(1, x**2)
            return lnpdf, dlnpdf, d2lnpdf


class LogNormal(HyperPrior):
    """
    Log Normal distribution class for hyper-parameter priors.
        f(x; mu, sigma) = 1/(x * sigma * sqrt(2 π)) *
                          exp(-(ln(x) - mu)^2 / (2 * sigma^2)) , x > 0
    """

    def __init__(self, guess=1, **kwargs):
        self.guess = mean = guess
        if  "std" in kwargs:
            self._μ = log(mean**2 / sqrt(kwargs["std"]**2 + mean**2))
            self._σ = sqrt(log(1 + kwargs["std"]**2/mean**2))
    
    def auto_fill(self, Rk2):
        mean = sum(Rk2) / (Rk2 != 0).sum()
        std = sum((Rk2 - mean)**2) / ((Rk2 != 0).sum() - 1.0)
        self._μ = log(mean**2/(std**2 + mean**2))
        self._σ = sqrt(log(1 + std**2/mean**2))
        
    def __call__(self, x=None, grad=False):
        # TODO: if mean and std not defined, use auto_fill with Rk2      
        # if not hasattr(self, '_μ'):
        #   self.auto_fill(self.Rk2)
        if x is None:
            x = self.guess
        lnpdf = (-log(self._σ*x) - 0.5 * log(2.0 * π) -
                 (log(x) - self._μ)**2 / 2 * self._σ**2)
        if not grad: 
            return lnpdf
        dlnpdf = (self._μ - self._σ**2 - log(x)) / (self._σ**2 * x)
        if grad is True:
            return lnpdf, dlnpdf
        if grad == 'Hess':             
            d2lnpdf = (-self._μ + self._σ**2 + log(x) - 1) / (self._σ * x)**2
            return lnpdf, dlnpdf, d2lnpdf
            
class Beta(HyperPrior):
    """
    Beta hyper-parameter priors for parameters bounded by [0,1]
        f(x; a, b) = const * x^(a - 1) * (1 - x)^(b - 1)
    """
    def __init__(self, a, b, guess=1.):
        self.guess = guess
        self._a = a
        self._b = b
        
    def __call__(self, x=None, grad=False):
        if x is None:
            x = self.guess
        x = x / 2.
        a = self._a - 1.
        b = self._b - 1.
        lnpdf = a*log(x) + b*log(1.-x) + log(.5)
        if not grad:
            return lnpdf
        dlnpdf = a / x + b / (x - 1)
        if grad is True:
            return lnpdf, dlnpdf
        if grad == 'Hess':
            d2lnpdf = -a / x**2 - b / (x - 1)**2
            return lnpdf, dlnpdf, d2lnpdf
            
class Gamma(HyperPrior):
    """
    Gamma hyper-parameter priors
        f(x ; k , θ) = x^(k - 1) * exp(-x / θ) / (θ^k * Γ(k))
    """
    def __init__(self, mean, std, guess=1):
        self.guess = guess
        self._k = (mean / std)**2
        self._θ = std**2 / mean
        self._denomenator = self._k * log(self._θ) +  log(gamma(self._k))
        
    def __call__(self, x=None, grad=False):
        if x is None:
            x = self.guess
        k = self._k - 1.
        lnpdf = k * log(x) - x / self._θ - self._denomenator
        if not grad:
            return lnpdf
        dlnpdf = k / x - 1 / self._θ
        if grad is True:
            return lnpdf, dlnpdf
        if grad == 'Hess':
            d2lnpdf = -k / x**2
            return lnpdf, dlnpdf, d2lnpdf
            
class Bounded(HyperPrior):
    """
    Bounded constant hyper-parameter prior
       f = const if inside of bounds, else zero
       Meant for use when optimizing likelihood is desired
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
        if grad is True:
            return array([val]),array([0.0])
        if grad == 'Hess': 
            return array([val]),array([0.0]),array([0.0])
