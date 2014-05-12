# -*- coding: utf-8 -*-
"""
Docstring for the transforms module - needs to be written
"""
# Created Sep 2013
# @author: Sean T. Smith

from abc import ABCMeta, abstractmethod

from numpy import sqrt, mean, var
from scipy.special import erf, erfinv, betainc, betaincinv
from scipy import exp, log

class BaseTransform():
    """
    Provide methods & interfaces for variable transformations in the gpr class.
    
    For advanced use, user-defined variable transformations will need to
    inherit this base class and define both the transfor and inv_transform
    methods in the derived class.
    This base class provides the abstract interface for the transfor and
    inv_transform methods, and it provides the methods: __init__ and __call__.
    """
    __metaclass__ = ABCMeta
    def __init__(self, yd):
        """Create a transformation object using dependent variable data."""
        return None
    def __call__(self, x, inverse=False):
        """Perform the transformation or inverse transformation."""
        if not inverse:
            return self.transform(x)
        else:
            return self.inv_transform(x)
    
    @abstractmethod
    def transform(self, y):
        """
        Perform a variable transformation to a normally distributed variable.
        
        Argument
        --------
        y: array,
            dependent variable values to be transformed.
        
        Returns
        -------
        z: array (same shape as y),
            transformed dependent variable values.
        """
        return
    @abstractmethod
    def inv_transform(self, z):
        """
        Perform the inverse variable transformation consistent with transform.
        
        Argument
        --------
        z: array,
            already transformed dependent variable values.
        
        Returns
        -------
        y: array (same shape as z),
            dependent variable values in the original function space.
        """
        return

class Logarithm(BaseTransform):
    r"""
    Transform a variable on a semi-infinite suppot [0,\infty) to an infinite
    support (-\infty,\infty) using the logarithm transformation.
    .. math::
        z = \log(y),
        y = \exp(z).
    """
    def transform(self, y):
        return log(y)
    def inv_transform(self, z):
        return exp(z)

class Probit(BaseTransform):
    """
    Transform a variable on a finite suppot [0,1] to an infinite support
    (-\infty,\infty) using the probit transformation.
    .. math::
        z = \sqrt{2}\erf^{-1}(2y-1),
        y = \frac{1}{2}[\erf(\frac{z}{\sqrt{2}})+1].
    """
    def transform(self, y):
        return sqrt(2.0)*erfinv(2.0*y - 1.0)
    def inv_transform(self, z):
        return 0.5*(erf(z*sqrt(0.5)) + 1.0)

class ProbitBeta(BaseTransform):
    r"""
    Transform a variable on a finite suppot [0,1] to an infinite support
    (-\infty,\infty) using a modified probit transformation.
    .. math::
        z = \sqrt{2 \sigma^2} \erf^{-1}(2 I(y,\alpha,\beta)-1) + \mu,
        y = I^{-1}(\frac{1}{2}[\erf(\frac{z-\mu}{\sqrt{2 \sigma^2}})+1], a, b).
    """
    def __init__(self, yd):
        # This transformatiom requires a bit of extra work to initialize.
        y_mean = mean(yd)
        y_var = var(yd)
        tmp = ( (y_mean*(1.0 - y_mean))/y_var - 1.0 )
        self.alpha = y_mean*tmp
        self.beta = (1.0 - y_mean)*tmp
        z_prime = betainc(self.alpha, self.beta, yd)
        z_prime = sqrt(2.0)*erfinv(2.0*z_prime - 1.0)
        self.mu = mean(z_prime)
        self.sigma2 = var(z_prime)
    def transform(self, y):
        z_prime = betainc(self.alpha, self.beta, y)
        return sqrt(2.0*self.sigma2)*erfinv(2.0*z_prime - 1.0) + self.mu
    def inv_transform(self, z):
        y_prime = 0.5*(erf((z-self.mu)/sqrt(2.0*self.sigma2)) + 1.0)
        return betaincinv(self.alpha, self.beta, y_prime)

class Logit(BaseTransform):
    r"""
    Transform a variable on a finite suppot [0,1] to an infinite support
    (-\infty,\infty) using the logit transformation.
    .. math::
        z = \ln(\frac{y}{1-y}),
        y = \frac{1}{\e^{-z}+1}.
    """
    def transform(self, y):
        return log(y/(1.0 - y))
    def inv_transform(self, z):
        return 1.0/(exp(-z)+1.0)