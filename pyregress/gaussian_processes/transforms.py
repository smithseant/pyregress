# -*- coding: utf-8 -*-
"""
Docstring for the transforms module - needs to be written
Created Sep 2013
@author: Sean T. Smith
"""
from abc import ABCMeta, abstractmethod
from numpy import sqrt, mean, var, empty, tile, exp, log, pi as π

from scipy.special import erf, erfinv, beta, betainc, betaincinv

class BaseTransform(metaclass=ABCMeta):
    """
    Provide methods & interfaces for variable transformations in the gpr class.

    For advanced use, user-defined variable transformations will need to
    inherit this base class and define both the transform and inv_transform
    methods in the derived class.
    This base class provides the abstract interface for the transform and
    inv_transform methods, and it provides the methods: __init__ and __call__.
    """
    def __init__(self, *args, **kwargs):
        """Create a transformation object."""
        pass
    def __call__(self, x, inverse=False, grad_z=None, hess_z=None):
        """Perform the transformation or inverse transformation."""
        if not inverse:
            return self.transform(x)
        else:
            return self.inv_transform(x, grad_z=grad_z, hess_z=hess_z)

    @abstractmethod
    def transform(self, y):
        """
        Perform a variable transformation to a normally distributed variable.

        Argument
        --------
        y:  array,
            dependent variable values to be transformed.

        Returns
        -------
        z:  array (same shape as y),
            transformed dependent variable values.
        """
        return
    @abstractmethod
    def inv_transform(self, z, grad_z=None, hess_z=None):
        """
        Perform the inverse variable transformation consistent with transform.

        Argument
        --------
        z:  array,
            already transformed dependent variable values.
        grad_z:  array-2D (optional),
            derivative of the transformed variable.
        hess_z:  array-3D (optional),
            second derivative of the transformed variable.

        Returns
        -------
        y:  array (same shape as z),
            dependent variable values in the original function space.
        grad_y:  array-2D (optional, depending on the argument grad_z),
            derivatives of untransformed variable.
        hess_y:  array-3D (optional, depending on the argument hess_z),
            second derivatives of the untransformed variable.
        """
        return

class Logarithm(BaseTransform):
    r"""
    Transform a variable on a semi-infinite support [0,\infty) to an infinite
    support (-\infty,\infty) using the logarithm transformation.
    .. math::
        z = \log(y),
        y = \exp(z).
    """
    def transform(self, y):
        return log(y)
    def inv_transform(self, z, grad_z=None, hess_z=None):
        y = exp(z)
        if grad_z is None:
            return y
        grad_y = y.reshape((-1, 1)) * grad_z
        if hess_z is None:
            return y, grad_y
        hess_y = empty(hess_z.shape)
        for i in range(hess_z.shape[1]):
            for j in range(hess_z.shape[2]):
                hess_y[:, i, j] = y * (grad_z[:, i] * grad_z[:, j] + hess_z[:, i, j])
        return y, grad_y, hess_y

class Logit(BaseTransform):
    r"""
    Transform a variable on a finite support [0,1] to an infinite support
    (-\infty,\infty) using the logit transformation.
    .. math::
        z = \ln(\frac{y}{1-y}),
        y = \frac{1}{\e^{-z}+1}.
    """
    def transform(self, y):
        return log(y/(1.0 - y))
    def inv_transform(self, z, grad_z=None, hess_z=None):
        y = 1 / (exp(-z) + 1)
        if grad_z is None:
            return y
        fz = exp(z) / (1 + exp(z))**2
        grad_y = fz.reshape((-1, 1)) * grad_z
        if hess_z is None:
            return y, grad_y
        hess_y = empty(hess_z.shape)
        for i in range(hess_z.shape[1]):
            for j in range(hess_z.shape[2]):
                hess_y[:, i, j] = fz * ((exp(z) - 1) / (exp(z) + 1) * grad_z[:, i] * grad_z[:, j] + hess_z[:, i, j])
        return y, grad_y, hess_y

class Probit(BaseTransform):
    r"""
    Transform a variable on a finite support [0,1] to an infinite support
    (-\infty, \infty) using the probit transformation.
    .. math::
        z = \sqrt{2}\erf^{-1}(2y-1),
        y = \frac{1}{2}[\erf(\frac{z}{\sqrt{2}})+1].
    """
    def transform(self, y):
        return sqrt(2) * erfinv(2 * y - 1)
    def inv_transform(self, z, grad_z=None, hess_z=None):
        y = 0.5 * (erf(z * sqrt(0.5)) + 1.0)
        if grad_z is None:
            return y
        fz = exp(-0.5 * z**2) / sqrt(2 * π)
        grad_y = fz.reshape((-1, 1)) * grad_z
        if hess_z is None:
            return y, grad_y
        hess_y = empty(hess_z.shape)
        for i in range(hess_z.shape[1]):
            for j in range(hess_z.shape[2]):
                hess_y[:, i, j] = fz * (-z * grad_z[:, i] * grad_z[:, j] + hess_z[:, i, j])
        return y, grad_y, hess_y

class ProbitBeta(BaseTransform):
    r"""
    Transform a variable on a finite support [0,1] to an infinite support
    (-\infty,\infty) using a modified probit transformation.
    .. math::
        z = \sqrt{2 \sigma^2} \erf^{-1}(2 I(y,\alpha,\beta)-1) + \mu,
        y = I^{-1}(\frac{1}{2}[\erf(\frac{z-\mu}{\sqrt{2 \sigma^2}})+1], a, b).
    """
    def __init__(self, yd):
        # This transformation requires a bit of extra work to initialize.
        y_mean = mean(yd)
        y_var = var(yd)
        tmp = (y_mean * (1 - y_mean)) / y_var - 1
        self.alpha = y_mean * tmp
        self.beta = (1 - y_mean) * tmp
        z_prime = betainc(self.alpha, self.beta, yd)
        z_prime = sqrt(2) * erfinv(2 * z_prime - 1)
        self.mu = mean(z_prime)
        self.sigma2 = var(z_prime)
    def transform(self, y):
        z_prime = betainc(self.alpha, self.beta, y)
        return sqrt(2 * self.sigma2) * erfinv(2 * z_prime - 1) + self.mu
    def inv_transform(self, z, grad_z=None, hess_z=None):
        mu, s2, a, b = self.mu, self.sigma2, self.alpha, self.beta
        yp = 0.5 * (erf((z - mu) / sqrt(2 * s2)) + 1)
        y = betaincinv(a, b, yp)
        if grad_z is None:
            return y
        fz = exp( -(z - mu)**2 / (2 * s2) ) / sqrt(2 * π * s2)
        fy = y**(a - 1) * (1 - y)**(b - 1) / beta(a, b)
        fzfy = tile(fz / fy, (1, grad_z.shape[1]))
        grad_y = fzfy * grad_z
        if hess_z is None:
            return y, grad_y
        hess_y = empty(hess_z.shape)
        for i in range(hess_z.shape[1]):
            for j in range(hess_z.shape[2]):
                hess_y[:, i, j] = fz / fy * ( ((mu - z) / s2 - fz / fy*((a - 1) / y - (b - 1) / (1 - y))) * grad_z[:, i] * grad_z[:, j] + hess_z[:, i, j] )
        return y, grad_y, hess_y
