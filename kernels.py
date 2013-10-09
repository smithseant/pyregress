# -*- coding: utf-8 -*-
"""
Docstring for the kernels module - needs to be written
"""
# Created Sep 2013
# @author: Sean T. Smith

from abc import ABCMeta, abstractmethod
from numpy import array, diag
from scipy import exp, log

class BaseKernel:
    """
    Provide methods & an interface for kernels in the gpr class.
    
    For advanced use, user-defined kernels will need to inherit this base class
    and define both __init__ and __call__ methods in the derived class.
    This base class provides the abstract interface for the __call__ method
    and provides the methods: __init__ and declare_hyper.
    """
    __metaclass__ = ABCMeta
    
    def __init__(self, Nparams, params):
        """Create a BaseKernel object - inheritted classes should call this
        using the super function (as in the examples of commonly used kernels).
        
        Arguments
        ---------
        Nparams: int,
            number of kernel parameters for this specific kernel.
        params: list of floats or arrays,
            values for the kernel parameters.
        """
        
        self.Np = Nparams
        self.p = params
        self.Nhyper = 0
        self.hyper = [False]*self.Np
        # -- add a test that throws an error if the number of parameters doesn't match --
    
    def declare_hyper(self, hyper_params):
        """
        Declare which kernel parameters will be treated as hyper parameters.
        
        Arguments
        ---------
        hyper_params: bool or list of bools,
            which of this kernel's parameters are labled hyper parameters -
            bool, label all or none; list, manually label each individually.
        
        Returns
        -------
        Nhyper: int,
            resulting number of hyper parameters.
        """
        
        if not hyper_params or hyper_params=='none':
            self.Nhyper = 0
            self.hyper = [False]*self.Np
        elif hyper_params==True or hyper_params=='all':
            self.Nhyper = self.Np  # -- generalize so length can be an array --
            self.hyper = [True]*self.Np
        else:
            self.Nhyper = hyper_params.count(True)  # -- generalize so length can be an array --
            self.hyper = hyper_params
        # -- add a test that throws an error if the number of parameters doesn't match --
        return self.Nhyper
    
    @abstractmethod
    def __call__(self, R, grad=False):
        """
        Calculate and return kernel values given the radius array.
        
        Arguments
        ---------
        self: BaseKernel,
            kernel parameters (self.p) and hyper-parameter labels (self.hyper).
        R: array-2D,
            distance matrix (radius).
        grad: bool (optional),
            when grad is not False, must return gradK.
        
        Returns
        -------
        K: array-2D,
            kernel values - shape must match argument R.
        gradK: list of arrays (optional - depending on argument grad),
            partial derivative of kernel with respect to each hyper parameter.
        """
        return
    
class Noise(BaseKernel):
    r"""
    White noise kernel object.
    ..math::
        K_{i,j}(R) = w^2*\delta_{i,j}*\delta_D(R_{i,j}),
    where the parameter list includes only the weight, p = [w].
    White noise is discontinuous.
    """
    def __init__(self, params):
        super(Noise, self).__init__(1, params)
    def __call__(self, R, grad=False):
        K = diag( array([1.0*b for b in R.diagonal() == 0.0]) )
        if not grad:
            return self.p[0]**2*K
        else:
            if not self.hyper[0]:
                Kprime = []
            else:
                Kprime = [ 2.0*self.p[0]*K ]
            return (self.p[0]**2*K, Kprime)

class OU(BaseKernel):
    r"""
    Ornstein-Uhlenbeck kernel object.
    .. math::
        K(R) = w^2*\exp( -R/l ),
    where the parameter list is p = [w, l].
    Ornstein-Uhlenbeck is continuous, but not smooth.
    """
    def __init__(self, params):
        super(OU, self).__init__(2, params)
    def __call__(self, R, grad=False):
        K = exp(-R/self.p[1])
        if not grad:
            return self.p[0]**2*K
        else:  # -- generalize so p[1] can be an array --
            Kprime = []
            if self.hyper[0]:
                Kprime += [ 2.0*self.p[0]*K ]
            if self.hyper[1]:
                Kprime += [ self.p[0]**2*R/self.p[1]*K ] 
            return (self.p[0]**2*K, Kprime)

class GammaExp(BaseKernel):
    r"""
    Gamma-exponential kernel object.
    .. math::
        K(R) = w^2*\exp( -(R/l)^{\gamma} ),
    where the parameter list is p = [w, l, gamma].
    Gamma-exponential is continuous, and when gamma=2 it is smooth.
    """
    def __init__(self, params):
        """Docstring under __init__"""
        super(GammaExp, self).__init__(3, params)
    def __call__(self, R, grad=False):
        K = exp(-(R/self.p[1])**self.p[2])
        if not grad:
            return self.p[0]**2*K
        else:  # -- generalize so p[1] can be an array --
            Kprime = []
            if self.hyper[0]:
                Kprime += [ 2.0*self.p[0]*K ]
            if self.hyper[1]:
                tmp = self.p[0]**2*(R/self.p[1])**self.p[2]
                Kprime += [ self.p[2]*tmp/self.p[1]*K ]
            if self.hyper[2]:
                tmp = self.p[0]**2*(R/self.p[1])**self.p[2]
                Kprime += [ -tmp*log(R/self.p[1])*K ]
            return (self.p[0]**2*K, Kprime)

class SquareExp(BaseKernel):
    r"""
    Squared-exponential kernel object.
    .. math::
        K(R) = w^2*\exp( -1/2 *(R/l)^2 ),
    where the parameter list is p = [w, l].
    Squared-exponential is continuous and infinitely differentiable.
    """
    def __init__(self, params):
        super(SquareExp, self).__init__(2, params)
    def __call__(self, R, grad=False):
        K = exp(-0.5*(R/self.p[1])**2)
        if not grad:
            return self.p[0]**2*K
        else:  # -- generalize so p[1] can be an array --
            Kprime = []
            if self.hyper[0]:
                Kprime += [ 2.0*self.p[0]*K ]
            if self.hyper[1]:
                Kprime += [ self.p[0]**2*(R/self.p[1])**2/self.p[1]*K ]
            return (self.p[0]**2*K, Kprime)

class RatQuad(BaseKernel):
    r"""
    Rational-quadratic kernel object.
    .. math::
        K(R) = w^2*( 1 + \frac{R^2}{2*alpha*l^2} )^2,
    where the parameter list is p = [w, l, alpha].
    Rational quadratic is SE over a gamma distribution of length scales.
    """
    def __init__(self, params):
        super(RatQuad, self).__init__(3, params)
    def __call__(self, R, grad=False):
        K = (1.0 + (R/self.p[1])**2/(2.0*self.p[2]))**(-self.p[2])
        if not grad:
            return self.p[0]**2*K
        else:  # -- generalize so p[1] can be an array --
            Kprime = []
            if self.hyper[0]:
                Kprime += [ 2.0*self.p[0]*K ]
            if self.hyper[1]:
                Kprime += [ self.p[0]**2*(R/self.p[1])**2/self.p[1]*K ]
            return (self.p[0]**2*K, Kprime)

# -- would like to add periodic, but that would require more general handling of R --