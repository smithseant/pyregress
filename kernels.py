# -*- coding: utf-8 -*-
"""
Docstring for the kernels module - needs to be written
"""
# Created Sep 2013
# @author: Sean T. Smith

from abc import ABCMeta, abstractmethod
from numbers import Number
from numpy import array, empty, zeros, ones, eye, sum, prod, ix_, expand_dims, concatenate, tile
from scipy import exp, log

# TODO: Add periodic, but it would require general handling of multiple Rs.

class Kernel:
    """
    Provide methods & an interface for kernels in the GPR class.
    
    User-defined kernels will need to inherit this baseclass and define
    both __init__ and __call__ methods in the derived class. This base
    class provides the abstract interface for the __call__ method and
    provides the methods: __init__, declare_hyper, and map_hyper.
    """
    __metaclass__ = ABCMeta
    
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
        (self.p, self.Np) = ({}, 0)
        (self.hp, self.Nhp, self.hp_iterable) = ({}, 0, [])
        for (key, val) in params.iteritems():
            if isinstance(val, Number):
                self.p[key] = val
                self.Np += 1
            elif isinstance(val, HyperPrior):
				if val == True:
					val = Constant()  # can we change the iterator mid-stream?
                self.p[key] = val.guess
                self.Np += 1
                self.hp[key] = val
                self.hp_iterable += [key]
            elif isinstance(val, list):
                self.p[key] = [None]*len(val)
                self.Np += len(val)
                self.hp[key] = [None]*len(val)
                for i in range(len(val)):
                    if isinstance(val[i], Number):
                        self.p[key][i] = val[i]
                    elif isinstance(val[i], HyperPrior):
						if val == True:
							val = Constant()  # can we change the iterator mid-stream?
                        self.p[key][i] = val[i].guess
                        self.hp[key][i] = val[i]
                        self.hp_iterable += [i]
            self.Nhp = len(self.hp_iterable)
    
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
    
    def map_hyper(self, p_mapped, unmap=False):
        """
        Replace hyper-parameter values with pointers to a 1D array.
        
        Arguments
        ---------
        p_mapped: array-1D,
            array to which the hyper-parameters will point.
        unmap: bool (optional),
            if true, hyper-parameter pointers will be replaced by values.
        """
        # TODO: throw an error if the number of hyper-parameters doesn't match.
        for (hp, i) in zip(self.hp_iterable, range(self.Nhp)):
            if not isinstance(hp, int):
                p_mapped[i] = self.p[hp]
                self.p[hp] = p_mapped[i:i+1]
            else:
                p_mapped[i] = self.p['l'][hp]
                self.p['l'][hp] = p_mapped[i:i+1]
        return (self, p_mapped)

    def _ln_priors(self, params, grad=False):
        """
        Calculate log of prior distributions for hyper-parameters.
        
        Arguments
        ---------
        params: array-1D
            array of hyper-parameter values.
            
        Returns
        -------
        logPrior: scalar value
            summation of values of log prior probabilities evaluated at 
            values provided by params
            
        PriorGrad: array-1D
            array of gradients of log prior probabilities evaluated at 
            values provided by params
        """
        
        logPrior = 0.0
        if (grad == True):
            PriorGrad = array([])
            for f in self.Prior:
                (prior,d_prior) = f(params,True)
                for p in prior:
                    logPrior += p
                if isinstance(d_prior,list):
                    for i in len(d_prior):
                        d_prior[i] = d_prior[i]
                else:
                    d_prior = d_prior
                PriorGrad = concatenate((PriorGrad,d_prior),axis=1)
            return logPrior, PriorGrad
        else:
            for f in self.Prior:
                logPrior += log(abs(f(params)))
            return logPrior

    @abstractmethod
    def __call__(self, Rk, grad=False, **options):
        """
        Calculate and return kernel values given the radius array.
        
        Arguments
        ---------
        Rk: array-3D,
            directional radius matrix (difference between points).
        grad: bool (optional),
            when grad is True also return Kgrad, and when grad is 'Hess'
            also return Khess.
        options: any additional options (opt_name=opt_value),
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


class Noise(Kernel):
    r"""
    White noise kernel object.
    ..math::
        K(R, data; w) = w^2 * I, or a zero matrix based on the data,
    with the weight parameter, w, and a flag indicating inclusion or not.
    White noise is discontinuous.
    """
    def __init__(self, params):
        super(Noise, self).__init__(1, params)
    def __call__(self, Rk, grad=False, **options):
        w = self.p['w']
        w2 = w**2
        if options.has_key['data'] and options['data']==True:
            K0 = eye(Rk.shape[0], Rk.shape[1])
        else:
            K0 = zeros(Rk.shape[:2])
        if not grad:
            # K = w2*K0
            return w2*K0
        Kgrad = empty((Rk.shape[0], Rk.shape[1], self.Nhp))
        if self.hp['w']:
            # dK/dw:
            Kgrad[:,:,0] = 2.0*w*K0
        if grad != 'Hess':
            return (w2*K0, Kgrad)
        Khess = empty((Rk.shape[0], Rk.shape[1], self.Np, self.Np))
        if self.hp['w']:
            # d^2K/dw^2:
            Khess[:,:,0,0] = 2.0*K0
        return (w2*K0, Kgrad, Khess)

class SquareExp(Kernel):
    r"""
    Squared-exponential kernel object.
    .. math::
        K(R; w, l) = w^2*\exp( -1/2 *(R/l)^2 ),
    whith the parameters of weight, w, and length, l. For multiple
    dimensions, the length can be a single value applied to all directions
    or it can be a list with a separate value in each direction.
    Squared-exponential is continuous and infinitely differentiable.
    """
    def __init__(self, params):
        p_bounds = {'w':(0.0, None), 'l':(0.0, None)}
        super(SquareExp, self).__init__(2, params, p_bounds)
    def __call__(self, Rk, grad=False, **options):
        (w, l) = (self.p['w'], self.p['l'])
        if not isinstance(l, list):
            R2l2 = (sum(Rk,2)/w)**2
        else:
            R2l2 = zeros(Rk.shape[:2])
            for k in xrange(Rk.shape[2]):
                R2l2 += (Rk[:,:,k]/l[k])**2
        w2 = w**2
        K0 = exp(-0.5*R2l2)
        if not grad:
            # K = w2*K0
            return w2*K0
        # First derivatives:
        Kgrad = empty((Rk.shape[0], Rk.shape[1], self.Nhp))
        for (i, h) in zip(range(self.Nhp), self.hp_iterable):
            if h == 'w':
                # dK/dw:
                Kgrad[:,:,i] = 2.0*w*K0
            elif h == 'l':
                # dK/dl:
                Kgrad[:,:,i] = w2*R2l2/l*K0
            elif isinstance(h, int):
                # dK/dl_h
                Kgrad[:,:,i] = w2*Rk[:,:,h]**2/l[h]**3*K0
        if grad != 'Hess':
            return (w2*K0, Kgrad)
        # Second derivatives:
        Khess = empty((Rk.shape[0], Rk.shape[1], self.Nhp, self.Nhp))
        for (i, h1) in zip(xrange(self.Nhp), self.hp_iterable):
            for (j, h2) in zip(xrange(i, self.Nhp), self.hp_iterable[i:]):
                if h1 == 'w' and h2 == 'w':
                    # d^2K/dw^2:
                    Khess[:,:,i,j] = 2.0*K0
                elif h1 == 'w' and h2 == 'l':
                    # d^2K/dwdl:
                    Khess[:,:,i,j] = 2.0*w*R2l2/l*K0
                    Khess[:,:,j,i] = Khess[:,:,i,j]
                elif h1 == 'w' and isinstance(h2, int):
                    # d^2K/dwdl_i:
                    Khess[:,:,i,j] =  (2.0*w *
                                       Rk[:,:,h1]**2/l[h1]**3*K0)
                    Khess[:,:,j,i] = Khess[:,:,i,j]
                elif h1 == 'l' and h2 == 'l':
                    # d^2K/dl^2:
                    Khess[:,:,i,j] = w2*R2l2/l**2*(R2l2-3.0)*K0
                elif isinstance(h1, int) and isinstance(h2, int) and h1 == h2:
                    # d^2K/dl_i^2:
                    Khess[:,:,i,j] = ( w2*Rk[:,:,h1]**2/l[h1]**4 *
                                      ((Rk[:,:,h1]/l[h1])**2 - 3.0)*K0 )
                elif isinstance(h1, int) and isinstance(h2, int):
                    # d^2K/dl_i dl_j:
                    Khess[:,:,i,j] = (w2 * (Rk[:,:,h1]*Rk[:,:,h2])**2/
                                      (l[h1]*l[h2])**3 * K0)
                    Khess[:,:,j,i] = Khess[:,:,i,j]
        return (w2*K0, Kgrad, Khess)

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
    def __init__(self, params):
        p_bounds = {'w':(0.0, None), 'l':(0.0, None), 'gamma':(0.0, 2.0)}
        super(GammaExp, self).__init__(3, params, p_bounds)
    def __call__(self, Rk, grad=False, **options):
        (w, l, g) = (self.p['w'], self.p['l'], self.p['gamma'])
        if not isinstance(l, list):
            Rglg = (sum(Rk,2)/l)**g
        else:
            Rglg = zeros(Rk.shape[:2])
            for k in xrange(Rk.shape[2]):
                Rglg += (Rk[:,:,k]/l[k])**g
        w2 = w**2
        K0 = exp(-Rglg)
        if not grad:
            # K = w2*K0
            return w2*K0
        # First derivatives:
        Kgrad = empty((Rk.shape[0], Rk.shape[1], self.Nhp))
        for (i, h) in zip(range(self.Nhp), self.hp_iterable):
            if h == 'w':
                # dK/dw:
                Kgrad[:,:,i] = 2.0*w*K0
            elif h == 'l':
                # dK/dl:
                Kgrad[:,:,i] = g*w2*Rglg/l*K0
            elif isinstance(h, int):
                # dK/dl_h:
                Kgrad[:,:,i] = g*w2*(Rk[:,:,h]/l[h])**g/l[h]*K0
            elif h == 'gamma':
                # dK/dgamma:
                gamma_tmp = zeros(Rk.shape[:2])
                for k in xrange(Rk.shape[2]):
                    gamma_tmp += (Rk[:,:,k]/l[k])**g * log(Rk[:,:,k]/l[k])
                Kgrad[:,:,i] = -w2*gamma_tmp*K0
        if grad != 'Hess':
            return (w2*K0, Kgrad)
        # Second derivatives:
        Khess = empty((Rk.shape[0], Rk.shape[1], self.Nhp, self.Nhp))
        for (i, h1) in zip(xrange(self.Nhp), self.hp_iterable):
            for (j, h2) in zip(xrange(i, self.Nhp), self.hp_iterable[i:]):
                if h1 == 'w' and h2 == 'w':
                    # d^2K/dw^2:
                    Khess[:,:,i,j] = 2.0*K0
                elif h1 == 'w' and h2 == 'l':
                    # d^2K/dwdl:
                    Khess[:,:,i,j] = 2.0*g*w*Rglg/l*K0
                    Khess[:,:,j,i] = Khess[:,:,i,j]
                elif h1 == 'w' and isinstance(h2, int):
                    # d^2K/dwdl_h:
                    Khess[:,:,i,j] = 2.0*g*w*(Rk[:,:,h2]/l[h2])**g/l[h2]*K0
                    Khess[:,:,j,i] = Khess[:,:,i,j]
                elif h1 == 'w' and h2 == 'gamma':
                    # d^2K/dwdgamma:
                    Khess[:,:,i,j] = -2.0*w*gamma_tmp*K0
                    Khess[:,:,j,i] = Khess[:,:,i,j]
                elif h1 == 'l' and h2 == 'l':
                    # d^2K/dl^2:
                    Khess[:,:,i,j] = g*w2*Rglg/l**2*(g*Rglg-(g+1.0))*K0
                elif h1 == 'l' and h2 == 'gamma':
                    # d^2K/dldgamma:
                    Khess[:,:,i,j] = w2/l*(Rglg + g*(1.0-Rglg)*gamma_tmp)*K0
                    Khess[:,:,j,i] = Khess[:,:,i,j]
                elif isinstance(h1, int) and isinstance(h2, int) and h1 == h2:
                    # d^2K/dl_h^2:
                    Khess[:,:,i,j] = ( g*w2*Rk[:,:h1]**g/l**(g+2.0)*
                                      (g*(Rk[:,:,h1]/l)**g-(g+1.0))*K0 )
                elif isinstance(h1, int) and isinstance(h2, int):
                    # d^2K/dl_h1 dl_h2:
                    Khess[:,:,i,j] = ( g**2*w2*(Rk[:,:,h1]*Rk[:,:,h2])**g/
                                      (l[h1]*l[h2])**(g+1.0)*K0 )
                    Khess[:,:,j,i] = Khess[:,:,i,j]
                elif isinstance(h1, int) and h2 == 'gamma':
                    # d^2K/dl_h dgamma:
                    Khess[:,:,i,j] = ( w2*Rk[:,:,h1]**g/l**(g+1.0)*
                                      (1.0+g*log(Rk[:,:,h1]/l)+g*gamma_tmp)*K0 )
                    Khess[:,:,j,i] = Khess[:,:,i,j]
                elif h1 == 'gamma' and h2 == 'gamma':
                    # d^2K/dgamma^2:
                    Khess[:,:,i,j] = w2*gamma_tmp*(gamma_tmp-1.0)*K0
        return (w2*K0, Kgrad, Khess)

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
    def __init__(self, params):
        p_bounds = {'w':(0.0, None), 'l':(0.0, None), 'alpha':(0.0, None)}
        super(RatQuad, self).__init__(3, params, p_bounds)
    def __call__(self, Rk, grad=False, **options):
        (w, l, a) = (self.p['w'], self.p['l'], self.p['alpha'])
        if not isinstance(l, list):
            R2l2 = (sum(Rk,2)/l)**2
        else:
            R2l2 = zeros(Rk.shape[:2])
            for k in xrange(Rk.shape[2]):
                R2l2 += (Rk[:,:,k]/l[k])**2
        w2 = w**2
        all_tmp = 1.0 + R2l2/(2.0*a)
        K0 = all_tmp**(-a)
        if not grad:
            return w2*K0
        # First derivatives:
        Kgrad = empty((Rk.shape[0], Rk.shape[1], self.Nhp))
        for (i, h) in zip(range(self.Nhp), self.hp_iterable):
            if h == 'w':
                # dK/dw:
                Kgrad[:,:,i] = 2.0*w*K0
            elif h == 'l':
                # dK/dl:
                Kgrad[:,:,i] = w2*R2l2/(l*all_tmp)*K0
            elif isinstance(h, int):
                # dK/dl_h:
                Kgrad[:,:,i] = w2*Rk[:,:,h]**2/(l[h]**3*all_tmp)*K0
            elif h == 'alpha':
                # dK/dalpha:
                alpha_tmp = (all_tmp-1)/all_tmp - log(all_tmp)
                Kgrad[:,:,i] = w2*alpha_tmp*K0  
        if grad != 'Hess':
            return (w2*K0, Kgrad)
        # Second derivatives:
        Khess = empty((Rk.shape[0], Rk.shape[1], self.Nhp, self.Nhp))
        for (i, h1) in zip(xrange(self.Nhp), self.hp_iterable):
            for (j, h2) in zip(xrange(i, self.Nhp), self.hp_iterable[i:]):
                if h1 == 'w' and h2 == 'w':
                    # d^2K/dw^2:
                    Khess[:,:,i,j] = 2.0*K0
                elif h1 == 'w' and h2 == 'l':
                    # d^2K/dwdl:
                    Khess[:,:,i,j] = 2.0*w*R2l2/(l*all_tmp)*K0
                    Khess[:,:,j,i] = Khess[:,:,i,j]
                elif h1 == 'w' and isinstance(h2, int):
                    # d^2K/dwdl_h:
                    Khess[:,:,i,j] = 2.0*w*Rk[:,:,h]**2/(l[h]**3*all_tmp)*K0
                    Khess[:,:,j,i] = Khess[:,:,i,j]
                elif h1 == 'w' and h2 == 'alpha':
                    # d^2K/dwdalpha:
                    Khess[:,:,i,j] = 2.0*w*alpha_tmp*K0 
                    Khess[:,:,j,i] = Khess[:,:,i,j]
                elif h1 == 'l' and h2 == 'l':
                    # d^2K/dl^2:
                    Khess[:,:,i,j] = w2*R2l2/l**2*((a+1.0)/a*R2l2/all_tmp - 3.0)/all_tmp*K0
                elif h1 == 'l' and h2 == 'alpha':
                    # d^2K/dldalpha:
                    Khess[:,:,i,j] = w2*R2l2/l*((a+1.0)/a*(all_tmp-1.0)/all_tmp - log(all_tmp))/all_tmp*K0
                    Khess[:,:,j,i] = Khess[:,:,i,j]
                elif isinstance(h1, int) and isinstance(h2, int) and h1 == h2:
                    # d^2K/dl_h^2:
                    Khess[:,:,i,j] = w2*Rk[:,:,h1]**2/l[h1]**4*((a+1.0)/a*(Rk[:,:,h1]/l[h1])**2/all_tmp - 3.0)/all_tmp*K0
                elif isinstance(h1, int) and isinstance(h2, int):
                    # d^2K/dl_h1 dl_h2:
                    Khess[:,:,i,j] = (a+1.0)/a *w2*(Rk[:,:,h1]*Rk[:,:,h2]/all_tmp)**2/(l[h1]*l[h2])**3*K0
                    Khess[:,:,j,i] = Khess[:,:,i,j]
                elif isinstance(h1, int) and h2 == 'alpha':
                    # d^2K/dl_h dalpha:
                    Khess[:,:,i,j] = ( w2*Rk[:,:,h1]**2/l[h1]**3*((a+1.0)/a*(all_tmp-1.0)/all_tmp - log(all_tmp) ) /all_tmp*K0 )
                    Khess[:,:,j,i] = Khess[:,:,i,j]
                elif h1 == 'alpha' and h2 == 'alpha':
                    # d^2K/dalpha^2:
                    Khess[:,:,i,j] = ( w2*(R2l2**2/(4.0*a**3*all_tmp**2)+
                                           alpha_tmp**2)*K0 )
        return (w2*K0, Kgrad, Khess)