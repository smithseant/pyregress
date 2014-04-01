# -*- coding: utf-8 -*-
"""
Docstring for the kernels module - needs to be written
"""
# Created Sep 2013
# @author: Sean T. Smith

from abc import ABCMeta, abstractmethod
from numpy import array, empty, zeros, diag, eye, sum, prod, where
from scipy import exp, log

class Kernel:
    """
    Provide methods & an interface for kernels in the GPR class.
    
    User-defined kernels will need to inherit this baseclass and define
    both __init__ and __call__ methods in the derived class. This base
    class provides the abstract interface for the __call__ method and
    provides the methods: __init__, declare_hyper, and map_hyper.
    """
    __metaclass__ = ABCMeta
    
    def __init__(self, Nparams, params, param_bounds):
        """
        Create a Kernel object.
        
        Arguments
        ---------
        Nparams: int,
            number of kernel parameters for this specific kernel.
        params: dict,
            names and values (might be a list) of the kernel parameters.
        param_bounds: dict,
            each parameter's domain in the form of a tuple (min, max).
            For unbounded use the value of None for min and/or max.
        """
        # TODO: throw an error if the parameters doesn't match.
        self.Np = Nparams
        self.p = params
        self.Nhp = 0
        self.hp = [False]*self.Np
        for i in range(len(self.p)):
            if isinstance(self.p[i], list):
                self.hp[i] = [False]*len(self.p[i])
    
    @abstractmethod
    def __call__(self, Rk, grad=False, **options):
        """
        Calculate and return kernel values given the radius array.
        
        Arguments
        ---------
        Rk: array-3D,
            directional radius matrix (difference between points).
        grad: bool (optional),
            when grad is not False, must return gradK.
        options: any additional options (opt_name=opt_value),
            specific options for specific kernels, otherwise ignored.
        
        Returns
        -------
        K: array-2D,
            kernel values - shape must match argument R.
        grad: array-3D (optional - depending on argument grad),
            partial of kernel (first two dimensions) with respect to each
            hyper parameter (third dimension).
        Hess: array-4D (optional - depending on argument grad),
            second derivative for all combinations of two hyper parameters.
        """
        return
    
    def __add__(self, other):
        """Overload '+' so Kernel objects can be added."""
        if isinstance(other, KernelSum):
            return other.__add__(self, self_on_right=True)
        else:
            return KernelSum(self, other)
    
    def __mul__(self, other):
        """Overload '*' so Kernel objects can be multiplied."""
        if isinstance(other, KernelProd):
            return other.__mul__(self, self_on_right=True)
        else:
            return KernelProd(self, other)
    
    def declare_hyper(self, hyper_params):
        """
        Declare which kernel parameters will be treated as hyper parameters.
        
        Arguments
        ---------
        hyper_params: bool or list of bools,
            which of this kernel's parameters are hyper-parameters -
            bool, 'all', 'none', or list for manual selection.
        
        Returns
        -------
        Nhyper: int,
            resulting number of hyper parameters.
        """
        if not hyper_params or hyper_params=='none':
            self.Nhp = 0
        elif hyper_params==True or hyper_params=='all':
            self.Nhp = self.Np
            self.hp[:] = [True]*self.Np
            for i in range(len(self.p)):
                if isinstance(self.p[i], list):
                    self.Nhp += len(self.p[i]) - 1
                    self.hp[i] = [True]*len(self.p[i])
        else:
            # TODO: throw an error if the number of parameters doesn't match.
            self.Nhp = hyper_params.count(True)
            self.hp[:] = hyper_params[:]
            for i in range(self.Np):
                if isinstance(self.p[i], list):
                    if isinstance(hyper_params[i], list):
                        self.Nhp += hyper_params[i].count(True)
                    elif hyper_params[i]==False or hyper_params[i]=='none':
                        self.hp[i] = [False]*len(self.p[i])
                    elif hyper_params[i]==True:
                        self.hp[i] = [True]*len(self.p[i])
                        self.Nhp += len(self.p[i]) - 1
                    elif hyper_params[i]=='all':
                        self.hp[i] = [True]*len(self.p[i])
                        self.Nhp += len(self.p[i])
        return self.Nhp
    
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
        im = 0
        for i in range(len(self.p)):
            if im == p_mapped.size:
                break
            if self.hp[i] is True:
                if not unmap:
                    p_mapped[im] = self.p[i]
                    self.p[i] = p_mapped[im:im+1]
                else:
                    self.p[i] = p_mapped[im]
                im += 1
            elif isinstance(self.p[i], list):
                for j in range(len(self.p[i])):
                    if self.hp[i][j] is True:
                        if not unmap:
                            p_mapped[im] = self.p[i][j]
                            self.p[i][j] = p_mapped[im:im+1]
                        else:
                            self.p[i][j] = p_mapped[im]
                        im += 1
        return (self, p_mapped)


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
        w2 = self.p['w']**2
        if options.has_key['data'] and options['data']==True:
            K0 = eye(Rk.shape[0], Rk.shape[1])
        else:
            K0 = zeros(Rk.shape[:2])
        if not grad:
            # K = w2*K0
            return w2*K0
        Kprime = empty((Rk.shape[0], Rk.shape[1], self.Nhp))
        if self.hp['w']:
            # dK/dw:
            Kprime[:,:,0] = 2.0*self.p['w']*K0
        if grad != 'Hess':
            return (w2*K0, Kprime)
        Khess = empty((Rk.shape[0], Rk.shape[1], self.Np, self.Np))
        if self.hp['w']:
            # d^2K/dw^2:
            Khess[:,:,0,0] = 2.0*K0
        return (w2*K0, Kprime, Khess)

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
        if not isinstance(self.p['l'], list):
            R2l2 = (sum(Rk,2)/self.p['w'])**2
        else:
            R2l2 = zeros(Rk.shape[:2])
            for k in xrange(Rk.shape[2]):
                R2l2 += (Rk[:,:,k]/self.p['l'][k])**2
        w2 = self.p['w']**2
        K0 = exp(-0.5*R2l2)
        if not grad:
            # K = w2*K0
            return w2*K0
        
        
        
        Kprime = empty((Rk.shape[0], Rk.shape[1], self.Nhp))
        h = 0
        if self.hp['w']:
            # dK/dw:
            Kprime[:,:,h] = 2.0*self.p['w']*K0
            h += 1
        if self.hp['l'] and not isinstance(self.hp['l'], list):
            # dK/dl:
            Kprime[:,:,h] = w2*R2l2/self.p['l']*K0
            h += 1
        elif isinstance(self.hp['l'], list):
            for i in xrange(len(self.hp['l'])):
                if self.hp['l'][i]:
                    Kprime[:,:,h] = w2*Rk[:,:,i]**2/self.p['l'][i]**3*K0  # dK/dl_i
                    h += 1
        if grad != 'Hess':
            return (w2*K0, Kprime)
        Khess = empty((Rk.shape[0], Rk.shape[1], self.Np, self.Np))
        h = 0
        if self.hp['w']:
            # d^2K/dw^2:
            Khess[:,:,h,h] = 
            h += 1
        if self.hp['l'] and not isinstance(self.hp['l'], list):
            # d^2K/dl^2:
            Khess[:,:,h,h] = 
            if self.hp['w']:
                # d2K/dwdl:
                Khess[:,:,h,0] = Khess[:,:,0,h] = 
            h += 1
        elif isinstance(self.hp['l'], list):
            for i in xrange(len(self.hp['l'])):
                if self.hp['l'][i]:
                    # d^2K/dl_i^2:
                    Khess[:,:,h,h] =
                    k = 0
                    if self.hp['w']:
                        # d^2K/dwdl_i:
                        Khess[:,:,h,k] = Khess[:,:,k,h] = 
                        k += 1
                    for j in xrange(i):
                        if self.hp['l'][j]:
                            # d^2K/dl_idl_j:
                            Khess[:,:,h,k] = Khess[:,:,k,h] = 
                            k += 1
                    h += 1
        return (w2*K0, Kprime, Khess)
        
        
        Kprime = empty((Rk.shape[0], Rk.shape[1], self.Nhp))

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
    def __call__(self, Rk2, grad=False, **options):
        if not isinstance(self.p['l'], list):
            R2l2 = sum(Rk2,2)/(self.p['l']**2)
        else:
            R2l2 = zeros(Rk2.shape[:2])
            for k in xrange(Rk2.shape[2]):
                R2l2 += Rk2[:,:,k]/(self.p['l'][k]**2)
        w2 = self.p['w']**2
        K0 = exp(-R2l2**(0.5*self.p['gamma']))
        if not grad:
            return w2*K0
        else:
            Kprime = empty((Rk2.shape[0], Rk2.shape[1], self.Nhp))
            h = 0
            if self.hp[0]:
                Kprime[:,:,h] = 2.0*self.p['w']*K0
                h += 1
            if self.hp[1] and not isinstance(self.hp[1], list):
                tmp = w2*R2l2**(0.5*self.p['gamma'])
                Kprime[:,:,h] = self.p['gamma']*tmp/self.p['l']*K0
                h += 1
            elif isinstance(self.hp[1], list):
                for k in xrange(len(self.hp[1])):
                    if self.hp[1][k]:
                        tmp = Rk2[:,:,k]/self.p['l'][k]**2
                        tmp *= R2l2**(0.5*self.p['gamma'] - 1)
                        Kprime[:,:,h] = where(R2l2 != 0.0,
                                w2*self.p['gamma']/self.p['l'][k]*tmp*K0, 0.0)
                        h += 1
            if self.hp[2]:
                Kprime[:,:,h] = where(R2l2 != 0.0,
                            -w2*R2l2**(0.5*self.p['gamma'])*log(R2l2)*K0, 0.0)
            return (w2*K0, Kprime)

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
    def __call__(self, Rk2, grad=False, **options):
        if not isinstance(self.p['l'], list):
            R2l2 = sum(Rk2,2)/self.p['l']**2
        else:
            R2l2 = zeros(Rk2.shape[:2])
            for k in xrange(Rk2.shape[2]):
                R2l2 += Rk2[:,:,k]/self.p['l'][k]**2
        w2 = self.p['w']**2
        tmp = 1.0 + R2l2/(2.0*self.p['alpha'])
        K0 = tmp**(-self.p['alpha'])
        if not grad:
            return w2*K0
        else:
            Kprime = empty((Rk2.shape[0], Rk2.shape[1], self.Nhp))
            h = 0
            if self.hp[0]:
                Kprime[:,:,h] = 2.0*self.p['w']*K0
                h += 1
            if self.hp[1] and not isinstance(self.hp[1], list):
                Kprime[:,:,h] = w2*R2l2*K0/(self.p['l']*tmp)
                h += 1
            elif isinstance(self.hp[1], list):
                for k in xrange(len(self.hp[1])):
                    if self.hp[1][k]:
                        Kprime[:,:,h] = w2*Rk2[:,:,k]*K0/(self.p['l'][k]**3*tmp)
                        h += 1
            if self.hp[2]:
                Kprime[:,:,h] = w2*((tmp-1)/tmp - log(tmp))*tmp**(-self.p['alpha'])
            return (w2*K0, Kprime)

# -- would like to add periodic, but it would require general handling of Rs --


class KernelSum(Kernel):
    """Modified Kernel class that holds the sum of Kernel objects."""
    def __init__(self, k1, k2):
        self.terms = [k1, k2]
        # Can we get away with not creating self.p or self.hp here, i.e.
        # do we only reference the parameters at the individual-kernal level?
        (self.Np, self.p) = (k1.Np + k2.Np, [k1.p, k2.p])
        (self.Nhp, self.hp) = (k1.Nhp + k1.Nhp, [k1.hp, k2.hp])
    
    def __add__(self, other, self_on_right=False):
        if isinstance(other, KernelSum):
            self.terms += other.terms
        elif isinstance(other, Kernel):
            self.terms += [other]
        else:
            # TODO: throw an error!
            pass
        self.Np += other.Np
        self.Nhp += other.Nhp
        if not self_on_right:
            self.p += [other.p]
            self.hp += [other.hp]
        else:
            self.p = [other.p] + self.p
            self.hp = [other.hp] + self.hp
        return self
    
    def declare_hyper(self, hyper_params):
        # TODO: throw and error if length of hyper_params doesn't match terms.
        for k in range(len(self.terms)):
            self.Nhp += self.terms[k].declare_hyper(hyper_params[k])
        return self.Nhp
    
    def map_hyper(self, p_mapped, unmap=False):
        # TODO: throw and error if the length of p_mapped is not Nhyper.
        i = 0
        for k in range(len(self.terms)):
            self.terms[k].map_hyper(p_mapped[i:i+self.terms[k].Nhp], unmap)
            i += self.terms[k].Nhp
        return (self, p_mapped)
    
    def __call__(self, Rk2, grad=False, **options):
        if not grad:
            K = zeros(Rk2.shape[:2])
            for kern in self.terms:
                K += kern(Rk2)
            return K
        else:
            (K, Kprime) = (0.0, empty((Rk2.shape[0], Rk2.shape[1], self.Nhp)))
            h = 0
            for kern in self.terms:
                (K_t, Kprime[:,:,h:h+kern.Nhp]) = kern(Rk2, grad)
                K += K_t
                h += kern.Nhp
            return (K, Kprime)


class KernelProd(Kernel):
    """Modified Kernel class that holds the product of Kernel objects."""
    def __init__(self, k1, k2):
        self.terms = [k1, k2]
        (self.Np, self.p) = (k1.Np + k2.Np, [k1.p, k2.p])
        (self.Nhp, self.hp) = (k1.Nhp + k1.Nhp, [k1.hp, k2.hp])
    
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
        if not self_on_right:
            self.p += [other.p]
            self.hp += [other.hp]
        else:
            self.p = [other.p] + self.p
            self.hp = [other.hp] + self.hp
        return self
    
    def declare_hyper(self, hyper_params):
        # TODO: throw and error if length of hyper_params doesn't match terms.
        for k in range(len(self.terms)):
            self.Nhp += self.terms[k].declare_hyper(hyper_params[k])
        return self.Nhp
    
    def map_hyper(self, p_mapped, unmap=False):
        # TODO: throw and error if the length of p_mapped is not Nhyper.
        i = 0
        for k in range(len(self.terms)):
            self.terms[k].map_hyper(p_mapped[i:i+self.terms[k].Nhp], unmap)
            i += self.terms[k].Nhp
        return (self, p_mapped)
    
    def __call__(self, Rk2, grad=False, **options):
        if not grad:
            return prod([kern(Rk2, grad) for kern in self.terms])
        else:
            (K, Kprime) = (0.0, empty((Rk2.shape[0], Rk2.shape[1], self.Nhp)))
            h = 0
            for kern in self.terms:
                (K_t, Kprime[:,:,h:h+kern.Nhp]) = kern(Rk2, grad)
                K *= K_t
                h += kern.Nhp
            return (K, Kprime)