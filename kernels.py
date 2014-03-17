# -*- coding: utf-8 -*-
"""
Docstring for the kernels module - needs to be written
"""
# Created Sep 2013
# @author: Sean T. Smith

from abc import ABCMeta, abstractmethod
from numpy import array, empty, zeros, diag, sum, prod, where, ones, concatenate
from scipy import exp, log, sqrt, pi

"""
Provided prior distributions
"""
class logNormal:
    """Log Normal distribution class for hyper-parameter priors."""
    def __init__(self, **args):
        if args.has_key( "mean" ) and args.has_key( "std" ):
            self.mu = log(args["mean"]**2/(args["std"]**2 + args["mean"]**2))
            self.sigma = sqrt(log(1 + args["std"]**2/args["mean"]**2)) 
        #self.mu = sum(Rk2)/(Rk2 != 0).sum()
        #self.sigma = sum((Rk2-self.mu)**2)/((Rk2 != 0).sum()-1.0)
        
    
    def auto_fill(self,Rk2):
        mean = sum(Rk2)/(Rk2 != 0).sum()
        std = sum((Rk2-self.mu)**2)/((Rk2 != 0).sum()-1.0)
        self.mu = log(mean**2/(std**2 + mean**2))
        self.sigma = sqrt(log(1 + std**2/mean**2))
        
    def __call__(self, x, derivative=False):
        twosigsqr = 2.0*self.sigma**2
        #sigmaSqr2pi = self.sigma*sqrt(2.0*pi)
        #pdf = 1.0/(x*sigmaSqr2pi)*exp(-(log(x)-self.mu)**2/twosigsqr)
        
        log_pdf = -log(self.sigma*x)-0.5*log(2.0*pi)-(log(x)-self.mu)**2/twosigsqr
        if derivative == False: 
            return log_pdf
        else:            
            #t1 = exp(-(self.mu-log(x))**2/twosigsqr)
            #t2 = -self.mu + self.sigma**2 + log(x)
            #d_pdf = t1*t2/(sigmaSqr2pi * self.sigma**2 * x**2)  
            log_dpdf = (-self.mu + self.sigma**2 + log(x))/(self.sigma**2 * x)
            return log_pdf, log_dpdf
            
class jeffreys:
    """Jefferys' distribution class for hyper-parameter priors."""
    def __init__(self):
        pass
    def __call__(self, x, derivative=False):
        log_pdf = log(1.0/x)
        if derivative == False:
            return log_pdf
        else:
            log_dpdf = -1.0/x
            return log_pdf, logdpdf
        
class constant:
    """Constant class for when hyper-parameter is not being marginalized"""
    def __init__(self):
        pass
    def __call__(self, x, derivative=False):
        if derivative == False:
            return array([1.0])
        else:
            return array([1.0]),array([])

class Kernel:
    """
    Provide methods & an interface for kernels in the GPR class.
    
    User-defined kernels will need to inherit this baseclass and define
    both __init__ and __call__ methods in the derived class. This base
    class provides the abstract interface for the __call__ method and
    provides the methods: __init__, declare_hyper, and map_hyper.
    """
    __metaclass__ = ABCMeta
    
    def __init__(self, Nparams, params):
        """
        Create a Kernel object.
        
        Arguments
        ---------
        Nparams: int,
            number of kernel parameters for this specific kernel.
        params: list of floats or arrays,
            values for the kernel parameters.
        """
        # TODO: throw an error if the number of parameters doesn't match.
        if (Nparams != len(params)):
            raise InputError("Length of parameter vector input into kernel" +
                             "does not match specified number of parameters" +
                             Nparams, "!=", len(params))
        self.Np = Nparams
        self.p = params
        self.Nhp = 0
        self.hp = [False]*self.Np
        for i in range(len(self.p)):
            if isinstance(self.p[i], list):
                self.hp[i] = [False]*len(self.p[i])
        self.Prior = []
    
    @abstractmethod
    def __call__(self, Rk2, grad=False):
        """
        Calculate and return kernel values given the radius array.
        
        Arguments
        ---------
        self: Kernel,
            kernel parameters (self.p) and hyper-parameter labels (self.hp).
        Rk2: array-3D,
            directional square distance matrix (radius squared).
        grad: bool (optional),
            when grad is not False, must return gradK.
        
        Returns
        -------
        K: array-2D,
            kernel values - shape must match argument R.
        gradK: list of arrays (optional - depending on argument grad),
            partial of kernel with respect to each hyper parameter.
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
        
        Priors: list of functions defining hyper-parameter's 
            prior distribution for calculating posterier instead of
            marginal likelihood (set to constant if not being marginalized
            or by default if specified as True)
        
        Returns
        -------
        Nhyper: int,
            resulting number of hyper parameters.
        """       
                
        if not hyper_params or hyper_params=='none':
            self.Nhp = 0
            self.Prior = [constant()]
        elif (False not in hyper_params) and (not isinstance(hyper_params,list) ): #or hyper_params=='all':
            self.Nhp = self.Np
            self.hp[:] = [True]*self.Np
            self.Prior = hyper_params
            for i in range(len(self.p)):
                if isinstance(self.p[i], list):
                    self.Nhp += len(self.p[i]) - 1
                    self.hp[i] = [True]*len(self.p[i])
                    self.Prior[i] = [hyper_params[i]]*len(self.p[i])
        else:
            # TODO: throw an error if the number of parameters doesn't match.
            self.Nhp = len(hyper_params) - hyper_params.count(False)
            self.hp[:] = hyper_params[:]
            for i in range(self.Np):
                if isinstance(self.p[i], list):
                    if isinstance(hyper_params[i], list):
                        self.Nhp += len(hyper_params[i]) - hyper_params[i].count(False) - 1
                        self.Prior += hyper_params[i][:]
                    elif hyper_params[i]==False or hyper_params[i]=='none':
                        self.hp[i] = [False]*len(self.p[i])
                        self.Prior += [constant()]*len(self.p[i])
                    elif hyper_params[i]=='all':
                        self.hp[i] = [True]*len(self.p[i])
                        self.Nhp += len(self.p[i]) - 1
                        self.Prior += [hyper_params[i]]*len(self.p[i])
                    elif hyper_params[i]!=False:
                        self.hp[i] = [True]*len(self.p[i])
                        self.Nhp += len(self.p[i]) - 1
                        self.Prior += [hyper_params[i]]*len(self.p[i])
                else:
                    if hyper_params[i] == False:
                        self.Prior += [constant()]
                    else:
                        self.Prior += [hyper_params[i]]
                        
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
            if self.hp[i] is not False and (not isinstance(self.hp[i],list) ):
                if not unmap:
                    p_mapped[im] = self.p[i]
                    self.p[i] = p_mapped[im:im+1]
                else:
                    self.p[i] = p_mapped[im]
                im += 1
            elif isinstance(self.p[i], list):
                for j in range(len(self.p[i])):
                    if self.hp[i][j] is not False:
                        if not unmap:
                            p_mapped[im] = self.p[i][j]
                            self.p[i][j] = p_mapped[im:im+1]
                        else:
                            self.p[i][j] = p_mapped[im]
                        im += 1
        return (self, p_mapped)
        
    def calc_logP(self, params, grad=False):
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

class Noise(Kernel):
    r"""
    White noise kernel object.
    ..math::
        K_{i,j}(R) = w^2*\delta_{i,j}*\delta_D(R_{i,j}),
    where the parameter list includes only the weight, params = [w].
    White noise is discontinuous.
    """
    def __init__(self, params):
        super(Noise, self).__init__(1, params)
    def __call__(self, Rk2, grad=False):
        w2 = self.p[0]**2
        if Rk2.shape[0] == Rk2.shape[1]:
            R2diag = sum(Rk2.diagonal(), 0)
            K = diag( array([1.0*b for b in R2diag == 0.0]) )
        else:
            K = zeros(Rk2.shape[:2])
        if not grad:
            return w2*K
        else:
            Kprime = empty((Rk2.shape[0], Rk2.shape[1], self.Nhp))
            if self.hp[0]:
                Kprime[:,:,0] = 2.0*self.p[0]*K
            return (w2*K, Kprime)

class OU(Kernel):
    r"""
    Ornstein-Uhlenbeck kernel object.
    .. math::
        K(R) = w^2*\exp( -R/l ),
    where the parameter list is params = [w, l].
    Ornstein-Uhlenbeck is continuous, but not smooth.
    """
    def __init__(self, params):
        super(OU, self).__init__(2, params)
    def __call__(self, Rk2, grad=False):
        if not isinstance(self.p[1], list):
            R2l2 = sum(Rk2,2)/self.p[1]**2
        else:
            R2l2 = zeros(Rk2.shape[:2])
            for k in xrange(Rk2.shape[2]):
                R2l2 += Rk2[:,:,k]/self.p[1][k]**2
        w2 = self.p[0]**2
        Rl = R2l2**0.5
        K = exp(-Rl)
        if not grad:
            return w2*K
        else:
            Kprime = empty((Rk2.shape[0], Rk2.shape[1], self.Nhp))
            h = 0
            if self.hp[0]:
                Kprime[:,:,h] = 2.0*self.p[0]*K
                h += 1
            if self.hp[1] and not isinstance(self.hp[1], list):
                Kprime[:,:,h] = w2*Rl/self.p[1]*K
                h += 1
            elif isinstance(self.hp[1], list):
                for k in xrange(len(self.hp[1])):
                    if self.hp[1][k]:
                        Kprime[:,:,h] = where(Rl != 0.0,
                                     w2*Rk2[:,:,k]/(self.p[1][k]**3*Rl)*K, 0.0)
                        h += 1
            return (w2*K, Kprime)

class GammaExp(Kernel):
    r"""
    Gamma-exponential kernel object.
    .. math::
        K(R) = w^2*\exp( -(R/l)^{\gamma} ),
    where the parameter list is params = [w, l, gamma].
    Gamma-exponential is continuous, and when gamma=2 it is smooth.
    """
    def __init__(self, params):
        """Docstring under __init__"""
        super(GammaExp, self).__init__(3, params)
    def __call__(self, Rk2, grad=False):
        if not isinstance(self.p[1], list):
            R2l2 = sum(Rk2,2)/(self.p[1]**2)
        else:
            R2l2 = zeros(Rk2.shape[:2])
            for k in xrange(Rk2.shape[2]):
                R2l2 += Rk2[:,:,k]/(self.p[1][k]**2)
        w2 = self.p[0]**2
        K = exp(-R2l2**(0.5*self.p[2]))
        if not grad:
            return w2*K
        else:
            Kprime = empty((Rk2.shape[0], Rk2.shape[1], self.Nhp))
            h = 0
            if self.hp[0]:
                Kprime[:,:,h] = 2.0*self.p[0]*K
                h += 1
            if self.hp[1] and not isinstance(self.hp[1], list):
                tmp = w2*R2l2**(0.5*self.p[2])
                Kprime[:,:,h] = self.p[2]*tmp/self.p[1]*K
                h += 1
            elif isinstance(self.hp[1], list):
                for k in xrange(len(self.hp[1])):
                    if self.hp[1][k]:
                        tmp = Rk2[:,:,k]/self.p[1][k]**2
                        tmp *= R2l2**(0.5*self.p[2] - 1)
                        Kprime[:,:,h] = where(R2l2 != 0.0,
                                          w2*self.p[2]/self.p[1][k]*tmp*K, 0.0)
                        h += 1
            if self.hp[2]:
                Kprime[:,:,h] = where(R2l2 != 0.0,
                                    -w2*R2l2**(0.5*self.p[2])*log(R2l2)*K, 0.0)
            return (w2*K, Kprime)

class SquareExp(Kernel):
    r"""
    Squared-exponential kernel object.
    .. math::
        K(R) = w^2*\exp( -1/2 *(R/l)^2 ),
    where the parameter list is params = [w, l].
    Squared-exponential is continuous and infinitely differentiable.
    """
    def __init__(self, params):
        super(SquareExp, self).__init__(2, params)
    def __call__(self, Rk2, grad=False):
        
        if not isinstance(self.p[1], list):
            R2l2 = sum(Rk2,2)/(self.p[1]**2)
        else:
            R2l2 = zeros(Rk2.shape[:2])
            for k in xrange(Rk2.shape[2]):
                R2l2 += Rk2[:,:,k]/(self.p[1][k]**2)
        w2 = self.p[0]**2
        K = exp(-0.5*R2l2)
        if not grad:
            return w2*K
        else:
            Kprime = empty((Rk2.shape[0], Rk2.shape[1], self.Nhp))
            h = 0
            if self.hp[0]:
                Kprime[:,:,h] = 2.0*self.p[0]*K
                h += 1
            if self.hp[1] and not isinstance(self.hp[1], list):
                Kprime[:,:,h] = w2*R2l2/self.p[1]*K
                h += 1
            elif isinstance(self.hp[1], list):
                for k in xrange(len(self.hp[1])):
                    if self.hp[1][k]:
                        Kprime[:,:,h] = w2*Rk2[:,:,k]/self.p[1][k]**3*K
                        h += 1
            return (w2*K, Kprime)

class RatQuad(Kernel):
    r"""
    Rational-quadratic kernel object.
    .. math::
        K(R) = w^2*( 1 + \frac{R^2}{2*\alpha*l^2} )^{-\alpha},
    where the parameter list is params = [w, l, alpha].
    Rational quadratic is SE over a gamma distribution of length scales.
    """
    def __init__(self, params):
        super(RatQuad, self).__init__(3, params)
    def __call__(self, Rk2, grad=False):
        if not isinstance(self.p[1], list):
            R2l2 = sum(Rk2,2)/self.p[1]**2
        else:
            R2l2 = zeros(Rk2.shape[:2])
            for k in xrange(Rk2.shape[2]):
                R2l2 += Rk2[:,:,k]/self.p[1][k]**2
        w2 = self.p[0]**2
        tmp = 1.0 + R2l2/(2.0*self.p[2])
        K = tmp**(-self.p[2])
        if not grad:
            return w2*K
        else:
            Kprime = empty((Rk2.shape[0], Rk2.shape[1], self.Nhp))
            h = 0
            if self.hp[0]:
                Kprime[:,:,h] = 2.0*self.p[0]*K
                h += 1
            if self.hp[1] and not isinstance(self.hp[1], list):
                Kprime[:,:,h] = w2*R2l2*K/(self.p[1]*tmp)
                h += 1
            elif isinstance(self.hp[1], list):
                for k in xrange(len(self.hp[1])):
                    if self.hp[1][k]:
                        Kprime[:,:,h] = w2*Rk2[:,:,k]*K/(self.p[1][k]**3*tmp)
                        h += 1
            if self.hp[2]:
                Kprime[:,:,h] = w2*((tmp-1)/tmp - log(tmp))*tmp**(-self.p[2])
            return (w2*K, Kprime)
 
# -- would like to add periodic, but it would require general handling of R --

class KernelSum(Kernel):
    """Modified Kernel class that holds the sum of Kernel objects."""
    def __init__(self, k1, k2):
        self.terms = [k1, k2]
        (self.Np, self.p) = (k1.Np + k2.Np, [k1.p, k2.p])
        (self.Nhp, self.hp) = (k1.Nhp + k1.Nhp, [k1.hp, k2.hp])
        self.terms[0].Prior = []
        self.terms[1].Prior = []

    
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
    
    def __call__(self, Rk2, grad=False):
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
            
    def calc_logP(self, params, grad=False):
        logPrior = 0.0
        if grad == False:
            for k in xrange(len(self.terms)):
                logPrior += self.terms[k].calc_logP(params)
            return logPrior
        else:
            d_logPrior = array([]) 
            for k in xrange(len(self.terms)):
                (logP,d_logP) = self.terms[k].calc_logP(params,True)
                logPrior += logP
                d_logPrior = concatenate((d_logPrior,d_logP),axis=1)
            return logPrior, d_logPrior

class KernelProd(Kernel):
    """Modified Kernel class that holds the product of Kernel objects."""
    def __init__(self, k1, k2):
        self.terms = [k1, k2]
        (self.Np, self.p) = (k1.Np + k2.Np, [k1.p, k2.p])
        (self.Nhp, self.hp) = (k1.Nhp + k1.Nhp, [k1.hp, k2.hp])
        self.terms[0].Prior = []
        self.terms[1].Prior = []
    
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

    def __call__(self, Rk2, grad=False):
        if not grad:
            K = 0.0
            for kern in self.terms:
                K *= kern(Rk2, grad)
            return K
        else:
            (K, Kprime) = (0.0, empty((Rk2.shape[0], Rk2.shape[1], self.Nhp)))
            h = 0
            for kern in self.terms:
                (K_t, Kprime[:,:,h:h+kern.Nhp]) = kern(Rk2, grad)
                K *= K_t
                h += kern.Nhp
            return (K, Kprime)
            
    def calc_logP(self, params, grad=False):
        logPrior = 0.0
        if grad == False:
            for k in xrange(len(self.terms)):
                logPrior += self.terms[k].calc_logP(params)
            return logPrior
        else:
            d_logPrior = array([]) 
            for k in xrange(len(self.terms)):
                (logP,d_logP) = self.terms[k].calc_logP(params,True)
                logPrior += logP
                d_logPrior = concatenate((d_logPrior,d_logP),axis=1)
            return logPrior, d_logPrior