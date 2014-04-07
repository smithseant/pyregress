# -*- coding: utf-8 -*-
"""
Docstring for the features module

Includes additional features for pyregress includeing prior distributions for
    hyper-parameters and a class for derivative inputs

Provided prior distributions (log(P) and d_log(P))
    Currently include log-Normal, Jeffreys', and constant distributions.
    Marginalized class also created to hold space of hyper-parameters that are 
    not being explored.
"""

from numpy import array, sum, divide, concatenate, squeeze, mean, amin, copy
from scipy import log, sqrt, pi
from scipy.special import gamma

class shift_to_zero:
    """shift data above zero and (optional) scale be mean value"""
    def __init__(self):
        self._shift = 0.
        self._scale = 1.
    
    def __call__(self,original_data,scale=False):
        data = copy(original_data)
        if (data < 0.).any():
            self._shift = amin(data)
            data -= self._shift
        if scale == True:
            self._scale = mean(data)
            data /= self._scale
        return data

    def reverse(self,data):
        return (data*self._scale)+self._shift


class derivative:
    """Derivative input class."""
    def __init__(self, *args):
        #deduce if three arays were passed for an array of three values
        self._derivative_position = 0
        
        if len(args) == 3:
            if isinstance(args[0],float):
                self._x    = array([args[0]])
                self._dy   = array([args[1]])
                self._xref = array([args[2]])
            else:
                self._x    = args[0]
                self._dy   = args[1]
                self._xref = args[2]
        # TODO: output error is more than 3 args
    def merge(self,x,y):
        self._derivative_position = len(y)
        new_x = concatenate((squeeze(x.T),self._x),axis=1)
        new_y = concatenate((squeeze(y),self._dy),axis=1)
        return new_x,new_y
        
    def separate(self,x,y):
        old_x = x[:self._derivative_position]
        old_y = y[:self._derivative_position]
        return old_x,old_y
    
"""Prior hyper-parameter distributions"""

class HyperPrior:
    """Base class for hyper-parameter prior distriubution classes"""
    def __init__(self, guess=1.):
        self._guess = guess
    
    def __call__(self):
        pass
    

class logNormal(HyperPrior):
    """Log Normal distribution class for hyper-parameter priors.
        To keep parameter positive"""
    def __init__(self, guess=1., **args):
        self._guess = guess
        if args.has_key( "mean" ) and args.has_key( "std" ):
            self._mu = log(args["mean"]**2/ \
                        (args["std"]**2 + args["mean"]**2))
            self._sigma = sqrt(log(1 + args["std"]**2/args["mean"]**2))   
    
    def auto_fill(self,Rk2):
        mean = sum(Rk2)/(Rk2 != 0).sum()
        std = sum((Rk2-mean)**2)/((Rk2 != 0).sum()-1.0)
        self._mu = log(mean**2/(std**2 + mean**2))
        self._sigma = sqrt(log(1 + std**2/mean**2))
        
    def __call__(self, x, derivative=False):
        # TODO: if mean and std not defined, use auto_fill with Rk2      
        #if not hasattr(self,'mu'):
        #   self.auto_fill(self.Rk2)        
        
        twosigsqr = 2.0*self._sigma**2
        log_pdf = -log(self._sigma*x)-0.5*log(2.0*pi)- \
                    (log(x)-self._mu)**2/twosigsqr
        if derivative == False: 
            return log_pdf
        else:             
            log_dpdf = (-self._mu + self._sigma**2 + \
                        log(x))/(self._sigma**2 * x)
            return log_pdf, log_dpdf
            
class jeffreys(HyperPrior):
    """Jefferys' distribution class for hyper-parameter priors.
        Non informative prior"""
    def __call__(self, x, derivative=False):
        log_pdf = -log(x)
        if derivative == False:
            return log_pdf
        else:
            log_dpdf = divide(-1.0,x)
            return log_pdf, log_dpdf
        
class constant(HyperPrior):
    """Constant class for hyper-parameter"""
    def __call__(self, x, derivative=False):
        if derivative == False:
            return array([1.0])
        else:
            return array([1.0]),array([0.0])
            
class marginalized(HyperPrior):
    """Class for when hyper-parameter is being marginalized"""
    def __call__(self, x, derivative=False):
        if derivative == False:
            return array([1.0])
        else:
            return array([1.0]),array([])
            
class Beta(HyperPrior):
    """Beta distribution class for hyper-parameter priors.
        For parameters bounded by [0,1]"""
    def __init__(self, a, b, guess=1.):
        self._guess = guess
        self._a = a
        self._b = b
        
    def __call__(self, x, derivative=False):
        a = self._a-1.
        b = self._b-1.
        log_pdf = a*log(x) + b*log(1.-x)
        if derivative==False:
            return log_pdf
        else:
            log_dpdf = a/x + b/(x-1.)
            return log_pdf, log_dpdf
            
class Gamma(HyperPrior):
    """Gamma distribution class for hyper-parameter priors"""
    def __init__(self, k, theta, guess=1.):
        self._guess = guess
        self._k = k
        self._theta = theta
        self._demonenator = k*log(theta) +  log(gamma(k))
        
    def __call__(self, x, derivative=False):
        k = self._k - 1.
        log_pdf = k*log(x) -x/self._theta - self._demonenator
        if derivative==False:
            return log_pdf
        else:
            log_dpdf = k/x - 1./self._theta
            return log_pdf, log_dpdf