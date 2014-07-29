# -*- coding: utf-8 -*-
"""
Docstring for the pyregress module - needs work.

For basic useage see the documentation in the GPP class.
This docstring covers more advanced topics.
Performance:
  Calculation time will greatly depend on which Blas/Lapack libs are used.
  Most default python/numpy/scipy packages are based on unoptimized libs
  (including linux repositories and downloaded executables).
Beyond commonly used kernels and basis functions:
  --give an overview of Kernel object and BaseBasis interface.--
Reading the code and development:
  Notation used throughout the code:
    X => independent variables,
    Y => dependent variables,
    Z => transformed dependent variable,
    d => data values (observations),
    i => inferred values,
    s => sampled values,
    R => distance (radius) in independent variable space,
    K => kernel values (covariance matrix),
    hp => parameters (hyper-parameters) of the kernel,
    p => derivative (prime) of a variable,
    H => explicit basis functions evaluated at X,
    Th => Linear coefficients to the basis functions.
"""
# Created Sep 2013
# @author: Sean T. Smith
__all__ = ['GPP']

#from termcolor import colored  # may not work on windows
from numpy import (array, diag, empty, eye, hstack, maximum, ndarray, ones,
                   resize, shape, sqrt, std, sum, tile, zeros)                   
from numpy.linalg.linalg import LinAlgError, svd
from numpy.random import randn
from scipy import log, pi
from scipy.linalg import cho_factor, cho_solve

from kernels import Kernel
from transforms import BaseTransform, Logarithm, Logit,  Probit, ProbitBeta
from multi_newton import MD_Newton

HLOG2PI = 0.5*log(2.0*pi)

class GPP:
    """
    Doctring for the GPP class - needs work.
    
    Examples
    --------
    >>> import numpy as np
    >>> from pyregress import GPP, Noise, SquareExp, RatQuad
    >>> Xd = array([[0.1], [0.3], [0.6]])
    >>> Yd = array([[0.0], [1.0], [0.5]])
    >>> myGPP = GPP( Xd, Yd, Noise(w=0.1) + SquareExp(w=1.0, l=0.3) )
    >>> print myGPP( np.array([[0.2]]) )
    [[ 0.52641732]]
    
    >>> Xd = array([[0.00, 0.00], [0.50,-0.10], [1.00, 0.00],
    ...             [0.15, 0.50], [0.85, 0.50], [0.50, 0.85]])
    >>> Yd = array([[0.10], [0.30], [0.60], [0.70], [0.90], [0.90]])
    >>> myGPP = GPP(Xd, Yd, RatQuad(w=0.6, l=0.3, alpha=1.0),
    ...             explicit_basis=[0, 1], transform='Probit')
    >>> print myGPP( np.array([[0.10, 0.10], [0.50, 0.42]]) )
    [[ 0.22770558]
     [ 0.78029862]]
    """
    def __init__(self, Xd, Yd, Cov, Xscaling=None,
                 Ymean=None, explicit_basis=None, transform=None,
                 minimize_hp=True):
        """
        Create a GPP object and prepare for inference.
        
        Arguments
        ---------
        Xd:  array-2D,
            independent-variable observed values. First dimension is for
            multiple observations, second dimension for multiple variables.
        Yd:  array-1D [or column-shaped 2D],
            dependent-variable observed values - same length as the first
            dimension of Xd.
        Cov:  Kernel object,
            prior covariance kernel. Options include: Noise, SquareExp,
            GammaExp, RatQuad, or the sum of any of these.
        Xscaling:  string or array-1D (optional),
            pre-scaling of the independent variables (kernel anisotropy).
            Range scaling: 'range'; standard deviation scaling: 'std'; and
            manual scaling: array (same length as the second dimension of Xd).
        Ymean:  function (optional),
            prior mean of the dependent variable at Xd & Xi. It must accpet
            input in form of Xd, and must provide output the same shape as Yd.
            If omitted, a prior mean of zero is assumed.
        explicit_basis:  list of ints (optional),
            explicit basis functions are specified by any combination of the
            integers: 0, 1, 2 - each corresponding to its polynomial order.
        transform:  string or BaseTransform object (optional)
            specify a dependent variable transformation with the name of a
            BaseTransform class (as a string) or a BaseTransform object.
            Options include: Logarithm, Logit, Probit, or ProbitBeta.
        
        Raises
        ------
        InputError:
            an exception is thrown for incompatible format of any inputs.
        """
        self.__call__ = self.inference
        
        # Independent variables
        if Xd.ndim == 1:
            self.Xd = Xd.reshape((-1, 1))
        elif Xd.ndim == 2:
            self.Xd = Xd
        else:
            raise InputError("GPP argument Xd must be a 2D array.", Xd)
        self.Nd, self.Nx = Xd.shape
        if not Xscaling:
            self.xscale = ones(self.Nx)
        elif Xscaling == 'range':
            self.xscale = (Xd.max(0) - Xd.min(0))**2
        elif Xscaling == 'std':
            self.xscale = std(Xd, axis=0)
        elif Xscaling.shape == (self.Nx,):
            self.xscale = Xscaling
        else:
            raise InputError("GPP argument Xscaling must be one of: " +
                             "False, True, 'range', 'std', or 1D array " +
                             "(same length as the 2nd dim. of Xd)" , Xscaling)
        
        # Dependent variable
        if Yd.shape[0] != self.Nd:
            raise InputError("GPP argument Yd must have the same length as " +
                             "the 1st dim. of Xd.", Yd)
        self.Yd = Yd.reshape((-1, 1)).copy()
        if transform is None:
            self.trans = None
        elif isinstance(transform, basestring):
            self.trans = eval(transform+'(self.Yd)')
            self.Yd = self.trans(self.Yd)
        elif isinstance(transform, BaseTransform):
            self.trans = transform
            self.Yd = self.trans(self.Yd)
        else:
            raise InputError("GPP argument transform must be BaseTransform " +
                             "class (string of name) or object.", transform)
        self.prior_mean = Ymean
        if self.prior_mean is not None:
            if self.trans is None:
                self.Yd -= self.prior_mean(Xd).reshape((-1, 1))
            else:
                self.Yd -= self.trans(self.prior_mean(Xd).reshape(-1, 1))
        self.basis = explicit_basis
        if self.basis is not None:
            self.Nth, self.Hd = self._basis(Xd)
        # Kernel (prior covariance)
        self.kernel = Cov
        if not isinstance(Cov, Kernel):
            raise InputError("GPP argument Cov must be a Kernel object.", Cov)

        # Do as many calculations as possible in preparation for the inference
        # -- Create a separate function for the following? --
        self.Rdd = self._radius(self.Xd, self.Xd)
        if (self.kernel.Nhp > 0 and minimize_hp is True):
            self.maximize_hyper_posterior()
        self.Kdd = self.kernel(self.Rdd, block_diag=True)
        self.LKdd = cho_factor_gen(self.Kdd)
        self.invKdd_Yd = cho_solve_gen(self.LKdd, self.Yd)
        if self.basis is not None:
                self.invKdd_Hd = cho_solve_gen(self.LKdd, self.Hd)
                LSth = cho_factor_gen(self.Hd.T.dot(self.invKdd_Hd))
                self.Sth = cho_solve_gen(LSth, eye(self.Nth))
                self.Th = cho_solve_gen(LSth, self.Hd.T.dot(self.invKdd_Yd))
                HdTh = self.Hd.dot(self.Th)
                self.invKdd_HdTh = cho_solve_gen(self.LKdd, HdTh)           
    
    def _basis(self, X, grad=False):
        """Calculate the basis functions given independent variables."""
        if not ( isinstance(self.basis, list) and
                 all([[0, 1, 2].count(entry) == 1 for entry in self.basis]) ):
        # TODO: also check if there is less data than degrees of freedom.
            raise InputError("GPP argument explicit_basis must be a list " +
                         "with: 0, 1, and/or 2.", self.basis)
        # TODO: implement an interface for user defined basis functions.
        #elif isinstance(self.basis, basis_callable):
        # H = self.basis(X)
        # Nth = H.shape[1]
        N = X.shape[0]
        Nth = sum( self.Nx**array(self.basis) )
        H = empty((N, Nth))
        j = 0
        if self.basis.count(0):
            H[:, j] = 1.0
            j += 1
        if self.basis.count(1):
            H[:, j:j+self.Nx] = X
            j += self.Nx
        if self.basis.count(2):
            for ix in xrange(self.Nx):
                for jx in xrange(self.Nx):
                    H[:, j] = X[:, ix]*X[:, jx]
                    j += 1
        if not grad:
            return Nth, H

        Hp = zeros((N, self.Nx, Nth))
        j = 0
        if self.basis.count(0):
            j += 1
        if self.basis.count(1):
            for ix in xrange(self.Nx):
                Hp[:, ix, j] = 1.0
                j += 1
        if self.basis.count(2):
            for ix in xrange(self.Nx):
                for jx in xrange(self.Nx):
                    Hp[:, ix, j] += X[:, jx]
                    Hp[:, jx, j] += X[:, ix]
                    j += 1
        if not grad == 'Hess':
            return Nth, H, Hp

        Hpp = zeros((N, self.Nx, self.Nx, Nth))
        j = 0
        if self.basis.count(0):
            j += 1
        if self.basis.count(1):
            j += self.Nx
        if self.basis.count(2):
            for ix in xrange(self.Nx):
                Hpp[:, ix, ix, j] = 2.0
                j += self.Nx + 1
        return Nth, H, Hp, Hpp
    
    def _radius(self, X, Y):
        """Calculate the distance matrix (radius)."""
        # Previously used: cdist(X, Y, 'seuclidean',V=self.xscale),
        # which required: from scipy.spatial.distance import cdist.
        Nx, Ny = X.shape[0], Y.shape[0]
        Rk = empty((Nx, Ny, self.Nx))
        for k in xrange(self.Nx):
            Rk[:, :, k] = tile(X[:,[k]], (1, Ny)) - tile(Y[:,[k]].T, (Nx, 1))
            if isinstance(self.xscale, ndarray):
                Rk[:, :, k] /= self.xscale[k]
        return Rk
    
    def hyper_posterior(self, params=None, grad=True):
        """
        Negative log of the hyper-parameter posterior & its gradient.
        
        Arguments
        ---------
        params:  array-1D,
            hyper parameters in an array for the minimization routine.
        hp_mapped:  array-1D,
            hyper parameter values that map to self.Kernel.
        grad:  bool or string (optional),
            when grad is True, must return lnP_grad,
            when grad is 'Hess', must also return lnP_hess.
        
        Returns
        -------
        lnP_neg:  float,
            negative log of the hyper-parameter posterior.
        lnP_grad:  array-1D (optional - depending on argument grad),
            gradient of lnP_neg with respect to each hyper-parameter.
        lnP_hess:  array-2D (optional - depending on argument grad),
            Hessian matrix (2nd derivatives) of lnP_neg.
        """   
        Nd, Nhp = self.Nd, self.kernel.Nhp
        if not grad:
            K = self.kernel(self.Rdd, block_diag=True)
            lnprior = self.kernel._ln_priors(params)
        elif grad != 'Hess':
            K, Kp = self.kernel(self.Rdd, grad_hp=grad, block_diag=True)
            lnprior, dlnprior = self.kernel._ln_priors(params, grad=grad)
        else:
            K, Kp, Kpp = self.kernel(self.Rdd, grad_hp=grad, block_diag=True)
            lnprior, dlnprior, d2lnprior = \
                        self.kernel._ln_priors(params, grad=grad)
        try:
            LK = cho_factor(K)
        except LinAlgError as e:
            print ("GPP method hyper_posterior failed to factor the " +
                   "data kernel. This is most often an indication that the " +
                   "minimization routine is not converging.")
            print ('Current hyper-parameter values: ')
            print (repr(params))
            raise e
        invK_Y = cho_solve(LK, self.Yd)
        lnP_neg = ( float(self.Nd)*HLOG2PI + sum(log(diag(LK[0]))) +
                    0.5*self.Yd.T.dot(invK_Y) - lnprior )
                    
        if self.basis is not None:
            Nth = self.Nth
            beta = cho_solve(LK, self.Hd)
            LSth = cho_factor(self.Hd.T.dot(beta))
            Sth = cho_solve(LSth, eye(self.Nth))
            Th = cho_solve(LSth, self.Hd.T.dot(invK_Y))
            betaTh = beta.dot(Th)
            lnP_neg -= ( float(self.Nth)*HLOG2PI - sum(log(diag(LSth[0]))) +
                         0.5*Th.T.dot(self.Hd.T).dot(betaTh) )
        if not grad:
            return lnP_neg
        
        # grad == True or 'Hess':
        invK = cho_solve(LK, eye(Nd))
        invK_aa = invK - invK_Y.dot(invK_Y.T)
        lnP_grad = empty(Nhp)
        for j in xrange(Nhp):
            lnP_grad[j] = 0.5*sum(invK_aa.T * Kp[:,:,j]) - 1.0*dlnprior[j]
        if self.basis is not None:
            diff2 = betaTh.T - 2.0*invK_Y.T
            for j in xrange(Nhp):
                bKpb = beta.T.dot(Kp[:,:,j]).dot(beta)
                lnP_grad[j] -= 0.5*( sum(bKpb.T*Sth) +
                                     diff2.dot(Kp[:,:,j]).dot(betaTh) )
        if grad != 'Hess':
            return lnP_neg, lnP_grad
        
        # grad == 'Hess':
        invK_Kp, aa_Kp = empty((Nd, Nd, Nhp)), empty((Nd, Nd, Nhp))
        for j in xrange(Nhp):
            invK_Kp[:,:,j] = invK.dot(Kp[:,:,j])
            a_Kp = invK_Y.T.dot(Kp[:,:,j])
            aa_Kp[:,:,j] = 2.0*invK_Y.dot(a_Kp)
        lnP_hess = empty((Nhp, Nhp))
        for j in xrange(Nhp):
            for i in xrange(j + 1):
                lnP_hess[i, j] = 0.5*sum(
                    invK_aa.T*Kpp[:,:,i,j] - invK_Kp[:,:,i].T*invK_Kp[:,:,j] +
                    aa_Kp[:,:,i].T*invK_Kp[:,:,j] ) - d2lnprior[i,j]
                lnP_hess[j, i] = lnP_hess[i, j]
        if self.basis is not None:
            diff1 = betaTh.T - invK_Y.T
            bSb_2invK = beta.dot(Sth).dot(beta.T) - 2.0*invK
            bKp, ThbKp = empty((Nth, Nd, Nhp)), empty((Nd, Nhp))
            diff1Kpb, diff2Kp = empty((Nth, Nhp)), empty((Nd, Nhp))
            for j in xrange(Nhp):
                bKp[:,:,j] = beta.T.dot(Kp[:,:,j])
                ThbKp[:,j] = betaTh.T.dot(Kp[:,:,j])
                diff1Kpb[:,j] = diff1.dot(Kp[:,:,j]).dot(beta)
                diff2Kp[:,j] = diff2.dot(Kp[:,:,j])
            for j in xrange(Nhp):
                for i in xrange(Nhp):
                    my_mess = ( beta.T.dot(Kpp[:,:,i,j]).dot(beta) +
                                bKp[:,:,i].dot(bSb_2invK).dot(bKp[:,:,j].T) )
                    lnP_hess[i, j] -= 0.5 * ( sum(my_mess.T * Sth) +
                              diff2.dot(Kpp[:,:,i,j]).dot(betaTh) -
                              2.0*diff2Kp[:,i].dot(invK).dot(ThbKp[:,j].T) +
                              2.0*diff1Kpb[:,i].dot(Sth).dot(diff1Kpb[:,j].T) )
                    
        return lnP_neg, lnP_grad, lnP_hess
    
    
    def maximize_hyper_posterior(self, hyper_params=None):
        """
        Find the maximum of the hyper-parameter posterior.
        
        Arguments
        ---------
        hyper_params:  list of bools & HyperPriors, possibly nested (optional),
            indicates which kernel parameters are hyper-parameters by
            inputting thier prior. The nested structure corresponds to
            that from the argument Cov in GPP.__init__.
        """

        # Setup hyper-parameters & map values from a single array
        all_hyper, bounds = self.kernel._map_hyper()
        lo, hi = [], []
        [(lo.append(bounds[i+i]), hi.append(bounds[2*i+1]))
            for i in xrange(len(bounds)/2)]
        
        # Perform minimization
        MD_Newton(self.hyper_posterior, all_hyper,
                  options={'tol':1e-6, 'maxiter':200, 'bounds':(lo, hi)})
        
        all_hyper, bounds = self.kernel._map_hyper(all_hyper, unmap=True)
        return self, all_hyper
    
    
    def inference(self, Xi, infer_std=False, untransform=True, sum_terms=True,
                  exclude_mean=False, grad=False):
        """
        Make inferences (interpolation or regression) at specified locations.
        Limited to a single value of each hyper-parameters.
        This method is invoked when the GPP object is called as a function.
        
        Arguments
        ---------
        Xi:  array-2D,
            independent variables - where to make inferences. The first
            dimension is for multiple inferences, and second dimension must
            match the second dimension of the argurment Xd from __init__.
        infer_std:  bool or 'covar' (optional),
            if True, return the inferred standard deviation;
            if 'covar', return the full posterior covariance matrix.
        untransform:  bool (optional),
            if False, any inverse transformation is suppressed.
        sum_terms:  bool, int or list of ints (optional),
            if int or list of ints, then use only this subset of terms of the
            sum kernel, by index (Cov in GPP.__init__ must be a KernelSum).
            If True, all terms are included.
        exclude_mean:  bool (optional),
            if False include prior mean and basis functions, otherwise don't.
        grad:  bool or 'Hess' (optional),
            if True or 'Hess' return the gradient of the dependent variable,
            if 'Hess' also return the second derivatives.
        
        Returns
        -------
        post_mean:  array-2D,
            inferred mean at each location in the argument Xi.
        post_std: array-2D or list (optional - depending on infer_std),
            inferred standard deviation of the inferrences
            (for any inverse transformation, both the positive and negative
            standard deviations are returned - in that order).
        
        Raises
        ------
        InputError:
            an exception is thrown for incompatible format of any inputs.
        
        Note
        ----
        If prior_mean was specified for GPP class object, this function
            will also be applied to Xi data.
        """
        
        # TODO: calculation of the posterior mean of gradient and Hessian.        
        
        # Independent variables
        if Xi.ndim == 1:
            Xi = Xi.reshape((-1, 1))
        if Xi.ndim != 2 or Xi.shape[1] != self.Nx:
            raise InputError("GPP object argument Xi must be a 2D array " +
                             "(2nd dimension must match that of Xd.)", Xi)
        
        # Mixed i-d kernel & inference of posterior mean
        Rid = self._radius(Xi, self.Xd)
        
        if grad is False:
            Kid = self.kernel(Rid, block_diag=False, sum_terms=sum_terms)
        elif grad is True:
            Kid, Kid_grad = self.kernel(Rid, block_diag=False, 
                                        sum_terms=sum_terms, grad_r=grad)
        else:
            Kid, Kid_grad, Kid_hess = self.kernel(Rid, block_diag=False, 
                                                  sum_terms=sum_terms, 
                                                  grad_r=grad)
        
        if self.basis is None or exclude_mean:
            post_mean = Kid.dot(self.invKdd_Yd)
        else:
            post_mean = Kid.dot(self.invKdd_Yd - self.invKdd_HdTh)
            if grad is False:
                Nth, Hi = self._basis(Xi)
            elif grad is True:
                Nth, Hi, Hpi = self._basis(Xi, grad=grad)
            else:
                Nth, Hi, Hpi, Hppi = self._basis(Xi, grad=grad)
                
            post_mean += Hi.dot(self.Th)
            
        if grad is True or grad is 'Hess':
            post_mean_grad = empty((Rid.shape[0],Rid.shape[2]))
            if self.basis is None or exclude_mean:            
                for i in xrange(Rid.shape[2]):
                    post_mean_grad[:,i] = \
                        Kid_grad[:,:,i].dot(self.invKdd_Yd).reshape(-1)
            else:
                for i in xrange(Rid.shape[2]):
                    post_mean_grad[:,i] = \
                        Kid_grad[:,:,i].dot(self.invKdd_Yd-
                                            self.invKdd_HdTh).reshape(-1)
                post_mean_grad[:,:] += \
                    Hpi.dot(self.Th).reshape(shape(post_mean_grad))
                        
        if grad is 'Hess':
            post_mean_hess = empty((Rid.shape[0],Rid.shape[2],Rid.shape[2]))
            if self.basis is None or exclude_mean:            
                for i in xrange(Rid.shape[2]):
                    for j in xrange(Rid.shape[2]):
                        post_mean_hess[:,i,j] = \
                            Kid_hess[:,:,i,j].dot(self.invKdd_Yd).reshape(-1)
            else:
                for i in xrange(Rid.shape[2]):
                    for j in xrange(Rid.shape[2]):
                        post_mean_hess[:,i,j] = \
                            Kid_hess[:,:,i,j].dot(self.invKdd_Yd - 
                                                self.invKdd_HdTh).reshape(-1)
                post_mean_hess[:,:,:] += \
                            Hppi.dot(self.Th).reshape(shape(post_mean_hess))
        
        # Dependent variable
        if self.prior_mean is not None and not exclude_mean:
            Yi_mean = self.prior_mean(Xi).reshape((-1, 1))
            if self.trans is None or not untransform:
                post_mean = resize(post_mean, Yi_mean.shape) + Yi_mean
            else:
                post_mean = (resize(post_mean, Yi_mean.shape) + 
                            self.trans(Yi_mean))
        
        # Inference of posterior covariance
        if infer_std:
            Rii = self._radius(Xi, Xi)
            Kii = self.kernel(Rii, block_diag=True, sum_terms=sum_terms)
            post_covar = Kii - Kid.dot(cho_solve_gen(self.LKdd, Kid.T))
            if self.basis is not None:
                A = Hi - Kid.dot(self.invKdd_Hd)
                post_covar += A.dot(self.Sth.dot(A.T))
            post_var = maximum(0.0, diag(post_covar))
            post_std = sqrt(post_var).reshape((-1, 1))
        
        # Inverse transformation of the dependent variable
        if self.trans is not None and untransform:
            if infer_std:
                post_std = [self.trans(post_mean - post_std, inverse=True),
                            self.trans(post_mean + post_std, inverse=True)]
                post_mean = self.trans(post_mean, inverse=True)
                post_std = [post_std[0] - post_mean, post_mean - post_std[1]]
            else:
                post_mean = self.trans(post_mean, inverse=True)
        
        if grad is True:
            post_mean = post_mean, post_mean_grad
        if grad is 'Hess':
            post_mean = post_mean, post_mean_grad, post_mean_hess
        
        if not infer_std:
            return post_mean
        elif infer_std == 'covar':
            return post_mean, post_covar
        else:
            return post_mean, post_std
                
    
    def sample(self, Xs, Nsamples=1, sum_terms=True, exclude_mean=False, 
               grad=False):
        """
        Sample the Gassian process at specified locations.
        
        Arguments
        ---------
        Xs:  array-2D,
            independent variables - where to sample. First dimension is for
            multiple inferences, and second dimension mustmatch the second
            dimension of the argurment Xd from GPP.__init__.
        Nsamples: int (optional),
            allows the calculation of multiple samples at once.
        sum_terms:  bool, int or list of ints (optional),
            if int or list of ints, then use only this subset of terms of the
            sum kernel, by index (Cov in GPP.__init__ must be a KernelSum).
            For regression, standard use includes all terms except the noise.
            If True, all terms are included.
        exclude_mean:  bool (optional),
            if False include prior mean and basis functions, otherwise don't.
        
        Returns
        -------
        Ys:  array-2D,
            sample value at each location in the argument Xs.
        
        Raises
        ------
        InputError:
            an exception is thrown for incompatible format of any inputs.
        """
        Nx = Xs.shape[0]
        Ys_post, Cov = self.inference(Xs, infer_std='covar',
                                sum_terms=sum_terms, exclude_mean=exclude_mean,
                                grad=grad)
        if grad is True:
            Ys_post, Ys_post_grad = Ys_post
        if grad is 'Hess':
            Ys_post, Ys_post_grad, Ys_post_hess = Ys_post
        
        
        Z = randn(Nx, Nsamples)
        U,S,V = svd(Cov)
        sig = U.dot(diag(sqrt(S)))
        Ys = empty((Nx, Nsamples))
        for i in xrange(Nsamples):
            Ys[:, i] = sig.dot(Z[:, i])
            Ys[:, i] += Ys_post[:, 0]
        if self.trans is not None:
            Ys = self.trans(Ys, inverse=True)
        if grad is False:
            return Ys
        if grad is True:
            return Ys, Ys_post_grad
        if grad is 'Hess':
            return Ys, Ys_post_grad, Ys_post_hess


def cho_factor_gen(A, lower=False, **others):
    """Generalize scipy's cho_factor to handle arrays of lenth zero."""
    if A.size == 0:
        return empty(A.shape), lower
    else:
        try:
            return cho_factor(A, lower=lower, **others)
        except LinAlgError as e:
            print ("GPP method __init__ failed to factor data kernel." +
                   "This often indicates that X has near duplicates or " +
                   "the noise kernel has too small of weight.")
            raise e
        
def cho_solve_gen(C, b, **others):
    """Generalize scipy's cho_solve to handle arrays of lenth zero."""
    if C[0].size == 0:
        return empty(b.shape)
    else:
        return cho_solve(C, b, **others)


class Error(Exception):
    """Base class for exceptions in the pyregress module."""
    pass

class InputError(Error):  # -- not a ValueError? --
    """Exception raised for errors in input arguments."""
    def __init__(self, msg, input_argument=None):
        """
        Initialize an InputError.
        
        Arguments
        ---------
            msg:  string,
                explanation of the error.
            input_argument:  any (optional),
                input argument that is the source of error. Provided so
                the value can be reported when the error is caught.
        """
        self.args = (msg,)
        self.input_argument = input_argument


if __name__ == "__main__":
    from numpy import linspace, hstack, meshgrid, reshape, vstack
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    #from pyregress import *
    from kernels import Noise, SquareExp, RatQuad
    from transforms import Probit
    from hyper_params import LogNormal
    
    # TODO: Examples that provide verification!
    
    # Example 1:
    # Simple case, 1D with three data points and one regression point
    Xd1 = array([[0.1], [0.3], [0.6]])
    Yd1 = array([[0.0], [1.0], [0.5]])
    myGPP1 = GPP( Xd1, Yd1, Noise(w=0.1) + SquareExp(w=0.75, l=0.25) )
    #myGPP1 = GPP( Xd1, Yd1, SquareExp(w=0.75, l=0.25) )
    xi1 = array([[0.2]])
    #yi1 = myGPP1( xi1 )
    yi1, yi1_grad = myGPP1( xi1, grad=True, sum_terms=[1] )
    print 'Example 1:'
    print 'x = ', xi1, ',  y = ', yi1
    yi1_, yi1_grad_, yi1_hess_ = myGPP1( Xd1, grad='Hess', sum_terms=[1] )
    
    # Example 2:
    # 2D with six data points and two regression points
    Xd2 = array([[0.00, 0.00], [0.50,-0.10], [1.00, 0.00],
                 [0.15, 0.50], [0.85, 0.50], [0.50, 0.85]])
    Yd2 = array([[0.10], [0.30], [0.60], [0.70], [0.90], [0.90]])
    K2 =  RatQuad(w=0.6, l=LogNormal(guess=0.3, std=0.25),alpha=1.0)
    myGPP2 = GPP(Xd2, Yd2, K2, explicit_basis=[0, 1], transform='Probit')
    print 'Optimized value of the hyper-parameters:', myGPP2.kernel.get_hp()
    xi2 = array([[0.1, 0.1], [0.5, 0.42]])
    yi2, yi2_grad = myGPP2( xi2, grad=True )  
    print 'Example 2:'
    print 'x = ', xi2
    print 'y = ', yi2
    
    # Figures to support the examples
    # fig. example 1
    Xi1 = linspace(0.0, 0.75, 200)
    Yi1, Yi1std = myGPP1(Xi1, infer_std=True, sum_terms=1)
    Yi1, Yi1std = Yi1.reshape(-1), Yi1std.reshape(-1)
    
    Xig1 = (xi1 + 0.025*array([-1.0, 1.0])).reshape(-1,1)
    Yig1 = (yi1 + yi1_grad*0.025*array([-1.0, 1.0])).reshape(-1,1)
    Xdg1 = Xd1 + 0.025*array([-1.0, 1.0])
    Ydg1 = yi1_ + yi1_grad_*0.025*array([-1.0, 1.0])
    

    #Xig2_d1 = vstack((xi2[0,1]+0.025*array([-1.0, 1.0]),
    #                  array([xi2[1,1],xi2[1,1]])))
    #Xig2_d2 = vstack((array([xi2[0,1],xi2[0,1]]),
    #                  xi2[1,1]+0.025*array([-1.0, 1.0])))
    #Yig2 = yi2[1] + yi2_grad[:,1].reshape(-1,1)*0.025*array([-1.0, 1.0])
    
    
    fig1 = plt.figure(figsize=(5, 3), dpi=150)
    p1, = plt.plot(Xd1, Yd1, 'ko', label='Data')
    p2, = plt.plot(Xi1, Yi1, 'b-', linewidth=2.0, label='Inferred mean')
    plt.fill_between(Xi1, Yi1-Yi1std, Yi1+Yi1std, alpha=0.25)
    p3 = plt.Rectangle((0.0, 0.0), 1.0, 1.0, facecolor='blue',
                       alpha=0.25, label='Uncertainty (one std.)')
    p4, = plt.plot(xi1, yi1, 'ro', label='Example regression point')
    p5 = plt.plot(Xig1, Yig1, 'r-', linewidth=3.0, label='Inferred slope') 
    p6 = plt.plot(Xdg1[0,:], Ydg1[0,:], 'r-', Xdg1[1,:], Ydg1[1,:], 'r-', 
                  Xdg1[2,:], Ydg1[2,:], 'r-', linewidth=3.0)
    fig1.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)
    plt.title('Example 1', fontsize=16)
    plt.xlabel('Independent Variable, X', fontsize=12)
    plt.ylabel('Dependent Variable, Y', fontsize=12)
    plt.legend(loc='best', numpoints=1, prop={'size':8})

    # fig. example 2
    Ni = (30, 30)
    xi_1 = linspace(-0.2, 1.2, Ni[0])
    xi_2 = linspace(-0.2, 1.0, Ni[1])
    Xi_1, Xi_2 = meshgrid(xi_1, xi_2, indexing='ij')
    Xi2 = hstack([Xi_1.reshape((-1, 1)), Xi_2.reshape((-1, 1))])
    Yi2, Yi2std = myGPP2.inference(Xi2, infer_std=True)
        
    fig = plt.figure(figsize=(7, 5), dpi=150)
    ax = fig.gca(projection='3d')
    ax.plot_surface(Xi_1, Xi_2, Yi2.reshape(Ni), alpha=0.75,
                    linewidth=0.5, cmap=mpl.cm.jet, rstride=1, cstride=1)
    ax.plot_surface(Xi_1, Xi_2, reshape(Yi2+Yi2std[0], Ni), alpha=0.25,
                    linewidth=0.25, color='black', rstride=1, cstride=1)
    ax.plot_surface(Xi_1, Xi_2, reshape(Yi2-Yi2std[1], Ni), alpha=0.25,
                    linewidth=0.25, color='black', rstride=1, cstride=1)
    ax.scatter(Xd2[:, 0], Xd2[:, 1], Yd2, c='black', s=35)
    #ax.plot(Xig2_d1[0,:], Xig2_d1[1,:], Yig2[1,:], 'r-', linewidth=3.0)
    #ax.plot(Xig2_d2[0,:], Xig2_d2[1,:], Yig2[0,:], 'r-', linewidth=3.0)
    ax.set_zlim([0.0, 1.0])
    ax.set_title('Example 2', fontsize=16)
    ax.set_xlabel('Independent Variable, X1', fontsize=12)
    ax.set_ylabel('Independent Variable, X2', fontsize=12)
    ax.set_zlabel('Dependent Variable, Y', fontsize=12)

    plt.show()
    