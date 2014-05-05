# -*- coding: utf-8 -*-
"""
Docstring for the pyregress module - needs work.

For basic useage see the documentation in the GPR class.
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

#from termcolor import colored  # may not work on windows
from numpy import (ndarray, array, empty, ones, eye, shape, tile,
                   maximum, diag, sum, sqrt, resize, size, std)
from numpy.linalg.linalg import LinAlgError, svd
from numpy.random import randn
from scipy import pi, log
#from pyregress import *
from kernels import Kernel
from transforms import BaseTransform, Probit
from multi_newton import MD_Newton
from hyper_params import cho_factor_mod as cho_factor
from hyper_params import cho_solve_mod as cho_solve

HLOG2PI = 0.5*log(2.0*pi)

class GPR:
    """
    Doctring for the GPR class - needs work.
    
    Examples
    --------
    >>> import numpy as np
    >>> from pyregress import GPR, Noise, SquareExp, RatQuad
    >>> Xd = array([[0.1], [0.3], [0.6]])
    >>> Yd = array([[0.0], [1.0], [0.5]])
    >>> myGPR = GPR( Xd, Yd, Noise(w=0.1) + SquareExp(w=1.0, l=0.3) )
    >>> print myGPR( np.array([[0.2]]) )
    [[ 0.52641732]]
    
    >>> Xd = array([[0.00, 0.00], [0.50,-0.10], [1.00, 0.00],
    ...             [0.15, 0.50], [0.85, 0.50], [0.50, 0.85]])
    >>> Yd = array([[0.10], [0.30], [0.60], [0.70], [0.90], [0.90]])
    >>> myGPR = GPR(Xd, Yd, RatQuad(w=0.6, l=0.3, alpha=1.0),
    ...             explicit_basis=[0, 1], transform='Probit')
    >>> myGPR.maximize_hyper_posterior([False, True, False])
    >>> print myGPR( np.array([[0.10, 0.10], [0.50, 0.42]]) )
    [[ 0.22770558]
     [ 0.78029862]]
    """
    def __init__(self, Xd, Yd, Cov, x_scaling=None,
                 y_mean=None, explicit_basis=None, transform=None):
        """
        Create a GPR object and prepare for inference.
        
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
        x_scaling:  string or array-1D (optional),
            scaling of the independent variables. With 'range', it uses range
            scaling. With an array having the same length as the second
            dimension of Xd, it uses manual scaling.  With 'std' it scales
            by the standard deviation.
        y_mean:  a function (optional),
            prior mean of the dependent variable at Xd & Xi.  Must take input
            of data in form of Xd, and output in same shape as Yd.  If
            omitted, uses a prior mean of zero.  Will be used in inference
            if supplied for GPR object.
        explicit_basis:  list (optional),
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
            raise InputError("GPR argument Xd must be a 2D array.", Xd)
        (self.Nd, self.Nx) = shape(Xd)
        if not x_scaling:
            self.xscale = ones(self.Nx)
        elif x_scaling == 'range':
            self.xscale = (Xd.max(0)-Xd.min(0))**2
        elif x_scaling == 'std':
            self.xscale = std(Xd,axis=0)
        elif shape(x_scaling) == (self.Nx,):
            self.xscale = x_scaling
        else:
            raise InputError("GPR argument x_scaling must be one of: " +
                             "False, True, 'range', 'std', or 1D array " +
                             "(same length  + as the second dimension of Xd)"
                             , x_scaling)
        
        # Dependent variable
        if Yd.shape[0] != self.Nd:
            raise InputError("GPR argument Yd must have the same length as " +
                             "the first dimension of Xd.", Yd)
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
            raise InputError("GPR argument transform must be BaseTransform " +
                             "class (string of name) or object.", transform)
        self.prior_mean = y_mean
        if self.prior_mean is not None:
            if self.trans is None:
                self.Yd -= self.prior_mean(Xd).reshape((-1, 1))
            else:
                self.Yd -= self.trans(self.prior_mean(Xd).reshape(-1, 1))
        self.basis = explicit_basis
        if self.basis is not None:
            (self.Nth, self.Hd) = self._basis(Xd)
        # Kernel (prior covariance)
        self.kernel = Cov
        if not isinstance(Cov, Kernel):
            raise InputError("GPR argument Cov must be a Kernel object.", Cov)
        # Do as many calculations as possible in preparation for the inference
        self.Rdd = self._radius(self.Xd, self.Xd)
        # -- the following is repeated in maximize_hyper_posterior,
        #    create a separate function? --
        if (self.kernel.Nhp > 0):
            self.maximize_hyper_posterior()
    
        self.Kdd = self.kernel(self.Rdd, data=True)
        try:
            self.LKdd = cho_factor(self.Kdd)
        except LinAlgError as e:
            print ("GPR method __init__ failed to factor data kernel." +
                   "This is often an indication that Xd has duplicates or " +
                   "the noise kernel has too small of weight.")
            raise e
        self.invKdd_Yd = cho_solve(self.LKdd, self.Yd)
        if self.basis is not None:
                self.invKdd_Hd = cho_solve(self.LKdd, self.Hd)
                LSth = cho_factor(self.Hd.T.dot(self.invKdd_Hd))
                self.Sth = cho_solve(LSth, eye(self.Nth))
                self.Th = cho_solve(LSth, self.Hd.T.dot(self.invKdd_Yd))
                self.invKdd_HdTh = cho_solve(self.LKdd, self.Hd.dot(self.Th))           
    
    def _basis(self, X):
        """Calculate the basis functions given independent variables."""
        if ( isinstance(self.basis, list) and
             sum(self.basis.count(i) for i in [0, 1, 2]) > 0 ):
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
                        H[:, j] = X[:,ix]*X[:,jx]
                        j += 1
        # TODO: add a base class or abtract interface for the basis functions.
        #     http://dirtsimple.org/2004/12/python-interfaces-are-not-java.html
        #elif isinstance(self.basis,basis_callable):
        #    H = self.basis(X)
        #    self.Nth = H.shape[1]
        else:
            # TODO: check that there is more data than degrees of freedom.
            raise InputError("GPR argument explicit_basis must be a list " +
                             "with: 0, 1, and/or 2.", self.basis)
        return (Nth, H)
    
    
    def _radius(self, X, Y):
        """Calculate the distance matrix (radius)."""
        # Previously used: cdist(X, Y, 'seuclidean',V=self.xscale),
        # which required: from scipy.spatial.distance import cdist.
        (Nx, Ny) = (X.shape[0], Y.shape[0])
        if (X.shape[1] != Y.shape[1]): Xn = 0
        else: Xn = X.shape[1]
        Rk = empty((Nx, Ny, Xn))
        for k in list(range(Xn)):
            Rk[:, :, k] = tile(X[:,[k]], (1, Ny)) - tile(Y[:,[k]].T, (Nx, 1))
            if (self.xscale == 'range' or self.xscale == 'std' or
                (isinstance(self.xscale, ndarray) and len(self.xscale) > 0)):
                Rk[:, :, k] /= self.xscale[k]
        return Rk
    
    def hyper_posterior(self, params, grad=True):
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
        (Nd, Nhp) = (self.Nd, self.kernel.Nhp)
        if not grad:
            K = self.kernel(self.Rdd, grad=False, data=True)
            lnprior = self.kernel._ln_priors(params)
        elif grad != 'Hess':
            (K, Kp) = self.kernel(self.Rdd, grad=True, data=True)
            (lnprior, dlnprior) = self.kernel._ln_priors(params, grad=True)
        else:
            (K, Kp, Kpp) = self.kernel(self.Rdd, grad='Hess', data=True)
            (lnprior, dlnprior, d2lnprior) = \
                        self.kernel._ln_priors(params, grad='Hess')
        try:
            LK = cho_factor(K)
        except LinAlgError as e:
            print ("GPR method hyper_posterior failed to factor the " +
                   "data kernel. This is most often an indication that the " +
                   "minimization routine is not converging.")
            print ('Current hyper-parameter values: ')
            print (repr(params))
            raise e
        invK_Y = cho_solve(LK, self.Yd)
        lnP_neg = ( float(self.Nd)*HLOG2PI + sum(log(diag(LK[0]))) +
                    0.5*self.Yd.T.dot(invK_Y) - lnprior)
                    
        if self.basis is not None:
            Nth = self.Nth
            invK_H = cho_solve(LK, self.Hd)
            LSth = cho_factor(self.Hd.T.dot(invK_H))
            Sth = cho_solve(LSth, eye(self.Nth))
            Th = cho_solve(LSth, self.Hd.T.dot(invK_Y))
            betaTh = invK_H.dot(Th)
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
                bKpb = invK_H.T.dot(Kp[:,:,j]).dot(invK_H)
                lnP_grad[j] -= 0.5*( sum(bKpb.T*Sth) +
                                     diff2.dot(Kp[:,:,j]).dot(betaTh) )
        if grad != 'Hess':
            return (lnP_neg, lnP_grad)
        
        # grad == 'Hess':
        (invK_Kp, aa_Kp) = (empty((Nd, Nd, Nhp)), empty((Nd, Nd, Nhp)))
        for j in xrange(Nhp):
            invK_Kp[:,:,j] = invK.dot(Kp[:,:,j])
            a_Kp = invK_Y.T.dot(Kp[:,:,j])
            aa_Kp[:,:,j] = 2.0*invK_Y.dot(a_Kp)
        lnP_hess = empty((Nhp, Nhp))
        for j in xrange(Nhp):
            for i in xrange(j + 1):
                lnP_hess[i, j] = 0.5*sum( invK_aa.T*Kpp[:,:,i,j] -
                                          invK_Kp[:,:,i].T*invK_Kp[:,:,j] +
                                          aa_Kp[:,:,i].T*invK_Kp[:,:,j]
                                          ) - d2lnprior[i,j]
                lnP_hess[j, i] = lnP_hess[i, j]
        if self.basis is not None:
            diff1 = betaTh.T - invK_Y.T
            bSb_2invK = invK_H.dot(Sth).dot(invK_H.T) - 2.0*invK
            (bKp, ThbKp) = (empty((Nth, Nd, Nhp)), empty((Nd, Nhp)))
            (diff1Kpb, diff2Kp) = (empty((Nth, Nhp)), empty((Nd, Nhp)))
            for j in xrange(Nhp):
                bKp[:,:,j] = invK_H.T.dot(Kp[:,:,j])
                ThbKp[:,j] = betaTh.T.dot(Kp[:,:,j])
                diff1Kpb[:,j] = diff1.dot(Kp[:,:,j]).dot(invK_H)
                diff2Kp[:,j] = diff2.dot(Kp[:,:,j])
            for j in xrange(Nhp):
                for i in xrange(Nhp):
                    my_mess = ( invK_H.T.dot(Kpp[:,:,i,j]).dot(invK_H) +
                                bKp[:,:,i].dot(bSb_2invK).dot(bKp[:,:,j].T) )
                    lnP_hess[i, j] -= 0.5 * ( sum(my_mess.T * Sth) +
                              diff2.dot(Kpp[:,:,i,j]).dot(betaTh) -
                              2.0*diff2Kp[:,i].dot(invK).dot(ThbKp[:,j].T) +
                              2.0*diff1Kpb[:,i].dot(Sth).dot(diff1Kpb[:,j].T) )
                    
        return (lnP_neg, lnP_grad, lnP_hess)
    
    
    def maximize_hyper_posterior(self, hyper_params=None):
        """
        Find the maximum of the hyper-parameter posterior.
        
        Arguments
        ---------
        hyper_params:  list [possibly nested] (optional),
            list of bools and functions where the nested structure corresponds 
            to the argument Cov from __init__, and each bool or function
            indicates which kernel parameters are hyper-parameters and if they
            are, what is their prior.
        """                 

        # Setup hyper-parameters & map values from a single array
        all_hyper,bounds = self.kernel._map_hyper()        
        lo,hi = [],[]
        [(lo.append(bounds[i+i]),hi.append(bounds[2*i+1])) 
            for i in xrange(len(bounds)/2)]
        
        # Perform minimization 
        MD_Newton(self.hyper_posterior, all_hyper, args=('Hess'),
                  options={'tol':1e-6, 'maxiter':200, 'bounds':(lo,hi)})
        #multi_Dimensional_Newton(self.hyper_posterior, all_hyper, args=('Hess'),
        #                         options={'tol':1e-3, 'maxiter':200})

        all_hyper,bounds = self.kernel._map_hyper(all_hyper,unmap=True)
        return self, all_hyper
    
    
    def inference(self, Xi, infer_std=False, untransform=True, data=False):
        """
        Make inferences (interpolation or regression) at specified locations.
        Limited to a single value for any hyper-parameters.
        
        Arguments
        ---------
        Xi:  array-2D,
            independent variables - where to make inferences. First
            dimension is for multiple inferences, and second dimension must
            match the second dimension of the argurment Xd from __init__.
        infer_std:  bool or 'covar' (optional),
            if True, return the inferred standard deviation;
            if 'covar', return the full posterior covariance matrix.
        untransform:  bool (optional),
            if True, any inverse transformation is applied.
        data: bool (optional),
            if True, inferred points also include the Noise contribution.
        
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
        If prior_mean was specified for GPR class object, this function
            will also be applied to Xi data.
        """
        
        # TODO: calculation of the posterior mean of gradient and Hessian.
        
        # Independent variables
        if Xi.ndim == 1:
            Xi = Xi.reshape((-1, 1))
        #Ni = Xi.shape[0]
        if Xi.ndim != 2 or Xi.shape[1] != self.Nx:
            if self.Nx == 0: pass
            else:
                raise InputError("GPR object argument Xi must be a 2D array " +
                                 "(2nd dimension must match that of Xd.)", Xi)
        
        # Mixed i-d kernel & inference of posterior mean
        Rid = self._radius(Xi, self.Xd)
        Kid = self.kernel(Rid)
        if self.basis is None:
            post_mean = Kid.dot(self.invKdd_Yd)
        else:
            post_mean = Kid.dot(self.invKdd_Yd-self.invKdd_HdTh)
            (Nth, Hi) = self._basis(Xi)
            post_mean += Hi.dot(self.Th)
        
        # Dependent variable
        if self.prior_mean is not None:
            Yi_mean = self.prior_mean(Xi).reshape((-1,1))
            if self.trans is None or not untransform:
                post_mean = resize(post_mean,shape(Yi_mean)) + Yi_mean
            else:
                post_mean = (resize(post_mean,shape(Yi_mean)) + 
                            self.trans(Yi_mean))
                
        # Inference of posterior covariance
        if infer_std:
            Rii = self._radius(Xi, Xi)
            Kii = self.kernel(Rii, data=data)
            post_covar = (Kii - 
                    resize(Kid.dot(cho_solve(self.LKdd, Kid.T)),shape(Kii)))            
            if self.basis is not None:
                A = Hi - Kid.dot(self.invKdd_Hd)
                post_covar += A.dot(self.Sth.dot(A.T))
            post_var = maximum(0.0, diag(post_covar)).reshape((-1, 1))
            post_std = sqrt(post_var)
        
        # Inverse transformation of the dependent variable
        if self.trans is not None and untransform:
            if infer_std:
                post_std = [self.trans(post_mean-post_std, inverse=True),
                            self.trans(post_mean+post_std, inverse=True)]
                post_mean = self.trans(post_mean, inverse=True)
                post_std = [post_std[0]-post_mean,
                            post_mean-post_std[1]]
            else:
                post_mean = self.trans(post_mean, inverse=True)
        
        if not infer_std:
            return post_mean
        elif infer_std == 'covar':
            return (post_mean, post_covar)
        else:
            return (post_mean, post_std)
    
    
    def sample(self, Xs, data=True):
        """
        Sample the Gassian process at specified locations.
        
        Arguments
        ---------
        Xs:  array-2D,
            independent variables - where to sample. First dimension is for
            multiple inferences, and second dimension mustmatch the second
            dimension of the argurment Xd from __init__.
        
        Returns
        -------
        Ys:  array-2D,
            sample value at each location in the argument Xs.
        
        Raises
        ------
        InputError:
            an exception is thrown for incompatible format of any inputs.
        """
        Ns = shape(Xs)[0]
        (Ys_post, Cov) = self.inference(Xs, infer_std='covar',
                                        untransform=False, data=data)
        Z = randn(Ns).reshape(Ns)
        (U,S,V) = svd(Cov)
        Ys = U.dot(diag(sqrt(S))).dot(Z)
        if size(Ys_post > 0): Ys += Ys_post.reshape(shape(Ys))
        if self.trans is not None:
            Ys = self.trans(Ys, inverse=True)
        return Ys


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

__all__ = ['GPR']

if __name__ == "__main__":
    from numpy import linspace, hstack, meshgrid, reshape
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from pyregress import *
    
    # TODO: Examples that provide verification!
    
    # Example 1:
    # Simple case, 1D with three data points and one regression point
    Xd1 = array([[0.1], [0.3], [0.6]])
    Yd1 = array([[0.0], [1.0], [0.5]])
    myGPR1 = GPR( Xd1, Yd1, SquareExp(w=Constant(0.1),
                                      l=LogNormal(guess=.3,std=.25)) )
    #myGPR1 = GPR( Xd1, Yd1, Noise(w=0.01) + RatQuad(w=Constant(.3), l=Constant(.5), alpha=1.))
    xi1 = array([[0.2]])
    yi1 = myGPR1( xi1 )
    print 'Example 1:'
    print 'x = ', xi1, ',  y = ', yi1
    
    # Example 2:
    # 2D with six data points and two regression points
    Xd2 = array([[0.00, 0.00], [0.50,-0.10], [1.00, 0.00],
                 [0.15, 0.50], [0.85, 0.50], [0.50, 0.85]])
    Yd2 = array([[0.10], [0.30], [0.60], [0.70], [0.90], [0.90]])
    myGPR2 = GPR(Xd2, Yd2, RatQuad(w=0.6, l=0.3, alpha=1.0),
                 x_scaling=False, explicit_basis=[0, 1], transform='Probit')
    #print 'Optimized value of the hyper-parameters:', param    
    xi2 = array([[0.1, 0.1], [0.5, 0.42]])
    yi2 = myGPR2( xi2 )
    print 'Example 2:'
    print 'x = ', xi2
    print 'y = ', yi2
    
    # Figures to support the examples
    # fig. example 1
    Xi1 = linspace(0.0, 0.75, 200)
    (Yi1, Yi1std) = myGPR1(Xi1, infer_std=True)
    (Yi1, Yi1std) = (Yi1.reshape(-1), Yi1std.reshape(-1))
    
    fig1 = plt.figure(figsize=(5, 3), dpi=150)
    p1, = plt.plot(Xd1, Yd1, 'ko')
    p2, = plt.plot(Xi1, Yi1, 'b-', linewidth=2.0)
    plt.fill_between(Xi1, Yi1-Yi1std, Yi1+Yi1std, alpha=0.25)
    p3 = plt.Rectangle((0.0, 0.0), 1.0, 1.0, facecolor='blue', alpha=0.25)
    p4, = plt.plot(xi1, yi1, 'ro')
    fig1.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)
    plt.title('Example 1', fontsize=16)
    plt.xlabel('Independent Variable, X', fontsize=12)
    plt.ylabel('Dependent Variable, Y', fontsize=12)
    plt.legend([p1, p2, p3, p4], ('Data', 'Inferred mean',
               'Uncertainty (one std.)', 'Example regression point'),
               numpoints=1, loc='best', prop={'size':8})

    # fig. example 2
    Ni = (30, 30)
    xi_1 = linspace(-0.2, 1.2, Ni[0])
    xi_2 = linspace(-0.2, 1.0, Ni[1])
    (Xi_1, Xi_2) = meshgrid(xi_1, xi_2, indexing='ij')
    Xi2 = hstack([Xi_1.reshape((-1, 1)), Xi_2.reshape((-1, 1))])
    (Yi2, Yi2std) = myGPR2.inference(Xi2, infer_std=True)
    
    fig = plt.figure(figsize=(7, 5), dpi=150)
    ax = fig.gca(projection='3d')
    ax.plot_surface(Xi_1, Xi_2, Yi2.reshape(Ni), alpha=0.75,
                    linewidth=0.5, cmap=mpl.cm.jet, rstride=1, cstride=1)
    ax.plot_surface(Xi_1, Xi_2, reshape(Yi2+Yi2std[0], Ni), alpha=0.25,
                    linewidth=0.25, color='black', rstride=1, cstride=1)
    ax.plot_surface(Xi_1, Xi_2, reshape(Yi2-Yi2std[1], Ni), alpha=0.25,
                    linewidth=0.25, color='black', rstride=1, cstride=1)
    ax.scatter(Xd2[:, 0], Xd2[:, 1], Yd2, c='black', s=35)
    ax.set_zlim([0.0, 1.0])
    ax.set_title('Example 2', fontsize=16)
    ax.set_xlabel('Independent Variable, X1', fontsize=12)
    ax.set_ylabel('Independent Variable, X2', fontsize=12)
    ax.set_zlabel('Dependent Variable, Y', fontsize=12)

    plt.show()
    