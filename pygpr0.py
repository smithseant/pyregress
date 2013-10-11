# -*- coding: utf-8 -*-
"""
Docstring for the pygpr module - needs work.

For basic useage see the documentation in the GPR class.
This docstring covers more advanced topics.
Performance:
  Calculation time will greatly depend on which Blas/Lapack libs are used.
  Most default python/numpy/scipy packages are based on unoptimized libs
  (including linux repositories and downloaded executables).
Beyond commonly used kernels and basis functions:
  overview of BaseKernel object and BaseBasis interface.
Reading the code and development:
  Notation used throughout the code:
    X => independent variables,
    Y => dependent variables,
    Z => transformed dependent variable,
    d => data values (observations),
    i => inferred values,
    s => sampled values,
    R2 => square distance (radius^2) in independent variable space,
    K => kernel values (covariance matrix),
    p => parameters (hyper-parameters) of the kernel,
    H => explicit basis functions evaluated at X,
    Th => Linear coefficients to the basis functions.
"""
# Created Sep 2013
# @author: Sean T. Smith

#from termcolor import colored  # may not work on windows
from numpy import (array, empty, zeros, ones, eye, shape, tile,
                   maximum, diag, trace, sum, sqrt)
from numpy.linalg.linalg import LinAlgError
from scipy import pi, log
#from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
    
from kernels import BaseKernel, Noise, OU, GammaExp, SquareExp, RatQuad
from transforms import BaseTransform, Logarithm, Probit, ProbitBeta, Logit

HLOG2PI = 0.5*log(2*pi)

class GPR:
    """
    Doctring for the GPR class - needs work.
    
    Examples
    --------
    >>> from gpr import GPR
    >>> Xd = array([[0.1],[0.3],[0.6]])
    >>> Yd = array([[0.0],[1.0],[0.5]])
    >>> myGPR = GPR(Xd, Yd, {'Noise':[0.15],'SquareExp':[2.0,0.5]})
    >>> print myGPR( array([[0.2]]) )
    [[ 0.55758742]]
    
    >>> Xd = array([[0.00, 0.00],[0.50,-0.10],[1.00, 0.00],
    ...             [0.15, 0.50],[0.85, 0.50],[0.50, 0.85]])
    >>> Yd = array([[0.10],[0.30],[0.60],[0.70],[0.90],[0.90]])
    >>> myGPR = GPR(Xd, Yd, {'RatQuad':[0.6,0.75,1.0]}, anisotropy=False,
    ...             explicit_basis=[0,1], transform='Probit')
    >>> myGPR.maximize_hyper_posterior({'RatQuad':[False, True, False]})
    >>> print myGPR(array([[0.10, 0.10],[0.50, 0.42]]))
    [[ 0.22531132]
     [ 0.77618499]]
    
    """
    def __init__(self, Xd, Yd, Kspec, anisotropy='auto',
                 Yd_mean=None, explicit_basis=None, transform=None):
        """
        Create a GPR object and prepare for inference.
        
        Arguments
        ---------
        Xd:  array-2D,
            independent-variable observed values. First dimension is for
            multiple observations, and second dimension for multiple variables.
        Yd:  array-1D [or column-shaped 2D],
            dependent-variable observed values - same length as the first
            dimension of Xd.
        Kspec:  dict or list of BaseKernel objects,
            specification of the prior covariance kernel - a composite sum
            of base-kernels. Each dict key is the name of a BaseKernel class,
            while each dict value is a list of the parameter values in order.
            Options include: Noise, OU, GammaExp, SquareExp, or RatQuad.
            Alternatively, pass a list of pre-initialized BaseKernel objects.
        anisotropy:  string or bool or array-1D (optional),
            scaling of the independent variables. For 'auto' or True, it uses
            range scaling. For False, no scaling. For an array with the same
            length as the second dimension of Xd, it uses manual scaling.
        Yd_mean:  array-1D [or column-shaped 2D] (optional),
            prior mean of the dependent variable at Xd. Must be the same
            length as Yd. If omitted, uses a prior mean of zero.
        explicit_basis:  list (optional),
            explicit basis functions are specified by any combination of the
            integers: 0, 1, 2 - each corresponding to its polynomial order.
        transform:  string or BaseClass object (optional)
            specify a dependent variable transformation with the name of a
            BaseTransform class (as a string) or a BaseTransform object.
            Options include: Logarithm, Probit, ProbitBeta, or Logit.
        """
        self.__call__ = self.inference
        
        # Independent variables
        if Xd.ndim == 1:
            self.Xd = Xd.copy().reshape((-1,1))
        elif Xd.ndim == 2:
            self.Xd = Xd
        else:
            raise InputError("GPR argument Xd must be a 2D array.", Xd)
        (self.Nd, self.Nx) = shape(Xd)
        if not anisotropy:
            self.anisotropy = ones(self.Nx)
        elif anisotropy == True or anisotropy == 'auto':
            self.anisotropy = (Xd.max(0)-Xd.min(0))**2
        elif shape(anisotropy) == (self.Nx,):
            self.anisotropy = anisotropy
        else:
            raise InputError("GPR argument anisotropy must be one of: False, "+
                             "True, 'auto', or 1D array (same length as the "+
                             "second dimension of Xd).", anisotropy)
        
        # Dependent variable
        if Yd.shape[0] != self.Nd:
            raise InputError("GPR argument Yd must have the same length as "+
                             "the first dimension of Xd.", Yd)
        self.Yd = Yd.copy().reshape((-1,1))
        if transform is None:
            self.Yd = Yd.copy().reshape((-1,1))
            self.transform = None
        elif isinstance(transform, basestring):
            self.transform = eval(transform+'(self.Yd)')
            self.Yd = self.transform(self.Yd)
        elif isinstance(transform, BaseTransform):
            self.transform = transform
            self.Yd = self.transform(self.Yd)
        else:
            raise InputError("GPR argument transform must be a BaseTransform "+
                             "class (string of name) or object.", transform)
        if Yd_mean is not None:
            if Yd_mean.shape[0] != self.Nd:
                raise InputError("GPR argument Yd_mean must have the same "+
                                 "length as first dimension of Xd.", Yd_mean)
            if self.transform is None:
                self.Yd -= Yd_mean.reshape((-1,1))
            else:
                self.Yd -= self.transform(Yd_mean.reshape(-1,1))
        self.explicit_basis = explicit_basis
        if self.explicit_basis is not None:
            (self.Ntheta, self.Hd) = self._get_basis(Xd)
        
        # Kernel (prior covariance)
        if isinstance(Kspec, list):
            self.Kernel = Kspec
        elif isinstance(Kspec, dict):
            self.Kernel = []
            for (base_kern, params) in Kspec.iteritems():
                self.Kernel += [ eval(base_kern+'(params)') ]
        else:
            raise InputError("GPR argument Kspec must be list or dict.", Kspec)
        
        # Do as many calculations as possible in preparation for the inference
        self.R2dd = self._calculate_radius2(self.Xd, self.Xd)
        # -- the following is repeated in maximize_hyper_posterior,
        #    create a separate function? --
        self.Kdd = self.calculate_kernel(self.R2dd)
        try:
            self.LKdd = cho_factor(self.Kdd)
        except LinAlgError as e:
            print ("GPR method __init__ failed to factor data kernel."+
                   "This is most often an indication that the observed data, "+
                   "Xd, has duplicates or that the noise kernel has too "+
                   "small of weight.")
            raise e
        self.invKdd_Yd = cho_solve(self.LKdd, self.Yd)
        if self.explicit_basis is not None:
            self.invKdd_Hd = cho_solve(self.LKdd, self.Hd)
            LStheta = cho_factor(self.Hd.T.dot(self.invKdd_Hd))
            self.Stheta = cho_solve(LStheta, eye(self.Ntheta))
            self.Theta = cho_solve(LStheta, self.Hd.T.dot(self.invKdd_Yd))
            self.invKdd_HdTheta = cho_solve(self.LKdd, self.Hd.dot(self.Theta))
    
    
    def _get_basis(self, X):
        """Calculate the basis functions given the independent variables."""
        if ( isinstance(self.explicit_basis, list) and
             sum(self.explicit_basis.count(i) for i in [0, 1, 2]) > 0 ):
            N = X.shape[0]
            Ntheta = sum( self.Nx**array(self.explicit_basis) )
            H = empty((N, Ntheta))
            j = 0
            if self.explicit_basis.count(0):
                H[:,j] = 1.0
                j += 1
            if self.explicit_basis.count(1):
                H[:,j:j+self.Nx] = X
                j += self.Nx
            if self.explicit_basis.count(2):
                for ix in range(self.Nx):
                    for jx in range(self.Nx):
                        H[:,j] = X[:,ix]*X[:,jx]
                        j += 1
        # TODO: add a base class or abtract interface for the basis functions.
        #     http://dirtsimple.org/2004/12/python-interfaces-are-not-java.html
        #elif isinstance(self.explicit_basis,basis_callable):
        #    H = self.explicit_basis(X)
        #    self.Ntheta = H.shape[1]
        else:
            # TODO: check that there is more data than degrees of freedom.
            raise InputError("GPR argument explicit_basis must be a list "+
                             "with: 0, 1, and/or 2.", self.explicit_basis)
        return (Ntheta, H)
    
    def _calculate_radius2(self, X, Y):
        """
        Calculate the squared distance matrix (radius) between two lists.
        """
        # Previously used: cdist(X, Y, 'seuclidean',V=self.anisotropy), which
        # required: from scipy.spatial.distance import cdist
        (Nx, Ny) = (X.shape[0], Y.shape[0])
        Rk2 = empty((Nx, Ny, self.Nx))
        for k in xrange(self.Nx):
            Rk2[:,:,k] = ( tile(X[k,:].T**2, (1,Ny)) - 2.0*X[k,:].T.dot(Y[k,:])
                          +tile(Y[k,:]**2, (Nx,1)) )
            Rk2[:,:,k] *= self.anisotropy[k]
        return Rk2
    
    def calculate_kernel(self, R2, no_noise=False, grad=False):
        """
        Kernel (prior covariance matrix) function of a radius matrix.
        
        Arguments
        ---------
        R2:  array-3D,
            directional square distance matrix (radius^2) between two arrays
            of points.
        no_noise:  bool (optional),
            when no_noise is true, exclude Noise BaseKernels from calculation.
        grad:  bool (optional),
            when grad is not False, return gradK.
        
        Returns
        -------
        K:  array-2D,
            kernel values - shape matches first two dimensions of argument R.
        gradK:  list of arrays (optional - depending on argument grad),
            partial derivative of kernel with respect to each hyper parameter.
        """
        K = zeros(R2.shape[:2])
        gradK = []
        kernel_no_noise = [base_kern for base_kern in self.Kernel
                           if not (no_noise and isinstance(base_kern, Noise))]
        for base_kern in kernel_no_noise:
            if not grad or base_kern.Nhyper==0:
                K += base_kern(R2)
            else:
                (K_tmp, gradK_tmp) = base_kern(R2, grad=True)
                K += K_tmp
                gradK += gradK_tmp
        if not grad:
            return K
        else:
            return (K, gradK)
    
    def hyper_posterior(self, params, p_mapped, grad=True):
        """
        Negative log of the hyper-parameter posterior & its gradient.
        
        Arguments
        ---------
        params:  array-1D,
            hyper parameters in an array for the minimization routine.
        p_mapped:  array-1D,
            hyper parameter values that map to self.Kernel.
        grad:  bool (optional),
            when grad is not False, must return lnP_grad.
        
        Returns
        -------
        lnP_neg:  float,
            negative log of the hyper-parameter posterior.
        lnP_grad:  array-1D (optional - depending on argument grad),
            gradient of lnP_neg with respect to each hyper-parameter.
        """
        p_mapped[:] = params
        (K, Kprime) = self.calculate_kernel(self.Rdd, grad=True)
        try:
            LK = cho_factor(K)
        except LinAlgError as e:
            print ("GPR method hyper_posterior failed to factor the "+
                   "data kernel. This is most often an indication that the "+
                   "minimization routine is not converging.")
            print ('Current hyper-parameter values: ')
            print (repr(params))
            raise e
        invK_Y = cho_solve(LK, self.Yd)
        if self.explicit_basis is not None:
            invK_H = cho_solve(LK, self.Hd)
            LSth = cho_factor(self.Hd.T.dot(invK_H))
            Sth = cho_solve(LSth, eye(self.Ntheta))
            Th = cho_solve(LSth, self.Hd.T.dot(invK_Y))
            betaTh = invK_H.dot(Th)
        
        lnP_neg = ( float(self.Nd)*HLOG2PI + sum(log(diag(LK[0]))) +
                    0.5*self.Yd.T.dot(invK_Y) )  # -- subtract prior --
        if self.explicit_basis is not None:
            lnP_neg -= ( float(self.Ntheta)*HLOG2PI - sum(log(diag(LSth[0]))) +
                         0.5*Th.T.dot(self.Hd.T.dot(betaTh)) )
        
        if not grad:
            return lnP_neg
        else:
            lnP_grad = empty(shape(params))
            for j in range(lnP_grad.shape[0]):
                lnP_grad[j] = 0.5*( trace(cho_solve(LK,Kprime[j])) -
                                    invK_Y.T.dot(Kprime[j].dot(invK_Y)) )
                if self.explicit_basis is not None:
                    bKp = invK_H.T.dot(Kprime[j])
                    lnP_grad[j] -= ( 0.5*(trace(bKp.dot(invK_H).dot(Sth))) +
                                     Th.T.dot(bKp.dot(0.5*betaTh-invK_Y)) )
            return (lnP_neg, lnP_grad)
    
    
    def maximize_hyper_posterior(self, hyper_params=None):
        """
        Find the maximum of the hyper-parameter posterior.
        
        Arguments
        ---------
        hyper_params:  dict (optional),
            each dict key is a BaseKernel class name corresponding to
            an entry in the argument Kspec of __init__, and each value
            is a list of bools indicating which are hyper parameters.
        """
        
        # Setup hyper-parameters in the BaseKernels
        if hyper_params:
            Nhyper = 0
            for (input_kern, hyper_bool) in hyper_params.iteritems():
                for base_kern in self.Kernel:
                    if isinstance(base_kern, eval(input_kern)):
                        Nhyper += base_kern.declare_hyper(hyper_bool)
        else:
            Nhyper = sum(kern.Nhyper for kern in self.Kernel)
        
        # Map the hyper parameters to a single array (for minimization)
        all_hyper = empty(Nhyper)
        i = 0
        for kern in self.Kernel:
            kern.map_hyper(all_hyper[i:i+kern.Nhyper])
            i += kern.Nhyper
        
        # Perform minimization
        myResult = minimize(self.hyper_posterior, all_hyper,
                            args=(all_hyper,False), method='Nelder-Mead',
                            tol=1e-4, options={'maxiter':200, 'disp':True})
#        myResult = minimize(self.hyper_posterior, all_hyper,
#                            args=(all_hyper,), method='BFGS', jac=True,
#                            tol=1e-4, options={'maxiter':200, 'disp':True})
#        myResult = minimize(self.hyper_posterior, all_hyper,
#                            args=(all_hyper,), method='L-BFGS-B', jac=True,
#                            bounds=[(0.0,None)]*Nhyper, tol=1e-4,
#                            options={'maxiter':200, 'disp':True})
        
        #copy values back to Kernel (remove mapping)
        i = 0
        for kern in self.Kernel:
            kern.map_hyper(myResult.x[i:i+kern.Nhyper], unmap=True)
            i += kern.Nhyper
        
        # Do as many calculations as possible in preparation for the inference
        self.Kdd = self.calculate_kernel(self.Rdd)
        self.LKdd = cho_factor(self.Kdd)
        self.invKdd_Yd = cho_solve(self.LKdd, self.Yd)
        if self.explicit_basis is not None:
            self.invKdd_Hd = cho_solve(self.LKdd, self.Hd)
            LStheta = cho_factor(self.Hd.T.dot(self.invKdd_Hd))
            self.Stheta = cho_solve(LStheta, eye(self.Ntheta))
            self.Theta = cho_solve(LStheta, self.Hd.T.dot(self.invKdd_Yd))
            self.invKdd_HdTheta = cho_solve(self.LKdd, self.Hd.dot(self.Theta))
        return self
    
    
    def inference(self, Xi, Yi_mean=None, infer_std=False, no_noise=True):
        """
        Make inferences (interpolation or regression) at specified locations.
        
        Arguments
        ---------
        Xi:  array-2D,
            independent variables - where to make inferences. First
            dimension is for multiple inferences, and second dimension must
            match the second dimension of the argurment Xd in __init__.
        Yi_mean:  array-1D [or column-shaped 2D] (optional),
            prior mean of the inferences. Must be the same length as Yi_mean.
            If omitted, uses a prior mean of zero.
        infer_std:  bool (optional),
            if True, return the inferred standard deviation.
        no_noise:  bool (optional),
            if True, make inferences without contribution of the noise kernel.
        
        Returns
        -------
        post_mean:  array-2D,
            inferred mean at each location in the argument Xi.
        post_std: array-2D or list (optional - depending on infer_std),
            inferred standard deviation of the inferrences
            (if a variable transformation is used, both the positive and
            the negative standard deviations are returned - in that order).
        """
        
        # Independent variables
        if Xi.ndim == 1:
            Xi = Xi.copy().reshape((-1,1))
        Ni = Xi.shape[0]
        if Xi.ndim != 2 or Xi.shape[1] != self.Nx:
            raise InputError("myGPR argument Xi must be a 2D array (second "+
                             "dimension length must match that of Xd.", Xi)
        
        # Mixed i-d kernel & inference of posterior mean
        #Rid = cdist(Xi,self.Xd, 'seuclidean',V=self.anisotropy)
        R2id = self._calculate_radius2(Xi, self.Xd)
        Kid = self.calculate_kernel(R2id, no_noise=no_noise)
        if self.explicit_basis is None:
            post_mean = Kid.dot(self.invKdd_Yd)
        else:
            post_mean = Kid.dot(self.invKdd_Yd-self.invKdd_HdTheta)
            (Ntheta, Hi) = self._get_basis(Xi)
            post_mean += Hi.dot(self.Theta)
        
        # Dependent variable
        if Yi_mean is not None:
            if Yi_mean.shape[0] != Ni:
                raise InputError("myGPR argument Yi_mean must have the same "+
                                 "length as first dimension of Xi.", Yi_mean)
            Yi_mean= Yi_mean.copy().reshape((-1,1))
            if self.transform is None:
                post_mean += Yi_mean
            else:
                post_mean += self.transform(Yi_mean)
        
        # Inference of posterior covariance
        if infer_std:
            #Rii = cdist(Xi,Xi, 'seuclidean',V=self.anisotropy)
            R2ii = self._calculate_radius2(Xi, Xi)
            Kii = self.calculate_kernel(R2ii, no_noise=no_noise)
            post_covar = Kii-Kid.dot(cho_solve(self.LKdd, Kid.T))
            if self.explicit_basis is not None:
                A = Hi - Kid.dot(self.invKdd_Hd)
                post_covar += A.dot(self.Stheta.dot(A.T))
            post_var = maximum(0.0, diag(post_covar)).reshape((-1,1))
            post_std = sqrt(post_var)
        # -- option to return the full posterior covariance (would be needed
        #    to sample a processes from the posterior)? --
        
        # Inverse transformation of the dependent variable
        if self.transform is not None:
            if infer_std:
                post_std = [self.transform(post_mean-post_std, inverse=True),
                            self.transform(post_mean+post_std, inverse=True)]
                post_mean = self.transform(post_mean, inverse=True)
                post_std = [post_std[0]-post_mean,
                            post_mean-post_std[1]]
            else:
                post_mean = self.transform(post_mean, inverse=True)
        
        if not infer_std:
            return post_mean
        else:
            return (post_mean, post_std)


class Error(Exception):
    """Base class for exceptions in the gpr module."""
    pass

class InputError(Error):  # -- not a ValueError? --
    """Exception raised for errors in method input arguments.
    
    Arguments
    ---------
        msg:  string,
            explanation of the error.
        input_argument:  any (optional),
            input argument that is the source of the error. Provided so the
            value of that variable can be reported when the error is caught.
    """
    def __init__(self, msg, input_argument=None):
        self.args = (msg,)
        self.input_argument = input_argument


if __name__ == "__main__":
    from numpy import linspace, hstack, meshgrid, reshape
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Example 1:
    # Simple case, 1D with three data points and one regression point
    Xd1 = array([[0.1],[0.3],[0.6]])
    Yd1 = array([[0.0],[1.0],[0.5]])
    myGPR1 = GPR(Xd1, Yd1, {'Noise':[0.15],'SquareExp':[2.0,0.5]})
    Xi1_ex = array([[0.2]])
    Yi1_ex = myGPR1( Xi1_ex )
    
    # Example 2:
    # 2D with six data points and two regression points
    Xd2 = array([[0.0,0.0],[0.5,-0.1],[1.0,0.0],
                 [0.15,0.5],[0.85,0.5],[0.5,0.85]])
    Yd2 = array([[0.1],[0.3],[0.6],[0.7],[0.9],[0.9]])
    myGPR2 = GPR(Xd2, Yd2, {'RatQuad':[0.6,0.75,1.0]}, anisotropy=False,
                 explicit_basis=[0,1], transform='Probit')
    myGPR2.maximize_hyper_posterior({'RatQuad':[False, True, False]})
    Xi2_ex = array([[0.1,0.1],[0.5,0.42]])
    Yi2_ex = myGPR2(Xi2_ex)
    
    # Figures to support the examples
    # example 1
    Xi1 = linspace(0.0, 0.75)
    (Yi1, Yi1_std) = myGPR1(Xi1, infer_std=True)
    (Yi1, Yi1_std) = (Yi1.reshape(-1), Yi1_std.reshape(-1))
    
    fig1 = plt.figure(figsize=(5,3), dpi=150)
    p1, = plt.plot(Xd1,Yd1, 'ko')
    p2, = plt.plot(Xi1,Yi1, 'b-', linewidth=2.0)
    plt.fill_between(Xi1, Yi1-Yi1_std, Yi1+Yi1_std, alpha=0.25)    
    p3 = plt.Rectangle((0.0, 0.0), 1.0, 1.0, facecolor='blue', alpha=0.25)
    p4, = plt.plot(Xi1_ex,Yi1_ex, 'ro')
    fig1.subplots_adjust(left=0.15, right=0.95, bottom=0.15,top=0.9)
    plt.title('Example 1', fontsize=16)
    plt.xlabel('Independent Variable, X', fontsize=12)
    plt.ylabel('Dependent Variable, Y', fontsize=12)
    plt.legend([p1,p2,p3,p4], ('Data', 'Inferred mean',
               'Uncertainty (one std.)', 'Example regression point'),
               numpoints=1, loc='best', prop={'size':8})

    # example 2
    Ni = (30,30)
    xi_1 = linspace(-0.2,1.2,Ni[0])
    xi_2 = linspace(-0.2,1.0,Ni[1])
    (Xi_1,Xi_2) = meshgrid(xi_1,xi_2, indexing='ij')
    Xi2 = hstack([Xi_1.reshape((-1,1)), Xi_2.reshape((-1,1))])
    (Yi2, Yi2_std) = myGPR2.inference(Xi2, infer_std=True)
    
    fig = plt.figure(figsize=(7,5), dpi=150)
    ax = fig.gca(projection='3d')
    ax.plot_surface(Xi_1,Xi_2, Yi2.reshape(Ni), alpha=0.75,
                    linewidth=0.5,cmap=mpl.cm.jet, rstride=1,cstride=1)
    ax.plot_surface(Xi_1,Xi_2, reshape(Yi2+Yi2_std[0], Ni), alpha=0.25,
                    linewidth=0.25,color='black', rstride=1,cstride=1)
    ax.plot_surface(Xi_1,Xi_2, reshape(Yi2-Yi2_std[1], Ni), alpha=0.25,
                    linewidth=0.25,color='black', rstride=1,cstride=1)
    ax.scatter(Xd2[:,0],Xd2[:,1],Yd2, c='black', s=50)
    ax.set_zlim([0.0,1.0])
    ax.set_title('Example 2', fontsize=16)
    ax.set_xlabel('Independent Variable, X1', fontsize=12)
    ax.set_ylabel('Independent Variable, X2', fontsize=12)
    ax.set_zlabel('Dependent Variable, Y', fontsize=12)

    plt.show()