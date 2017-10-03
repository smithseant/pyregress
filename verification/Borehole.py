# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 12:37:05 2014

@author: ben

! Please note that the lib pyDOE is required to run this validation case

Validation case 'Borehole' from 
H. Chen et al., "Analysis methods for computer 
    experiments: How to assess and what counts?" Submitted to Statistical 
    Science June 29, 2014.
"""

from numpy import pi, log, zeros, sqrt, abs, ones, max
from numpy.random import permutation
from pyDOE import lhs
from pyregress import *

def Borehole(mean, samples=27, permutations=1, errorType='RMSE'):
    '''Borehole( mean, samples, permutations, errprType)
    
        Input: 
            mean - basis function type ('constant' or 'fl')
            samples - number of samples in lhd
            permutations - number of permutations of lhd desired
            errorType - type of error returned ('RMSE' or 'MAE')
            
        Output:
            error - array of errors (length = number of permutations)
    '''
    
    # Given Parameter Space
    rw = [0.05,   0.15]    # radius of borehole (m)
    r  = [100.,   5000.]   # radius of influence (m)
    Tu = [63070., 115600.] # transmissivity of upper aquifer (m2/yr)
    Hu = [990.,   1110.]   # potentiometric head of upper aquifer (m)
    Tl = [63.1,   116.]    # transmissivity of lower aquifer (m2/yr)
    Hl = [700.,   820.]    # potentiometric head of lower aquifer (m)
    L  = [1120.,  1680.]   # length of borehole (m)
    Kw = [9855.,  12045.]  # hydraulic conductivity of borehole (m/yr)
    
    # Create maximized latin hypercube design for training set
    #   then scale by parameter ranges
    parameters = 8
    lhd = lhs(parameters, samples=samples, criterion='m')
    doe = zeros((samples, parameters))
    
    def scale_doe(doe, lhd):
        doe[:,0] = lhd[:,0]*(rw[1] - rw[0]) + rw[0]
        doe[:,1] = lhd[:,1]*(r[1]  - r[0])  + r[0]
        doe[:,2] = lhd[:,2]*(Tu[1] - Tu[0]) + Tu[0]
        doe[:,3] = lhd[:,3]*(Hu[1] - Hu[0]) + Hu[0]
        doe[:,4] = lhd[:,4]*(Tl[1] - Tl[0]) + Tl[0]
        doe[:,5] = lhd[:,5]*(Hl[1] - Hl[0]) + Hl[0]
        doe[:,6] = lhd[:,6]*(L[1]  - L[0])  + L[0]
        doe[:,7] = lhd[:,7]*(Kw[1] - Kw[0]) + Kw[0]
        return doe
        
    doe = scale_doe(doe, lhd)
    
    # Create maximized latin hypercube design for test set
    #   then scale by parameter ranges
    samples_test = 10000
    parameters = 8
    lhd_test = lhs(parameters, samples=samples_test, criterion='m')
    doe_test = zeros((samples_test, parameters))
    doe_test[:,0] = lhd_test[:,0]*(rw[1] - rw[0]) + rw[0]
    doe_test[:,1] = lhd_test[:,1]*(r[1]  - r[0])  + r[0]
    doe_test[:,2] = lhd_test[:,2]*(Tu[1] - Tu[0]) + Tu[0]
    doe_test[:,3] = lhd_test[:,3]*(Hu[1] - Hu[0]) + Hu[0]
    doe_test[:,4] = lhd_test[:,4]*(Tl[1] - Tl[0]) + Tl[0]
    doe_test[:,5] = lhd_test[:,5]*(Hl[1] - Hl[0]) + Hl[0]
    doe_test[:,6] = lhd_test[:,6]*(L[1]  - L[0])  + L[0]
    doe_test[:,7] = lhd_test[:,7]*(Kw[1] - Kw[0]) + Kw[0]
    
    # Function of interest
    def borehole_function(x):
        rw_, r_, Tu_, Hu_, Tl_, Hl_, L_, Kw_ = x
        numerator   = 2.*pi*Tu_*(Hu_ - Hl_)
        denomenator = log(r_/rw_)*(1. + 
                        2.*L_*Tu_/(log(r_/rw_)*rw_**2.*Kw_)+ Tu_/Tl_)
        return numerator/denomenator
    
    # Types of Error
    #   y_mean - mean output of training set
    #   y_test  - testing set
    #   y_pred  - predicted set (corresponding to same parameters as testing)
    
    # Root Mean Square Error
    def RMSE(y_test, y_pred, y_mean):
        N = len(y_test)
        topsum = bottomsum = 0.
        for i in range(N):
            topsum += (y_test[i] - y_pred[i])**2
            bottomsum += (y_mean - y_pred[i])**2
        return sqrt(topsum/N)/sqrt(bottomsum/N)
        
    # Maximum Absolute Error
    def MAE(y_test, y_pred, y_mean):
        N = len(y_test)
        topmax = bottommax = 0.
        for i in range(N):
            topmax = max(abs(y_test[i] - y_pred[i]), topmax)
            bottommax = max(abs(y_mean - y_pred[i]), bottommax)
        return topmax/bottommax
    
    # Feed test parameters into function to gain corresponding outputs 
    output_test = zeros(samples_test)
    for i in range(samples_test):
        input_test = (doe_test[i,0], doe_test[i,1], doe_test[i,2], 
                      doe_test[i,3], doe_test[i,4], doe_test[i,5],
                      doe_test[i,6], doe_test[i,7])
        output_test[i] = borehole_function(input_test)  
      
      
    # Loop over specified number of permuations of LHD
    error_return = zeros(permutations)
    for p in range(permutations): 
        
        # Feed training parameters into function to gain corresponding outputs    
        output = zeros(samples)  
        for i in range(samples):
            input_ = (doe[i,0], doe[i,1], doe[i,2], doe[i,3], 
                        doe[i,4], doe[i,5], doe[i,6], doe[i,7])
            output[i] = borehole_function(input_)

        
        ybar = output.mean() # mean y value of training set
        
        # Create GP (specify mean as basis)
        if mean == 'constant':
            mean = [0]
        if mean == 'fl':
            mean = [0, 1] # specified as conts + linear term for all x
        myKernel = SquareExp(w=Jeffreys(guess=100.), 
                             l=[Jeffreys(guess=2.0),
                                Jeffreys(guess=2.0),
                                Jeffreys(guess=2.0),
                                Jeffreys(guess=2.0),
                                Jeffreys(guess=2.0),
                                Jeffreys(guess=2.0),
                                Jeffreys(guess=2.0),
                                Jeffreys(guess=2.0)])
        myGP = GPP( lhd, output, myKernel, explicit_basis=mean )
        
        # Evaluate GP at test points
        y_test = myGP(lhd_test)
        if errorType == 'RMSE':
            error_return[p] = RMSE(output_test, y_test, ybar) # if RMSE desired
        if errorType == 'MAE':
            error_return[p] = MAE(output_test, y_test, ybar) # if MAE desired
        #print 'Completed Error Calc with Mean Type -', mean 
        #print 'Error Value - ', error_return[p]
        
        for j in range(lhd.shape[1]):
            lhd[:,j] = permutation(lhd[:,j])
        doe = scale_doe(doe, lhd)

    return error_return

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    ''' 
        Can choose to make Latin Hypercube Design (LHD) 
            with 27 or 40 pts by specifying the number of samples
        Select from error type Root Mean Square Error (RMSE) or 
            maximum absolute error (MAE)
    
        With 27 samples the normalized RMSE: 
            using the constant mean value should be between 0.0 - 0.1  
            using the full linear mean should be between    0.0 - 0.4
        With 40 samples the normalized RMSE:
            using the constant mean value should be between 0.01 - 0.04
            using the full linear mean should be between    0.01 - 0.04
        With 27 samples the normalized MAE: 
            using the constant mean value should be between 0.0 - 0.2  
            using the full linear mean should be between    0.0 - 0.5
        With 40 samples the normalized MAE:
            using the constant mean value should be between 0.0 - 0.11
            using the full linear mean should be between    0.0 - 0.125
    '''
    
    permutations = 25 # permutations of lhd collumns
    samples = 27 # use either 40 or 27
    errors_const = zeros(permutations)
    errors_fl    = zeros(permutations)
    error_type = 'MAE' # type of error ('RMSE' or 'MAE')
    # use constant mean
    errors_const = Borehole('constant', samples, permutations) 
    print('Completed Constant Permutation')
    # use full linear mean
    errors_fl     = Borehole('fl', samples, permutations) 
    print('Completed Full Linear Permutation ')

    print('%s Const. Mean - ' %error_type)
    print(errors_const)
    print('%s Full Linear Mean - ' %error_type)
    print(errors_fl)
    
    # Plot
    fig = plt.figure(figsize=(6, 3))
    plt.plot(1.0*ones(permutations), errors_const, 'o')
    plt.plot(2.0*ones(permutations), errors_fl, 'o')

    plt.xticks([1, 2], 
               ['Constant', 'Full Linear'])
    plt.xlim(0.5, 2.5)
    if samples == 27: 
        plt.ylim(0, 0.8) # for LHD-27
    if samples == 40:
        plt.ylim(0, 0.1) # for LHD-40
    plt.ylabel('Normalized %s' %error_type, fontsize=10) 
    plt.title('%i-run mLHD with Gauss Kernel' %samples, fontsize=12)
    plt.show()
    
    
########################################################################    
    
     # Evaluate GP for all combinations of leaving one data point out
    #for i in range(samples):
        #x_ = vstack((lhd[:i,:],lhd[i+1:,:]))
        #y_ = hstack((output[:i],output[i+1:]))
        #xout = lhd[i,:].reshape(1,-1) # data point left out
        #yout = output[i] # data point left out

        #myGP = GPP( x_, y_, myKernel, explicit_basis=mean )
        #y_leftout[i] = myGP(xout)