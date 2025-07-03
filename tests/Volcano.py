# -*- coding: utf-8 -*-
"""
Validation case 'Volcano' from 
H. Chen et al., "Analysis methods for computer 
    experiments: How to assess and what counts?" Submitted to Statistical 
    Science June 29, 2014.

Created on Mon Nov 17, 2014 @author: Ben B. Schroeder
"""

from numpy import arange, array, log10, sqrt, abs, max, zeros, ones
from numpy.random import permutation
from pyregress import GPI, Noise, SquareExp, Jeffreys


def Volcano(outputType, mean, errorType):
    '''Volcano( outputType, mean, errorType)
    
        Input: 
            outputType - type of transformation on y data ('sqrt' or 'log')
            mean - basis function type ('constant' or 'fl')
            errorType - type of error returned ('RMSE' or 'MAE')
            
        Output:
            error - array of errors (length = number of permutations (25))
    '''    
    
    
    # Volcano Data: inputs (Volume and BasalAngle) and outputs (Height or log/sqrt)
    Run = arange(1,33)
    Volume =     array([9.64, 10.2, 9.83, 10.02, 10.58, 10.77, 10.39,
                        10.95, 9.55, 10.11, 9.73, 9.92, 10.48, 10.67, 
                        10.3, 10.86, 9.36, 8.8, 9.17, 8.98, 8.42, 
                        8.23, 8.61, 8.05, 9.45, 8.89, 9.27, 9.08, 
                        8.52, 8.33, 8.7, 8.14])
    BasalAngle = array([16.48, 15.08, 13.95, 17.61, 16.2, 15.36, 14.23,
                        17.33, 12.83, 18.73, 19.86, 11.7, 13.11, 18.45,
                        19.58, 11.98, 14.52, 15.92, 17.05, 13.39, 14.8,
                        15.64, 16.77, 13.67, 18.17, 12.27, 11.14, 19.3,
                        17.89, 12.55, 11.42, 19.02])
    Height =     array([43.72, 151.16, 68.12, 105.6, 341.39, 540.2, 195.45,
                        675.34, 38.58, 123.66, 58.17, 81.92, 278.09, 476.99,
                        168.18, 606.2, 21.93, 8.64, 17.63, 13.93, 0.45,
                        0.0, 2.39, 0.06, 27.26, 14.68, 30.43, 12.95, 
                        0.08, 1.83, 9.83, 0.0])
    Data = array([Run, Volume, BasalAngle,
                  Height, log10(Height + 1), sqrt(Height)]).T
                  
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
        return sqrt(topsum / N) / sqrt(bottomsum / N)
        
    # Maximum Absolute Error
    def MAE(y_test, y_pred, y_mean):
        N = len(y_test)
        topmax = bottommax = 0
        for i in range(N):
            topmax = max(abs(y_test[i] - y_pred[i]), topmax)
            bottommax = max(abs(y_mean - y_pred[i]), bottommax)
        return topmax / bottommax
        
    error_return = zeros(25)
    for i in range(25):

        # filter data into training and testing sets
        trainData_x = Data[:25, 1:3]
        testData_x  = Data[25:, 1:3]
        if outputType == 'log':
            trainData_y = Data[:25, 4]
            testData_y  = Data[25:, 4]
            ybar = trainData_y.mean() # mean y value of training set
        if outputType == 'sqrt':
            trainData_y = Data[:25, 5]
            testData_y  = Data[25:, 5]
            ybar = trainData_y.mean() # mean y value of training set
            
        # Create GP (specify mean as basis)
        if mean == 'constant':
            mean = [0]
        if mean == 'fl':
            mean = [0, 1] # specified as conts + linear term for all x
        myKernel = Noise(w=1e-5) + SquareExp(w=Jeffreys(guess=1.0),
                                             l=[Jeffreys(guess=1.0),
                                                Jeffreys(guess=1.0)])
        myGP = GPI( trainData_x, trainData_y, myKernel, explicit_basis=mean )
        
        # Evaluate GP at test points
        y_test = myGP( testData_x )
        if errorType == 'RMSE':
            error_return[i] = RMSE(testData_y, y_test, ybar) # if RMSE desired
        if errorType == 'MAE':
            error_return[i] = MAE(testData_y, y_test, ybar) # if MAE desired
        #print 'Completed Error Calc with Mean Type -', mean 
        #print 'Error Value - ', error_return[p]
        
        # Permutate data order 
        Data = permutation(Data)
        
    return error_return
    
    
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    '''
    Specify type of error desired 
        (root mean square error 'RMSE' or mean absolute error 'MAE' )
    
    With RMSE :
        constant mean with sqrt y should give errors between 0.05-0.3
        full linear mean with sqrt y should give errors between 0.05-0.15
        constant mean with log10(y+1) should give errors between 0.1-0.6
        full linear mean with log10(y+1) should give errors between 0.05-0.4        
        
    '''
    error_type = 'RMSE'
    
    # Error Calculations based upon use of sqrt(y)
    errors_const_sr = Volcano('sqrt', 'constant', error_type)
    errors_fl_sr = Volcano('sqrt', 'fl', error_type)    
    print('{} Const. Mean with Sqrt Y Data - '.format(error_type))
    print(errors_const_sr)
    print('{} Full Linear Mean with Sqrt Y Data - '.format(error_type))
    print(errors_fl_sr)
    
    # Error Calculations based upon use of log10(y+1)
    errors_const_log = Volcano('log', 'constant', error_type)
    errors_fl_log = Volcano('log', 'fl', error_type)
    print('{} Const. Mean with log10(Y+1) Data - '.format(error_type))
    print(errors_const_log)
    print('{} Full Linear Mean with log10(Y+1) Data - '.format(error_type))
    print(errors_fl_log)
    
    # Plot
    fig = plt.figure(figsize=(6, 3))
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9, 
                        wspace=0.25)
    ax1=fig.add_subplot(121)
    plt.plot(1 * ones(25), errors_const_sr, 'o')
    plt.plot(2 * ones(25), errors_fl_sr, 'o')
    plt.errorbar(1, 0.1725, yerr=.125 )
    plt.errorbar(2, 0.15, yerr=0.1 )
    plt.xticks([1, 2], 
               ['Constant', 'Full Linear'])
    plt.xlim(0.5, 2.5)
    #plt.ylim(0, 0.3) 
    plt.ylabel('Normalized {}'.format(error_type), fontsize=10)
    plt.title(' $\sqrt{y}$', fontsize=12)
    ax2=fig.add_subplot(122)
    plt.plot(1 * ones(25), errors_const_log, 'o')
    plt.plot(2 * ones(25), errors_fl_log, 'o')
    plt.errorbar(1, 0.35, yerr=.25 )
    plt.errorbar(2, 0.225, yerr=0.175 )
    plt.xticks([1, 2], 
               ['Constant', 'Full Linear'])
    plt.xlim(0.5, 2.5)
    #plt.ylim(0, 0.6)
    plt.title('$\log_{10}(y+1)$', fontsize=12)
    plt.show()