# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 09:17:08 2016

@author: bbschro
"""
__all__ = ['rprop']


from numpy import (ones, zeros, sign, inf, array, zeros_like, minimum, maximum,
                   logical_and, choose, select)
from numpy.linalg import norm


def rprop(func, params, max_iters=500, *args):
    '''rprop(func, params, *args):
    Inputs:
        func - function (must return f(params), df(params))
        params - array of inital guesses for parameter values
        *args - additional arguement that the function takes
                (function gets passed args = (a, b, ...))
    Outputs:
        params - final parameter values

    References:

    Igel, C. and Husken, M., 'Improving the Rprop Learning Algorithm',
        Proc. Second Int. Sym. Neural Comput., North Carolina pp. 115-121,
        ICSC Academic Press, 2000.

    Riedmiller, M., 'Rprop - Description and Implementation Details',
         Tech. Rep., University of Kerlsruhe, Jan. 1994
    '''

    # Initiate values
    stop_crit = 1E-4   # stopping criteria
    Delta0 = 0.1      # inital update value
    Deltamin = 1E-6   # minimum step size
    Deltamax = 50.0   # maximum step size
    eta_minus = 0.5   # decrease scale parameter
    eta_plus = 1.2    # increase scale parameter
    Delta = ones(len(params))*Delta0  # initiate parameter step
    Deltax = Delta.copy()  # changes for parameters
    if args == ():
        f, df = func(params)  # inital function evaluation
    else:
        f, df = func(params, args)
    params_init = params.copy()  # save initial parameter values
    f_init = f  # save inital function evaluation
    f_old = f.copy()
#    df_old = zeros(len(params))   # initiate last derivate array
    df_old = df.copy()   # initiate last derivate array

    # loop until criteria met
    inter = 0  # count number of iterations
    # stopping criteria: if maximum iterations reached
    for i in range(max_iters):
        positive = df*df_old > 0.0
        negative = df*df_old < 0.0
        improve = f > f_old  # check if function value improved
        Delta[positive] = minimum(Delta[positive]*eta_plus, Deltamax)
        Delta[negative] = maximum(Delta[negative]*eta_minus, Deltamin)

        df[negative] = 0.0
        temp = Deltax.copy()
        Deltax = -sign(df)*Delta
        back_track = logical_and(negative, improve.flatten())
        # Perform a backtrack step
        Deltax[back_track] = -temp[back_track]
        params += Deltax  # add step changes to parameters
        df_old = df       # save current derivative evaluation
        f_old = f         # save current function evaluation
        # updated function evaluation
        if args == ():
            f, df = func(params)
        else:
            f, df = func(params, args)
        inter += 1
        #  stopping criteria: if no curvature
        if norm(df, ord=inf) < stop_crit:
            break

    # Print results and return output
    print(inter, ' - iterations')
    print('Final params - ', params)
    print('Initial params - ', params_init)
    print('Final function val - ', f)
    print('Initial function val -', f_init)
    print('Function val change - ', f_init - f)
    return params

if __name__ == "__main__":

    def parabola(x, coef):
        a, b, c = coef[0]
        f = a*x**2 + b*x + c
        df = 2*a*x + b
        return array(f), array(df)

    def Rosenbrock_func(x, *args):
        if len(x) == 2:
            f = (1. - x[0])**2 + 100.*(x[1] - x[0]**2)**2
            dfdx = 400.*x[0]**3 - 400.*x[0]*x[1] + 2*x[0] - 2
            dfdy = 200*(x[1] - x[0]**2)
            df = array((dfdx, dfdy))
        else:
            f = sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
            xm = x[1:-1]
            xm_m1 = x[:-2]
            xm_p1 = x[2:]
            df = zeros_like(x)
            df[1:-1] = (200*(xm - xm_m1**2) - 400*(xm_p1 - xm**2)*xm -
                        2*(1 - xm))
            df[0] = -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0])
            df[-1] = 200*(x[-1] - x[-2]**2)
        return f, df

    # Test problem 1: parabola
    guess = array([2.0])
    rprop(parabola, guess, 500, (2., -1., 1.))

    # Test problem 2: 6D Rosenbrock function
    guess = array([1.3, 0.7, 0.8, 1.9, 1.2, 2.0])
    rprop(Rosenbrock_func, guess, 3000)
