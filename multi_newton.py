# -*- coding: utf-8 -*-
"""
Multi Dimensional Newton Solve
    Inputs:
       x - function inputs (vector)
       guess - initial guess (same size as x)
       f - function being minimized (function)
       df - function derivatives (function)
       d2f - function Hessian (function)
    (optional)
       tol - minimum change in parameter space (scalar)
       maxit - maximum number of iterations (int)
     Outputs:
       x - minimized function input values (vector)
       hist - array 
"""

from numpy import linspace, meshgrid, array, concatenate
from numpy import amax, amin, abs, dot, empty
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.linalg import inv, solve


def multi_Dimensional_Newton(x,guess,f,df,d2f,**args):

    # Initiate 
    x = guess.copy()
    dx = x.copy()
    iteration = 0
    
    def convergence_crit(dx,iteration,**args):
        test1 = test2 = True
        # is minimum step size met?  
        if args.has_key( "tol" ): 
            test1 = (abs(dx) > args["tol"]).any()
        # it maximum number of iterations met
        if args.has_key("maxiter"):
            test2 = iteration < args["maxiter"]
        return test1 & test2    
    
    # 1) Test for convergence
    while convergence_crit(dx,iteration,**args):
        
        # 2) Compute step direction
        dx = solve(-d2f(x),df(x))
        #dx = dot(-inv(d2f(x)),df(x))
        
        # 3) Update Estimate
        x += dx
        iteration += 1
        guess = concatenate((guess,x))
    
    guess = guess.reshape(-1,2)
    print 'Number of iterations - ',iteration
    print 'Final function value - ',f(x)
    return x,guess
    
if __name__ == "__main__":

    #Test Function
    #---------------------------------------------------
    def Rosenbrock_func(x):
        return (1.-x[0])**2 + 100.*(x[1]-x[0]**2)**2
    
    def d_Rosenbrock_func(x):
        dfdx = 400.*x[0]**3 - 400.*x[0]*x[1] + 2*x[0] - 2
        dfdy = 200*(x[1]-x[0]**2)
        return array((dfdx,dfdy))
        
    def Hess_Rosenbrock_func(x):
        d2fdx2 = 1200.*x[0]**2 - 400.*x[1] + 2. 
        d2fdxdy = -400.*x[0]
        d2fdy2 = 200
        return array([[d2fdx2,d2fdxdy],[d2fdxdy,d2fdy2]])
    #---------------------------------------------------
            
    func_inputs = empty(2)
    guess = array(([.5,-.2]))    
    
    (update,hist) = multi_Dimensional_Newton(func_inputs,guess, \
             Rosenbrock_func, d_Rosenbrock_func, Hess_Rosenbrock_func, \
             tol=1e-6, maxiter=200)
        
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    plt.plot(hist[:,0],hist[:,1],Rosenbrock_func(hist.T),'b-*')
    
    x_space = linspace(amin(hist[:,0]),amax(hist[:,0]))
    y_space = linspace(amin(hist[:,1]),amax(hist[:,1]))
    X,Y = meshgrid(x_space,y_space)
    Z = Rosenbrock_func(array([X,Y]))
    surf = ax.plot_surface(X,Y,Z,rstride=2,\
            cstride=2,cmap=cm.cool, \
            linewidth=0, antialiased=False)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Rosenbrock Function Minimization',fontsize=16)
    plt.show()