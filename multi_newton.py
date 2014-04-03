# -*- coding: utf-8 -*-
"""
Multi-Dimensional Newton Solve
    multi_Dimensional_Newton(guess, params, f, df, d2f, **args) 
    
    Inputs:
        guess  - initial guess (vector) \n
        params - additional inputs to function (tuple) \n
        f      - function being minimized (function),
                 must return (f(guess,params),df(guess,params),
                 d2f(guess,params))
        tol    - minimum change in parameter space (scalar, optional) \n
        maxit  - maximum number of iterations (int, optional) \n
        
    Outputs:
        x     - minimized function input values (vector) \n
        hist  - time history of minimization (vector, optional)
"""

from numpy import concatenate, abs, dot, shape
from scipy.linalg import inv, solve
from scipy.linalg import cho_factor, cho_solve

def multi_Dimensional_Newton(guess, params, func, **args):

    def convergence_crit(x, dx,iteration,**args):
        test1 = test2 = True
        # is minimum step size met?  
        if args.has_key( "tol" ): 
            test1 = (abs(dx) > args["tol"]).any()
        else:
            test1 = (abs(dx) > 1e6 )
        # it maximum number of iterations met
        if args.has_key("maxiter"):
            test2 = iteration < args["maxiter"]
        else:
            test2 = iteration < 200
            
        # trial test to keep parameters positive
        percent = 1.
        while (x < 0.).any():
            percent += percent/2.
            dx *= (percent - 1.)
            x -= dx
            
        return test1 & test2 

    # 1) Initiate 
    x = guess.copy()
    dx = x.copy()
    iteration = 0
        
    # 2) Test for convergence
    while convergence_crit(x, dx,iteration,**args):
                        
        # 3) Compute step direction
        (f,df,d2f) = func(x)
        dx = solve(-d2f,df)
        #dx = -inv(d2f).dot(df)        
        #A = cho_factor(d2f)
        #dx = cho_solve(A,-df)
        
        
        # 4) Update Estimate
        x += dx
        iteration += 1
        guess = concatenate((guess,x))
        
    print 'Number of iterations - ',iteration
    print 'Minimized parameters - ',x
    print 'Final function value - ',f
    if args.has_key("history"):
        guess = guess.reshape(-1,2)
        return x,guess
    else:
        return x
    
if __name__ == "__main__":

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    import matplotlib.pyplot as plt
    from numpy import amax, amin, linspace, meshgrid, array
    from numpy import zeros_like, asarray, diag

    #Test Function
    #---------------------------------------------------
    def Rosenbrock_func(x):
        if len(x) == 2:
            f = (1.-x[0])**2 + 100.*(x[1]-x[0]**2)**2

            dfdx = 400.*x[0]**3 - 400.*x[0]*x[1] + 2*x[0] - 2
            dfdy = 200*(x[1]-x[0]**2)
            df = array((dfdx,dfdy))  
            
            d2fdx2 = 1200.*x[0]**2 - 400.*x[1] + 2. 
            d2fdxdy = -400.*x[0]
            d2fdy2 = 200
            H = array([[d2fdx2,d2fdxdy],[d2fdxdy,d2fdy2]])    
    
        else:    
            # High dimensional Rosenbrock from scipy optimize docs
            f = sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

            xm = x[1:-1]
            xm_m1 = x[:-2]
            xm_p1 = x[2:]
            df = zeros_like(x)
            df[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
            df[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
            df[-1] = 200*(x[-1]-x[-2]**2)
        
            x = asarray(x)
            H = diag(-400*x[:-1],1) - diag(400*x[:-1],-1)
            diagonal = zeros_like(x)
            diagonal[0] = 1200*x[0]**2-400*x[1]+2
            diagonal[-1] = 200
            diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
            H = H + diag(diagonal)
            
        return (f, df, H)

    #---------------------------------------------------
            
    #guess = array([.5,-.2])   
    guess = array([1.3,0.7,0.8,1.9,1.2])
    params = array([])
        
    if len(guess) == 2:

        (update,hist) = multi_Dimensional_Newton(guess, params, \
                         Rosenbrock_func, tol=1e-6, maxiter=200, history=True)        
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
            
        plt.plot(hist[:,0],hist[:,1],Rosenbrock_func(hist.T)[0],'b-*')
            
        x_space = linspace(amin(hist[:,0]),amax(hist[:,0]))
        y_space = linspace(amin(hist[:,1]),amax(hist[:,1]))
        X,Y = meshgrid(x_space,y_space)
        Z = Rosenbrock_func(array([X,Y]))[0]
        surf = ax.plot_surface(X,Y,Z,rstride=2,\
                cstride=2,cmap=cm.cool, \
                linewidth=0, antialiased=False)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Rosenbrock Function Minimization',fontsize=16)
        plt.show()
        
    else:
        update = multi_Dimensional_Newton(guess, params, \
             Rosenbrock_func, tol=1e-6, maxiter=200)