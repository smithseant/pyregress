# -*- coding: utf-8 -*-
"""
Multi-Dimensional Newton Solve
    multi_Dimensional_Newton(function, guess, args, options={}) 
    
  Inputs:
    function : function being minimized (function), \
        must return (f(guess,params), \
        df(guess,params), d2f(guess,params)) \n
    guess : initial guess (array) \n
    args : currently same as guess, eventually guess will be eliminated \n
    tol : minimum change in parameter space (scalar, optional)  \n
    maxit : maximum number of iterations (int, optional) \n
    positive : try to force parameters to maintain positive values      
        
  Outputs:
    history : time history of minimization (vector, optional) \n
"""

from numpy import concatenate, abs, dot, min
from scipy.linalg import inv, solve
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import line_search

def multi_Dimensional_Newton(function, guess, args, options=None):

    def convergence_crit(x, dx, iteration, options=None):
        test1 = test2 = True
        # is minimum step size met?  
        if options.has_key('tol'): 
            test1 = (abs(dx) > options['tol']).any()
        else: test1 = (abs(dx) > 1e6 )
        # it maximum number of iterations met
        if options.has_key('maxiter'):
            test2 = iteration < options['maxiter']
        else: test2 = iteration < 200
            
        # trial test to keep parameters positive
        if options.has_key('positive'):
            if (x < 0.).any():
                change = (x/abs(dx)).min()
                x += dx*change*1.01
            
        return test1 & test2      
        
    def line_search1(func,x,dx,guess,alpha=1.0,c1=0.1,c2=0.9):
        
        # Strong Wolfe conditions
        # 1) sufficient decrease
        #    f(x+alpha*dx) <= f(x) + c1*alpha*df(x).T*dx
        #    c1 - (0,1)
        def sufficent_decrease():
            return (func(x+alpha*dx)[0] <= func(x)[0] + \
                c1*alpha*dot(func(x)[1],dx))
        
        # 2) curvature condition
        #    |df(x + alpha*dx)*dx| <= c2*|(f(x)dx|
        #    c2 - (c1,1)
        def curvature_cond():
            return (abs(dot(func(x+alpha*dx)[1],dx)) <= \
                    c2*abs(dot(func(x)[1],dx)))

        iters = 0
        while not ( sufficent_decrease() & curvature_cond() ):
            if iters > 5:
                alpha = 1.
                break
            #TODO replace with better method of chosing new alpha
            # backtraking line search
            alpha *= 0.75
            iters += 1
            
        #print 'alpha ',alpha
        return alpha
        
    # 1) Initiate
    func = lambda x : function(x, guess)
    x = args
    dx = x.copy()
    history = args.copy()
    iteration = 0
        
    # 2) Test for convergence
    while convergence_crit(x, dx, iteration, options):
                        
        # 3) Compute step direction
        (f,df,d2f) = func(x)
        dx = solve(-d2f, df)
        #dx = -inv(d2f).dot(df)        
        #A = cho_factor(d2f)
        #dx = cho_solve(A,-df)
        
        # Line search
        #alpha = line_search1(func, x, dx, guess)
        
        ff = lambda x : func(x)[0]
        dff = lambda x : func(x)[1]
        alpha = line_search(ff,dff,x,dx,gfk=df,old_fval=f)
        alpha=alpha[0]
        print alpha
       # raw_input()
        
        # 4) Update estimate       
        x += alpha*dx
        iteration += 1
        history = concatenate((history, x))
        #Copies values into guess, eventaully elemenate now that pmapped handeled
        guess[:] = x
        
    print 'Number of iterations - ',iteration
    print 'Minimized parameters - ',x
    print 'Final function value - ',f
    if options.has_key('history'):
        history = history.reshape(-1,2)
        return history
    
if __name__ == "__main__":

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    import matplotlib.pyplot as plt
    from numpy import amax, amin, linspace, meshgrid, array
    from numpy import zeros_like, asarray, diag
    plt.close('all')

    #Test Function
    #---------------------------------------------------
    def Rosenbrock_func(x,*args):
        
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
    guess = array([1.3,0.7,0.8,1.9,1.2,2.])
        
    if len(guess) == 2:

        hist = multi_Dimensional_Newton(Rosenbrock_func, guess, \
                          args=guess, options={'tol':1e-6, 'maxiter':200, \
                          'history':True})        
        
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
        multi_Dimensional_Newton(Rosenbrock_func, guess, \
                        args=guess, options={'tol':1e-6, 'maxiter':600 })#, \
                        #'positive':True})