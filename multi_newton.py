# -*- coding: utf-8 -*-
"""
Multi-Dimensional Newton Solve
    multi_Dimensional_Newton(function, guess, args, options={}) 
    
  Inputs:
    function : function being minimized (function), \
        must return (f(guess,params), \
        df(guess,params), d2f(guess,params)) \n
    guess : initial guess for parameters being minimized (array) \n
    args : currently same as guess, eventually guess will be eliminated \n
  Optional Inputs(specify within an options dictionary)
    tol : minimum change in parameter space (scalar, optional)  \n
    maxit : maximum number of iterations (int, optional) \n
    bounds : parameter bounds, specify as tuple of lists \n
             e.i ([low_1, low_2],[high_1, high_2]) \n
        
  Outputs:
    history : time history of minimization (vector, optional), \n
              also specify as True in options dictionary
"""

from numpy import concatenate, abs, dot, squeeze, array
from scipy.linalg import inv, solve
from scipy.linalg import cho_factor, cho_solve
#from scipy.optimize import line_search, brute
from linesearch_edited import line_search_wolfe2 as line_search
from linesearch_edited import _zoom

def multi_Dimensional_Newton(function, guess, args=None, options=None):

    def convergence_crit(x, dx, iteration, boundscount, options=None):
        test1 = test2 = True
        # is minimum step size met?  
        if options.has_key('tol'): 
            test1 = (abs(dx/x) > options['tol']).any()
        else: test1 = (abs(dx/x) > 1e-6 ).any()
        # it maximum number of iterations met
        if options.has_key('maxiter'):
            test2 = iteration < options['maxiter']
        else: test2 = iteration < 200
            
        # trial test to keep parameters within bounds
        if options.has_key('bounds'):
            low, high = options['bounds']
            if not isinstance(low, list):
                low=[low]
            if not isinstance(high,list):
                high=[high]
            if (x <= [float(i) for i in low]).any():
                change = ((x-low)/abs(dx)).min()
                boundscount[0] += 1
                x += dx*change*(1.+.001/boundscount[0])
            if (x > [float(i) for i in high]).any():
                change = ((x-high)/abs(dx)).max() 
                boundscount[1] += 1
                x -= dx*change*(1.+.001/boundscount[1])
                
        test3 = (sum(boundscount) < 20)
            
        if not test1: print '--- mix tol reached ---'
        if not test2: print '--- max iter reached ---' 
        if not test3: print '--- hit bounds too many times ---'           
        
        return test1 & test2 & test3
        
    def line_search1(func,x,dx,alpha=1.0,c1=1e-4,c2=0.9,bounds=None):
        
        fun = lambda a : func(x + a*dx)[0]      
        #dfun = lambda a : func(x + a*dx)[1]        
        #alpha_H = alpha
        #alpha_L = 0. 
        fun_star,dfun_star,d2funstar = func(x+alpha*dx)
        fun_0,dfun_0,d2fun0 = func(x)
        alpha_hi = 1.5
        alpha_lo = .75
        if bounds == None:
            bounds = ['-inf']*len(x),['inf']*len(x)
        low, high = bounds

        # Strong Wolfe conditions
        # 1) sufficient decrease
        #    f(x+alpha*dx) <= f(x) + c1*alpha*df(x).T*dx  ; c1 - (0,1)
        def sufficent_decrease():
            return (fun_star <= fun_0 + 
                c1*alpha*dot(dfun_0,dx))
        
        # 2) curvature condition
        #    |df(x + alpha*dx)*dx| <= c2*|(f(x)dx|   ;   c2 - (c1,1)
        def curvature_cond():
            return (abs(dot(dfun_star,dx)) <= 
                    c2*abs(dot(dfun_0,dx))) 
        iters = 0
        while not ( sufficent_decrease() & curvature_cond() ):
            if iters > 20:
                alpha = 1.
                break
                
            if not isinstance(low, list):
                low=[low]
            if not isinstance(high,list):
                high=[high]

            def check_low(alpha_try):
                if (x + alpha_try*dx < [float(i) for i in low]).any():
                    inv_a = abs((dx/(x-low)).min())
                    return .99/inv_a
                else: return alpha_try
                
            def check_high(alpha_try):
                if (x + alpha_try*dx > [float(i) for i in high]).any():
                    inv_a = abs((dx/(x-high)).min())
                    return .99/inv_a
                else: return alpha_try

            alpha_hi = check_low(alpha_hi)
            alpha_lo = check_low(alpha_lo)
            alpha_hi = check_high(alpha_hi)
            alpha_lo = check_high(alpha_lo)

            if (fun(alpha_hi) >= fun(alpha_lo)):
                alpha = alpha_lo
                alpha_lo *= .7
                alpha_hi *= .99
            else: 
                alpha = alpha_hi
                alpha_lo *= 1.01
                alpha_hi *= 1.3
            iters += 1

            fun_star,dfun_star,d2funstar = func(x+alpha*dx)
            
        #print 'alpha ',alpha
        return alpha

        
    # 1) Initiate
    func = lambda x : function(x, args)
    x = guess
    dx = x.copy()
    history = guess.copy()
    iteration = 0
    a=0
    boundscount= array((0,0))

    # 2) Test for convergence
    while convergence_crit(x, dx, iteration, boundscount, options):

        # 3) Compute step direction
        f, df, d2f = func(x)
        if iteration == 0: f_init = f
        if len(d2f) == 1: 
            dx = -df/d2f
        else:
            dx = solve(-d2f, df)
        #dx = -inv(d2f).dot(df)        
        #A = cho_factor(d2f)
        #dx = cho_solve(A,-df)        
        
        # 4) Line search      
        #ff = lambda x : func(x)[0]
        #dff = lambda x : func(x)[1]
        if options.has_key('bounds'):# and options['bounds'][0]==0.:
        #    alpha = line_search(ff,dff,x,dx,gfk=df,old_fval=f,bounded=True)
            a = line_search1(func,x,dx,bounds=options['bounds'])
        else:
        #    alpha = line_search(ff,dff,x,dx,gfk=df,old_fval=f)
        #a = alpha[0]
            a = line_search1(func,x,dx)
        
        # 5) Update estimate   
        x += a*dx
        iteration += 1
        history = concatenate((history, x))
        
    print 'Number of iterations - ',iteration
    print 'Minimized parameters - ',x
    print 'Initial function value - ',f_init
    print 'Final function value - ',f
    print 'Total function change - ',f_init - f
    if boundscount[0] > 0: print 'hit low bounds ',boundscount[0],' times'
    if boundscount[1] > 0: print 'hit upper bounds ',boundscount[1],' times'
    if options.has_key('history'):
        history = history.reshape(-1,len(guess))
        return history
    
if __name__ == "__main__":

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    import matplotlib.pyplot as plt
    from numpy import amax, amin, linspace, meshgrid, array
    from numpy import zeros_like, asarray, diag, cos, sin, exp, pi, log
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
        return f, df, H
        
    def test_func1(x,*args):
        #f(x) = (x + sin(x))*exp(-x**2)
    
        f = (x + sin(x))*exp(-x**2)
        df = exp(-x**2)*(-2.*x**2 - 2.*x*sin(x) + cos(x) + 1.)
        H = exp(-x**2) * (4.*x**2 + (4.*x**2 - 3.)*sin(x) - 6.*x - 4.*x*cos(x))
        return f, df, H
        
    def test_func2(x,*args):
        #f(x) = exp(x) - 2*x + 0.01/x - 1e-6/x**2
        f = exp(x) - 2.*x + 0.01/x - 1e-6/x**2
        df = exp(x) -2. - 0.01/x**2 + 2e-6/x**3 
        H = exp(x) + (0.02*x - 6e-6)/x**4 
        return f, df, H
    
    def test_func3(x,*args):
        #f(x) = 3. * x**2 + 1 + (log((x-pi)**2))/pi**4
        f = 3. * x**2 + (log((x-pi)**2))/pi**4
        df = 6.*x + 2./(pi**4 * (x-pi))
        H = 6. - 0.020532/(pi-x)**2
        return f, df, H
        
    def test_func4(x,*args):
        #f(x) = x**4 + 2*x**2 + x + 3
        f = x**4 + 2.*x**2 + x + 3.
        df = 4.*x**3 + 4.*x + 1.
        H = 12.*x**2 + 4.
        return f, df, H
        
    def test_func5(x,*args):
        #f(x) = (0.2*cos(x)-1)*exp(2y)*x
        f = (0.2*cos(x[0]) - 1.)*exp(2.*x[1])*x[0]
        
        #df/dx = exp(2y)*(-0.2*x*sin(x) + 0.2*cos(x) - 1)
        df1 = exp(2.*x[1])*(-0.2*x[0]*sin(x[0]) + 0.2*cos(x[0]) -1.0)
        #df/dy = 2x*exp(2y)*(0.2*cos(x)-1)
        df2 = 2.0*x[0]*exp(2.0*x[1])*(0.2*cos(x[0]) - 1.0)
        #df = [df/dx;df/dy]
        df = array((df1,df2))
        
        #d2f/dx2 = exp(2y)*(-0.4*sin(x) -0.2x*cos(x))
        d2f1 = exp(2.*x[1])*(-0.4*sin(x[0]) - 0.2*x[0]*cos(x[0]))
        #d2f/dxdy = 2*exp(2y)*(-0.2x*sin(x) + 0.2*cos(x) - 1)
        d2f2 = exp(2.*x[1])*(-0.4*x[0]*sin(x[0]) + 0.4*cos(x[0]) - 2.)
        #d2f/dydx = 0.4*exp(2y)*(-x*sin(x) + cos(x) - 5)
        d2f3 = exp(2.*x[1])*(-0.4*x[0]*sin(x[0]) + 0.4*cos(x[0]) - 2.)
        #d2f/dy2 = 4x*exp(2y)*(0.2*cos(x) - 1)
        d2f4 = 4.*x[0]*exp(2.*x[1])*(0.2*cos(x[0]) - 1.)
        #H = [d2f/dx2 d2f/dxdy; d2f/dydx d2f/dy2]
        H = array([[d2f1,d2f2],[d2f3,d2f4]])
        
        return f, df, H

    #---------------------------------------------------  
    #Test 1      
    print 'Test function 1'
    #guess = array([.5,-.2])   
    guess = array([1.3,0.7,0.8,1.9,1.2,2.])
        
    if len(guess) == 2:

        hist = multi_Dimensional_Newton(Rosenbrock_func, guess, 
                           options={'tol':1e-6, 'maxiter':200, 
                          'history':True})        
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
            
        plt.plot(hist[:,0],hist[:,1],Rosenbrock_func(hist.T)[0],'b-*')
            
        x_space = linspace(amin(hist[:,0]),amax(hist[:,0]))
        y_space = linspace(amin(hist[:,1]),amax(hist[:,1]))
        X,Y = meshgrid(x_space,y_space)
        Z = Rosenbrock_func(array([X,Y]))[0]
        surf = ax.plot_surface(X,Y,Z,rstride=2,
                cstride=2,cmap=cm.cool, 
                linewidth=0, antialiased=False)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Rosenbrock Function Minimization',fontsize=16)
        
    else:
        high = ['inf']*len(guess)
        low = ['-inf']*len(guess)
        multi_Dimensional_Newton(Rosenbrock_func, guess, 
                        options={'tol':1e-6, 'maxiter':600, 
                        'bounds':(low,high)})
         
    #------------------------------------------------------               
    #Test 2 
    print 'Test function 2'
    x_range1 = linspace(-6.,6.)
    guess1 = array([4.])
    hist1 = multi_Dimensional_Newton(test_func1, guess1, 
                                     options={'history':True,'bounds':(-10.,10.)})    
    fig2 = plt.figure()
    plt.plot(x_range1,test_func1(x_range1)[0])
    plt.plot(hist1,test_func1(hist1)[0],'go--')
    plt.title('Test 2', fontsize=16)  
    print 'Real min - ', min(test_func1(x_range1)[0])
    
    #Test 3
    print 'Test function 3'
    x_range2 = linspace(1E-6,1.)
    guess2 = array([.8])
    hist2 = multi_Dimensional_Newton(test_func2, guess2,
                                     options={'history':True,'bounds':(0.,1.)})
    fig3 = plt.figure()
    plt.plot(x_range2,test_func2(x_range2)[0])
    plt.plot(hist2,test_func2(hist2)[0],'go--')
    plt.title('Test 3', fontsize=16)    
    print 'Real min - ', min(test_func2(x_range2)[0])

    #Test 4
    print 'Test function 4'
    x_range3 = linspace(1E-6,2.)
    guess3 = array([.6])
    hist3 = multi_Dimensional_Newton(test_func3, guess3,
                                     options={'history':True,'bounds':(0.,2.)})
    fig4 = plt.figure()
    plt.plot(x_range3,test_func3(x_range3)[0])
    plt.plot(hist3,test_func3(hist3)[0],'go--')
    plt.title('Test 4', fontsize=16)    
    print 'Real min - ', min(test_func3(x_range3)[0])

    #Test 5
    print 'Test function 5'
    x_range4 = linspace(1E-6,2.)
    guess4 = array([1.2])
    hist4 = multi_Dimensional_Newton(test_func4, guess4,
                                     options={'history':True,'bounds':(0.,2.)})
    fig5 = plt.figure()
    plt.plot(x_range4,test_func4(x_range4)[0])
    plt.plot(hist4,test_func4(hist4)[0],'go--')
    plt.title('Test 5', fontsize=16) 
    print 'Real min - ', min(test_func4(x_range4)[0])

    #Test 6
    print 'Test function 6'
    x1_range = linspace(1,10)
    x2_range = linspace(-1.5,1.5)
    XX,YY = meshgrid(x1_range,x2_range)
    ZZ = test_func5(array([XX,YY]))[0]
    guess5 = array([5.,-1.])
    high = [10, 1.5]
    low = [1, -1.5]
    hist5 = multi_Dimensional_Newton(test_func5, guess5,
                                     options={'history':True,'bounds':(low,high)})
        
    fig6 = plt.figure()
    ax6 = fig6.gca(projection='3d')
    surf2 = ax6.plot_surface(XX, YY, ZZ, rstride=2, cstride=2, cmap=cm.cool)
    plt.plot(hist5[:,0],hist5[:,1],test_func5(hist5.T)[0])
    print 'Real min - ', ZZ.min()

    plt.show()
