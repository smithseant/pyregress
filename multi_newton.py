# -*- coding: utf-8 -*-

__all__ = ['MD_Newton']

from numpy import abs, array, concatenate, dot, size, squeeze, diag, float32
from scipy import identity
from scipy.linalg import eigvals, solve, svd, inv, eig
from scipy.optimize import line_search

def MD_Newton(function, x, options=None):
    """
    Multi-Dimensional Newton Solve
        MD_Newton(function, guess, args,
                  options={'tol':1e-6, 'maxiter':200, 'bounds':(lo,hi)})

    Inputs:
        function : function being minimized (function), \
            must return (f(params), \
            df(params), d2f(params)) \n
        x : parameters being minimized (array) \n
    Optional Inputs(specify within an options dictionary)
        tol : minimum change in parameter space (scalar, optional)  \n
        maxit : maximum number of iterations (int, optional) \n
        bounds : parameter bounds, specify as tuple of lists \n
                 e.i ([low_1, low_2],[high_1, high_2]) \n

    Outputs:
        history : time history of minimization (vector, optional), \n
                  also specify as True in options dictionary
"""

    if 'repress text' in options:
        repressText = options['repress text']
    else:
        repressText = False

    def convergence_crit(x, dx, iteration, boundscount, options=None):
        test1 = test2 = True
        # is minimum step size met?
        if 'tol' in options:
            _tol = options['tol']
        else:
            _tol = 1e-6
        if (x != 0.0).any() or (dx != 0.0).any():
            test1 = (abs(dx/x) > _tol).any()
        elif iteration == 0:
            test1 = True
        else:
            test1 = (abs(dx) > _tol).any()
        # is maximum number of iterations met
        if 'maxiter' in options:
            test2 = iteration < options['maxiter']
        else:
            test2 = iteration < 200

        # trial test to keep parameters within bounds
        #  * amount parameters pushed within bounds is a bit arbitrary (0.001)
        if 'bounds' in options:
            low, high = options['bounds']
            if (x <= [float(i) for i in low]).any():
                change = ((x - low)/abs(dx)).min()
                boundscount[0] += 1
                x += dx*change*(1.0 + 0.001/boundscount[0])  # *
            if (x > [float(i) for i in high]).any():
                change = ((x - high)/abs(dx)).max()
                boundscount[1] += 1
                x -= dx*change*(1.0 + 0.001/boundscount[1])  # *

        # check # of times bounds have been hit
        test3 = (sum(boundscount) < 20)

        if not repressText:
            if not test1:
                print('--- min tol reached ---')
            if not test2:
                print('--- max iter reached ---')
            if not test3:
                print('--- hit bounds too many times ---')

        return test1 & test2 & test3

    def line_search_bnd(f, x, dx, alpha=1.0, c1=1e-4, c2=0.9, bounds=False):

        # Strong Wolfe conditions
        # 1) sufficient decrease
        #    f(x+alpha*dx) <= f(x) + c1*alpha*df(x).T*dx  ; c1 -> (0,1)
        def sufficent_decrease():
            return (fun_star <= fun_0 + c1*alpha*dot(dfun_0, dx))

        # 2) curvature condition
        #    |df(x + alpha*dx)*dx| <= c2*|(f(x)dx|   ;   c2 -> (c1,1)
        def curvature_cond():
            return (abs(dot(dfun_star, dx)) <= c2*abs(dot(dfun_0, dx)))

        # Check that alpha value doesn't put x outside of bounds (low and high)
        #  *  amount alpha adjusted is arbitrarily chosen (.99)
        def check_low(alpha_try):
            if (x + alpha_try*dx < [float(i) for i in low]).any():
                inv_a = abs((dx/(x-low)).min())
                return 0.99/inv_a  # *
            else:
                return alpha_try

        def check_high(alpha_try):
            if (x + alpha_try*dx > [float(i) for i in high]).any():
                inv_a = abs((dx/(x-high)).min())
                return 0.99/inv_a  # *
            else:
                return alpha_try

        # initiation of line search parameters
        low, high = bounds
        iters = 0
        alpha = check_low(alpha)
        alpha = check_high(alpha)
        fun = lambda a: f(x+a*dx, grad=False)      
        fun_star, dfun_star = f(x+alpha*dx, grad=True)
        fun_0, dfun_0 = f(x, grad=True)
        alpha_hi = 1.5
        alpha_lo = 0.75

        # loop for adjusting percent of newton step taken wrt Wolfe cond.
        while not (sufficent_decrease() & curvature_cond()):
            # only linesearch for limited # of iterations
            if iters > 10:
                alpha = 0.004
                break
            # check potential alpha values against bounds
            alpha_hi = check_low(alpha_hi)
            alpha_lo = check_low(alpha_lo)
            alpha_hi = check_high(alpha_hi)
            alpha_lo = check_high(alpha_lo)
            # adjust alpha values, either expanding up or collapsing down
            #    * amounts alpha increased and decreased is arbitrary currently
            if (fun(alpha_hi) >= fun(alpha_lo)):
                alpha = alpha_lo
                alpha_lo *= .7  # *
                alpha_hi *= .99  # *
            else:
                alpha = alpha_hi
                alpha_lo *= 1.01  # *
                alpha_hi *= 1.3  # *
            iters += 1
            fun_star, dfun_star = f(x+alpha*dx, grad=True)
        return alpha

    def is_pos_def(x):
        return all(eigvals(x) > 0.0)

    def func(x_input, grad=True):
        tempHolder = x.copy()
        x[:] = x_input
        output = function(x, grad=grad)
        x[:] = tempHolder
        return output

    # 1) Initiate
    dx = x.copy()
    history = x.copy()
    iteration = a = 0
    boundscount = array((0, 0))

    # check that bounds are in form required in tests
    if 'bounds' in options:
        low_bounds, high_bounds = options['bounds']
        if not isinstance(low_bounds, list):
            low_bounds = [low_bounds]
        if not isinstance(high_bounds, list):
            high_bounds = [high_bounds]
        for i in range(len(low_bounds)):
            if low_bounds[i] == None:
                low_bounds[i] = '-inf'
            if high_bounds[i] == None:
                high_bounds[i] = 'inf'
        options['bounds'] = (low_bounds, high_bounds)
        if not repressText:
            print('low bounds - ', low_bounds)
            print('high bounds - ', high_bounds)

    # 2) Test for convergence
    while convergence_crit(x, dx, iteration, boundscount, options):
        # 3) Compute step direction
        f, df, d2f = func(x, grad='Hess')
        if size(df) > 1 and len(d2f) == 1:
            df, d2f = squeeze(df), squeeze(d2f)  # remove empty dimensions
        if iteration == 0:
            f_init = f
        if len(d2f) == 1:
            dx = -df/d2f
        else:
            # use steepest decent if Hess matrix is not pos. def.
                # future work on forcing pos. def. needed
            if not is_pos_def(d2f):

#                U,s,V = svd(d2f)
#                a_inv = dot(dot(V.T,inv(diag(s))),U.T)
#                dx = -df.dot(a_inv)

                # new spectral decomp. conditioning method
                w, v = eig(d2f) # decompose d2f
                w = abs(w)      # convert eigenvals to positive
                delta = 0.001   # minimal eigenval tolerated
                for i in range(len(v)):
                    if w[i] < delta:  # if eigenval below tolerance
                        w[i] = delta  # set eigenval to tolerance
                # reconstruct d2f, now pos. def.
                d2f = v.dot(diag(w)).dot(v.T).real

                # Original methods of steepest decent
                # beta = abs(d2f).max() * 5. # commented out for debug
            # else:
                # new spectral decomp. conditioning method
                #d2f = d2f

                # Original methods of steepest decent
                # beta = 1e-14 # beta value arbitrarily chosen
                # d2f = ( d2f + beta*identity(len(x)) )/(1.0 + beta)
            try:
                dx = solve(d2f, -df)
            except ValueError:
                print('debug')
        dx = squeeze(dx)

        # 4) Line search
        # if bounds specified use own linesearch, else use scipy's version
        if 'bounds' in options:
            a = line_search_bnd(func, x, dx, bounds=options['bounds'])
        else:
            if len(x) > 1:
                ff = lambda x : func(squeeze(x), grad=False)
                dff = lambda x : func(squeeze(x), grad=True)[1]
            else:
                ff = lambda x : func(x, grad=False)
                dff = lambda x : func(x, grad=True)[1]
            alpha = line_search(ff, dff, x, dx, gfk=df, old_fval=f)
            a = squeeze(alpha[0])

        # 5) Update estimate
        x += a*dx
        iteration += 1
        history = concatenate((history, x))

    if not repressText:
        print('Number of iterations - ', iteration)
        print('Minimized parameters - ', x)
        print('Initial function value - ', f_init)
        print('Final function value - ', f)
        print('Total function change - ', f_init - f)
        if boundscount[0] > 0:
            print('hit low bounds ', boundscount[0], ' times')
        if boundscount[1] > 0:
            print('hit upper bounds ', boundscount[1], ' times')
    if 'history' in options:
        history = history.reshape(-1, len(x))
        return history

__all__ = ['MD_Newton']

if __name__ == "__main__":

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    import matplotlib.pyplot as plt
    from numpy import amax, amin, linspace, meshgrid
    from numpy import zeros_like, asarray, diag, cos, sin, exp, pi, log
    plt.close('all')

    # Test Functions
    # ---------------------------------------------------
    def Rosenbrock_func(x, grad=False):
        
        if len(x) == 2:
            f = (1.-x[0])**2 + 100.*(x[1]-x[0]**2)**2
            if grad == False: return f
                
            dfdx = 400.*x[0]**3 - 400.*x[0]*x[1] + 2*x[0] - 2
            dfdy = 200*(x[1]-x[0]**2)
            df = array((dfdx,dfdy))  
            if grad == True: return f,df
                
            d2fdx2 = 1200.*x[0]**2 - 400.*x[1] + 2. 
            d2fdxdy = -400.*x[0]
            d2fdy2 = 200
            H = array([[d2fdx2,d2fdxdy],[d2fdxdy,d2fdy2]])    
            if grad == 'Hess': return f,df,H

        else:    
            # High dimensional Rosenbrock from scipy optimize docs
            f = sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
            if grad == False: return f
                
            xm = x[1:-1]
            xm_m1 = x[:-2]
            xm_p1 = x[2:]
            df = zeros_like(x)
            df[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
            df[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
            df[-1] = 200*(x[-1]-x[-2]**2)
            if grad == True: return f,df       
        
            x = asarray(x)
            H = diag(-400*x[:-1],1) - diag(400*x[:-1],-1)
            diagonal = zeros_like(x)
            diagonal[0] = 1200*x[0]**2-400*x[1]+2
            diagonal[-1] = 200
            diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
            H = H + diag(diagonal)
            if grad == 'Hess': return f,df,H
        
    def test_func1(x, grad=False):
        #f(x) = (x + sin(x))*exp(-x**2)
        f = (x + sin(x))*exp(-x**2)
        if grad==False: return f
        df = exp(-x**2)*(-2.*x**2 - 2.*x*sin(x) + cos(x) + 1.)
        if grad==True: return f,df
        H = exp(-x**2) * (4.*x**2 + (4.*x**2 - 3.)*sin(x) - 6.*x - 4.*x*cos(x))
        if grad == 'Hess': return f, df, H
        
    def test_func2(x, grad=False):
        #f(x) = exp(x) - 2*x + 0.01/x - 1e-6/x**2
        f = exp(x) - 2.*x + 0.01/x - 1e-6/x**2
        if grad==False: return f
        df = exp(x) -2. - 0.01/x**2 + 2e-6/x**3
        if grad==True: return f,df
        H = exp(x) + (0.02*x - 6e-6)/x**4 
        if grad == 'Hess': return f, df, H
    
    def test_func3(x, grad=False):
        #f(x) = 3. * x**2 + 1 + (log((x-pi)**2))/pi**4
        f = 3. * x**2 + (log((x-pi)**2))/pi**4
        if grad==False: return f
        df = 6.*x + 2./(pi**4 * (x-pi))
        if grad==True: return f,df
        H = 6. - 0.020532/(pi-x)**2
        if grad == 'Hess': return f, df, H
        
    def test_func4(x, grad=False):
        #f(x) = x**4 + 2*x**2 + x + 3
        f = x**4 + 2.*x**2 + x + 3.
        if grad==False: return f
        df = 4.*x**3 + 4.*x + 1.
        if grad==True: return f,df
        H = 12.*x**2 + 4.
        if grad == 'Hess': return f, df, H
        
    def test_func5(x, grad=False):
        #f(x) = (x**2+y-11)**2 + (x+y**2-7)**2 
        f = (x[0]**2+x[1]-11.)**2 + (x[0]+x[1]**2-7.)**2
        if grad==False: return f
        df1 = 2.*(2.*x[0]*(x[0]**2+x[1]-11.)+x[0]+x[1]**2-7.)
        df2 = 2.*(x[0]**2+2.*x[1]*(x[0]+x[1]**2-7.)+x[1]-11.)
        df = array((df1,df2))
        if grad==True: return f,df
        d2f1 = 12.*x[0]**2+4.*x[1]-42.
        d2f2 = 4.*(x[0]+x[1])
        d2f4 = 4.*x[0]+12.*x[1]-26.
        H = array([[d2f1,d2f2],[d2f2,d2f4]])
        if grad == 'Hess': return f, df, H

    #---------------------------------------------------  
    #Test 1      
    print('Test function 1')
    #guess = array([.5,-.2])   
    guess = array([1.3,0.7,0.8,1.9,1.2,2.])
        
    if len(guess) == 2:

        hist = MD_Newton(Rosenbrock_func, guess,
                         options={'tol':1e-6,'maxiter':200, 'history':True})        
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
            
        plt.plot(hist[:,0],hist[:,1],Rosenbrock_func(hist.T),'b-*')
            
        x_space = linspace(amin(hist[:,0]),amax(hist[:,0]))
        y_space = linspace(amin(hist[:,1]),amax(hist[:,1]))
        X,Y = meshgrid(x_space,y_space)
        Z = Rosenbrock_func(array([X,Y]))
        surf = ax.plot_surface(X,Y,Z,rstride=2,
                cstride=2,cmap=cm.cool, 
                linewidth=0, antialiased=False)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Rosenbrock Function Minimization',fontsize=16)
        
    else:
        #high = [None]*len(guess)
        #low = [None]*len(guess)
        MD_Newton(Rosenbrock_func, guess, 
                        options={'tol':1e-6, 'maxiter':200})#, 
                        #'bounds':(low,high)})
         
    #------------------------------------------------------               
    #Test 2 
    print('Test function 2')
    x_range1 = linspace(-6.,6.)
    guess1 = array([4.])
    hist1 = MD_Newton(test_func1, guess1, options={'history':True,
                                                   'bounds':(-10.,10.)})    
    fig2 = plt.figure()
    plt.plot(x_range1,test_func1(x_range1))
    plt.plot(hist1,test_func1(hist1),'go--')
    plt.title('Test 2', fontsize=16)  
    print('Real min - ', min(test_func1(x_range1)))
    
    #Test 3
    print('Test function 3')
    x_range2 = linspace(0.1,1.)
    guess2 = array([.8])
    hist2 = MD_Newton(test_func2, guess2, options={'history':True,
                                                   'bounds':(0.,1.)})
    fig3 = plt.figure()
    plt.plot(x_range2,test_func2(x_range2))
    plt.plot(hist2,test_func2(hist2),'go--')
    plt.title('Test 3', fontsize=16)    
    print('Real min - ', min(test_func2(x_range2)))

    #Test 4
    print('Test function 4')
    x_range3 = linspace(1E-6,2.)
    guess3 = array([.6])
    hist3 = MD_Newton(test_func3, guess3, options={'history':True,
                                                   'bounds':(0.,2.)})
    fig4 = plt.figure()
    plt.plot(x_range3,test_func3(x_range3))
    plt.plot(hist3,test_func3(hist3),'go--')
    plt.title('Test 4', fontsize=16)    
    print('Real min - ', min(test_func3(x_range3)))

    #Test 5
    print('Test function 5')
    x_range4 = linspace(1E-6,2.)
    guess4 = array([1.2])
    hist4 = MD_Newton(test_func4, guess4, options={'history':True,
                                                   'bounds':(0.,2.)})
    fig5 = plt.figure()
    plt.plot(x_range4,test_func4(x_range4))
    plt.plot(hist4,test_func4(hist4),'go--')
    plt.title('Test 5', fontsize=16) 
    print('Real min - ', min(test_func4(x_range4)))

    #Test 6
    print('Test function 6')
    x1_range = linspace(-4.,4.)
    x2_range = linspace(-4.,4.)
    XX,YY = meshgrid(x1_range,x2_range)
    ZZ = test_func5(array([XX,YY]))
    guess5 = array([0.,0.])
    high = [4., 4.]
    low = [-4., -4.]
    hist5 = MD_Newton(test_func5, guess5, options={'history':True,
                                                   'bounds':(low,high)})
        
    fig6 = plt.figure()
    ax6 = fig6.gca(projection='3d')
    surf2 = ax6.plot_surface(XX, YY, ZZ, rstride=2, cstride=2, cmap=cm.cool)
    plt.plot(hist5[:,0],hist5[:,1],test_func5(hist5.T))
    print('Real min - ', ZZ.min())

    plt.show()
