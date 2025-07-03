from numpy import ndarray, array, linspace, atleast_1d, sin
from pyregress import GPI, Noise, SquareExp, Logarithm, Logit, Probit


def io_fixed_opts(Xd_opts, Yd_opts, K_opts, Xscale_opts, Ymean_opts, Xi_opts):
   Xscale_opts = [None,
                 'range',
                 'std'] + Xscale_opts
   Ymean_opts = [None] + Ymean_opts
   basis_opts = [None,
                 [0],
                 [0, 1],
                 [0, 1, 2],
                 [0, 1, 2, 3]]
   trans_opts = [None,
                 Logarithm,
                 Logit,
                 Probit]
   opt_opts = [True,
               False]
   terms_opts = ['noiseless',
                 'all',
                 0]
   std_opts = [False,
               True,
               'covar']  # TODO: Still needs to be implemented!
   exclude_opts = [False,
                   True]  # TODO: Figure out how to check based on combination w/ trans. & untrans.
   untrans_opts = [True,
                   False]  # TODO: Not implemented yet!
   grad_opts = [False,
               True]  # TODO: Not implemented yet!

   for xd in Xd_opts:
      for yd in Yd_opts:
         for K in K_opts:
            for sca in Xscale_opts:
               for ymean in Ymean_opts:
                  for basis in basis_opts:
                     for trans in trans_opts:
                        for opt in opt_opts:
                           myGPI = GPI(xd, yd, K, sca, ymean, basis, trans, opt)
                           for xi in Xi_opts:
                              for terms in terms_opts:
                                 for exclude in exclude_opts:
                                    yi = myGPI(xi, kernel_terms=terms, exclude_mean=exclude)
                                    print(isinstance(yi, ndarray),
                                          yi.shape[0] == atleast_1d(xi).shape[0])

# The open-ended options:
Xd = array([0.50, 2.70, 3.60, 6.80, 5.70, 3.40])
Xd_opts = [Xd,
           Xd.reshape(-1, 1)]
Yd = array([0.04, 0.78, 0.93, 0.41, 0.63, 0.90])
Yd_opts = [Yd,
           Yd.reshape(-1, 1)]
K_opts = [SquareExp(w=2.5, l=2.0),
          SquareExp(w=2.5, l=2.0) + Noise(w=0.10)]
Xscale_opts = [array([10.0])]
Ymean_opts = [lambda x: sin(x / 7)]
Xi_opts = [1.6,
           array([1.6]),
           linspace(0, 6, 5),
           linspace(0, 6, 5).reshape(-1, 1)]
io_fixed_opts(Xd_opts, Yd_opts, K_opts, Xscale_opts, Ymean_opts, Xi_opts)

# myGPI_1D = GPI(Xd_1D, Yd_1D, K_1D, transform=Probit)

# xi = 1.6  # for 1D problems w/ a single point, any of these three forms are accepted
# # xi = array([1.6])
# # xi = array([1.6]).reshape((-1, 1))

# yi = myGPI_1D(xi)
# # yi, yi_grad = myGPI_1D(xi, grad=True)

# print(yi)

# Xi = linspace(0, 7.5, 200)
# Yi, Yi_std = myGPI_1D(Xi, infer_std=True)

# # plot it all...
# import matplotlib.pyplot as plt
# plt.figure(figsize=(7, 5))
# plt.plot(Xd_1D, Yd_1D, 'ko', label='observed data')
# plt.plot(Xi, Yi, 'b-', linewidth=2.0, label='inferred mean')
# plt.fill_between(Xi, Yi-Yi_std[0], Yi+Yi_std[1], alpha=0.25, label='inferred mean +/- std')
# plt.plot(xi, yi, 'ro', label='example regression point')
# plt.xlim(Xi[0], Xi[-1])
# plt.ylim(0, 1)
# plt.title('Example #1  (1D regression w/ 6 data points & a known kernel)', fontsize=12)
# plt.xlabel('independent variable, X', fontsize=12)
# plt.ylabel('dependent variable, Y', fontsize=12)
# plt.legend(loc='lower right', fontsize=10)
# plt.show()