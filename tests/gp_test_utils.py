from warnings import warn

from numpy import (array, empty, full, linspace, arange, atleast_2d, meshgrid, delete,
                   sqrt, exp, log, sin, cos, pi as π)
from numpy.linalg import eigh, solve
from numpy.random import default_rng
from scipy.special import erf, erfinv
from scipy.spatial.distance import cdist
from scipy.stats.qmc import Sobol

from pyregress import GPI, Noise, SquareExp, Logit, Probit, InputError
from pyregress.gaussian_processes.gaussian_process import (radius, warn_trans_μ, warn_trans_σ,
                                        error_trans_exclude, error_trans_covar, error_trans_grad)

# Setup of Examples:
def example_1D(seed=42, nd=8, ni=300, plot=False):
    my_rng = default_rng(seed=seed)
    std_norm = my_rng.standard_normal
    def transformed_mean(x, grad=False):
        y = log(1 / (1 / exp(x - 4) + 1 / exp(0.1 * x - 0.4))) + 2 * exp(-x / 2)
        if not grad:
            return y
        else:
            grad_y = ((exp(0.1 * x - 0.4) + 0.1 * exp(x - 4)) /
                      (exp(0.1 * x - 0.4) +  1  * exp(x - 4)) -
                      exp(-x / 2))
            return y, grad_y
    def prior_mean(x, grad=False):
        if not grad:
            return (1 + erf(transformed_mean(x) / sqrt(2))) / 2
        else:
            z, grad_z = transformed_mean(x, grad)
            y = (1 + erf(z / sqrt(2))) / 2
            grad_y = exp(-z**2 / 2) / sqrt(2 * π) * grad_z
            return y, grad_y
    def poly(x, x0=10, y0=-0.6, yi=0.4, grad=False):
        y = yi - 2 * (yi - y0) / x0 * x + (yi - y0) / x0**2 * x**2
        if not grad:
            return y
        else:
            grad_y = -2 * (yi - y0) / x0 + 2 * (yi - y0) / x0**2 * x
            return y, grad_y
    Xd = linspace(0, 10, nd).reshape((-1, 1))
    Xp = linspace(0, 20, ni).reshape((-1, 1))
    μZd = transformed_mean(Xd)
    μZp = transformed_mean(Xp)
    Hdβ = poly(Xd)
    Hpβ = poly(Xp)
    Xboth = array([el for el in Xd] + [el for el in Xp]).reshape((-1, 1))
    σd, w, ℓ = 0.02, 0.1, array([3])
    Kboth = w**2 * exp(-cdist(Xboth, Xboth, 'seuclidean', V=ℓ**2)**2 / 2)
    Λ, V = eigh(Kboth)
    Λ = Λ.clip(min=0)
    gp_both = (V * sqrt(Λ)) @ std_norm((Xboth.shape[0], 1))
    gp_d = gp_both[:nd]
    noise_d  = σd * std_norm((nd, 1))
    Zd = μZd + Hdβ + gp_d + noise_d
    Yd = (1 + erf(Zd / sqrt(2))) / 2

    if plot:
        from matplotlib import pyplot as plt
        gp_p = gp_both[nd:]
        Zp = μZp + Hpβ + gp_p
        Yp = (1 + erf(Zp / sqrt(2))) / 2
        myGPI = GPI(Xd, Yd, Noise(w=σd) + SquareExp(w=w, l=ℓ),
                    Ymean=prior_mean, explicit_basis=[0, 1, 2], transform=Probit())
        Yp_post, σp_post = myGPI(Xp, infer_std=True, untransform=True)

        plt.subplots(2, 1, figsize=(8, 11))
        plt.subplots_adjust(hspace=0)

        plt.subplot(2, 1, 1)
        plt.plot(Xp, μZp, linewidth=0.5, label='prior mean')
        plt.plot(Xp, Hpβ, linewidth=0.5, label='underlying bases')
        plt.plot(Xp, gp_p, linewidth=0.5, color='tab:cyan', label='GP')
        plt.plot(Xd, noise_d, marker='o', markersize=3, linestyle='none',
                color='tab:red', label='noise (iid)')
        plt.plot(Xp, Zp, linewidth=0.5, color='black', label='reality')
        plt.plot(Xd, Zd, marker='o', markersize=3, linestyle='none',
                 color='black', label='observed data')
        plt.xlim(Xp[0], Xp[-1])
        plt.grid(True, alpha=0.2)
        for lab in plt.gca().axes.get_xticklabels():
            lab.set_visible(False)
        plt.ylabel('transformed variable, Z', fontsize=12)
        plt.legend(loc='lower right', fontsize=10) ;

        plt.subplot(2, 1, 2)
        plt.plot(Xp, Yp, linewidth=0.5, color='black', label='reality')
        plt.plot(Xd, Yd, marker='o', markersize=3, linestyle='none',
                 color='black', label='observed data')
        plt.plot(Xp, Yp_post, color='tab:green', label='inferred mean')
        plt.fill_between(Xp[:, 0], Yp_post - 2 * σp_post[0], Yp_post + 2 * σp_post[1],
                         color='tab:green', alpha=0.25, label='inferred mean +/- 2σ')
        plt.xlim(Xp[0], Xp[-1])
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.2)
        plt.xlabel('independent variable, X', fontsize=12)
        plt.ylabel('dependent variable, Y', fontsize=12)
        plt.legend(loc='lower right', fontsize=10)

        plt.savefig('example_1D.pdf')

    return Xd, Yd, Xp, {'σd':σd, 'w':w, 'ℓ':ℓ}, prior_mean


class MultiDimFunc:
    def __init__(self, a, λ, φ):
        self.a = a
        self.λ = λ.reshape((1, -1))
        self.φ = φ.reshape((1, -1))
    def __call__(self, x, grad=False):
        xm = atleast_2d(x)
        y = self.a / 2 * (cos(2 * π * (self.λ * xm - self.φ)).prod(axis=1).reshape((-1, 1)) + 1)
        if not grad:
            return y
        else:
            grad_y = empty(xm.shape)
            for i in range(xm.shape[1]):
                grad_y[:, i] = (-self.a * π * self.λ[0, i] *
                                sin(2 * π * (self.λ[0, i] * xm[:, i] - self.φ[0, i])))
                grad_y[:, i] *= cos(2 * π * delete(self.λ * xm - self.φ, i, 1)).prod(axis=1)
            return y, grad_y

def example_nD(seed=42, dims=2, ni=(34, 35), log2_n_pts=3, plot=False):
    my_rng = default_rng(seed=seed)
    rand = my_rng.random
    std_norm = my_rng.standard_normal
    laplace = my_rng.laplace
    ln_norm = my_rng.lognormal
    nd = 2 ** log2_n_pts
    my_sobol_sampler = Sobol(d=dims, scramble=True, optimization='random-cd')
    Xd = my_sobol_sampler.random_base2(log2_n_pts)  # Constrain the support to the unit box for now.
    X1p, X2p = meshgrid(linspace(0, 1, ni[0]), linspace(0, 1, ni[1]), indexing='ij')
    Xp = array([X1p.reshape(-1), X2p.reshape(-1)]).T
    my_yprior = MultiDimFunc(0.4 * rand(1)[0], 2 / 3 * rand(dims), rand(dims))
    f_meand = my_yprior(Xd)
    f_meanp = my_yprior(Xp)
    μZd = log(f_meand / (1 - f_meand)).reshape((-1, 1))
    μZp = log(f_meanp / (1 - f_meanp)).reshape((-1, 1))
    Hd = empty((nd, (1 + dims)))
    Hd[:, 0] = 1
    Hd[:, 1:] = Xd
    β = 0.2 * laplace(size=dims + 1)
    Hdβ = (Hd @ β).reshape((-1, 1))
    if plot and dims == 2:
        Hp = empty((X1p.size, (1 + dims)))
        Hp[:, 0] = 1
        Hp[:, 1:] = Xp
        Hpβ = (Hp @ β).reshape((-1, 1))
        iter_both_dims = zip(X1p.reshape(-1), X2p.reshape(-1))
        Xboth = array([el for el in Xd] + [[el1, el2] for el1, el2 in iter_both_dims])
    else:
        Xboth = Xd
    σd, w, ℓ = 0.02, 0.1, ln_norm(-1.2, 0.5, dims)
    Kboth = w**2 * exp(-cdist(Xboth, Xboth, 'seuclidean', V=ℓ**2)**2 / 2)  # Consider using rational quadratic
    Λ, V = eigh(Kboth)
    Λ = Λ.clip(min=0)
    gp_both = (V * sqrt(Λ)) @ std_norm((Xboth.shape[0], 1))
    gp_d = gp_both[:nd]
    noise_d  = σd * std_norm((nd, 1))
    Zd = μZd + Hdβ + gp_d + noise_d
    Yd = 1 / (exp(-Zd) + 1)  # Consider using a different transformation
    if plot and dims == 2:
        gp_p = gp_both[nd:]
        Zp = (μZp + Hpβ + gp_p).reshape(X1p.shape)
        Yp = 1 / (exp(-Zp) + 1)
        myGPI = GPI(Xd, Yd, Noise(w=σd) + SquareExp(w=w, l=ℓ),
                    explicit_basis=[0, 1], transform=Logit())
        Yp_post, σp_post = myGPI(Xp, infer_std=True, untransform=True)
        Yp_post = Yp_post.reshape(X1p.shape)
        σp_post = σp_post[0].reshape(X1p.shape), σp_post[1].reshape(X1p.shape)

        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(8, 11))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(X1p, X2p, Yp, color='black', alpha=0.4, label='reality')
        ax.scatter(Xd[:, 0], Xd[:, 1], Yd, color='black', label='observed data')
        ax.plot_surface(X1p, X2p, Yp_post, color='tab:green', alpha=0.4, label='inferred mean')
        ax.plot_surface(X1p, X2p, Yp_post - 2 * σp_post[0],
                         color='tab:green', alpha=0.15, label='inferred mean +/- 2σ')
        ax.plot_surface(X1p, X2p, Yp_post + 2 * σp_post[1], color='tab:green', alpha=0.15)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('first indep. var., $X_1$', fontsize=12)
        ax.set_ylabel('second indep. var., $X_2$', fontsize=12)
        ax.set_zlabel('dependent variable, $Y$', fontsize=12)
        plt.legend(fontsize=10)

        plt.savefig('example_2D.pdf')

    return Xd, Yd, Xp, {'σd':σd, 'w':w, 'ℓ':ℓ}, my_yprior

def gold_standard_GPs(Xd, Yd, Xi, φ, s, ret_std, trans_type, untrans,
                      f_mean, exclude_mean, basis_type, ret_grad):
    """
    Provide easily interpreted & easily debugged code for each individual types of GP that results
    from a combination of the input options in [trans_type, f_mean, basis_type, ret_std, ret_grad].
    This represents a gold-standard against which the `GPI.__call__` can be compared.  For the sake
    of that comparison, this utility only returns the sufficient statistics of the posterior MAP.
    It is understood that this function will present excruciating redundancy.  That is intentional
    for the sake of encapsulated readablity.  A reference function that eliminates all redundancy
    called `consolidated` will be provided below.
    """
    nd, n_xdims = Xd.shape
    ni = Xi.shape[0]
    ind_d = arange(Xd.shape[0])
    ind_i = arange(Xi.shape[0])
    ind_ig = arange(Xi.shape[0] * (1 + n_xdims))
    if trans_type == "Log":
        trans = lambda y, dy=None: log(y) if dy is None else (log(y), dy / y.reshape((-1, 1)))
        inv_trans = lambda z, dz=None: exp(z) if dz is None else (exp(z), exp(z) * dz)
    elif trans_type == "Logit":
        trans = lambda y, dy=None: (log(y / (1 - y)) if dy is None else
                                   (log(y / (1 - y)),
                                    ((2 - y) / (y * (1 - y))).reshape((-1, 1)) * dy))
        inv_trans = lambda z, dz=None: (1 / (exp(-z) + 1) if dz is None else
                                       (1 / (exp(-z) + 1), exp(z) / (1 + exp(z))**2 * dz))
    elif trans_type == "Probit":
        trans = lambda y, dy=None: (sqrt(2) * erfinv(2 * y - 1) if dy is None else
                                   (sqrt(2) * erfinv(2 * y - 1),
                                   (sqrt(2 * π) * exp(erfinv(2 * y - 1)**2)).reshape((-1, 1)) * dy))
        inv_trans = lambda z, dz=None: ((1 + erf(z / sqrt(2))) / 2 if dz is None else
                                       ((1 + erf(z / sqrt(2))) / 2,
                                        exp(-0.5 * z**2) / sqrt(2 * π) * dz))

    # Each combination of input options:
    if not trans_type and not f_mean and not basis_type and not ret_std and not ret_grad:
        # Distance & auto-covariance
        σd, w, ℓ = φ['σd'], φ['w'], φ['ℓ']
        Rdd = cdist(Xd, Xd, 'seuclidean', V=(s * ℓ)**2)
        Rid = cdist(Xi, Xd, 'seuclidean', V=(s * ℓ)**2)
        Kdd = w**2 * exp(-0.5 * Rdd**2)
        Kdd[ind_d, ind_d] += σd**2
        Kid = w**2 * exp(-0.5 * Rid**2)
        # Inference
        μYi = (Kid @ solve(Kdd, Yd))
        return μYi.reshape(-1)
    elif trans_type and not f_mean and not basis_type and not ret_std and not ret_grad:
        # Move to the transformed space
        Zd = trans(Yd)
        # Distance & auto-covariance
        σd, w, ℓ = φ['σd'], φ['w'], φ['ℓ']
        Rdd = cdist(Xd, Xd, 'seuclidean', V=(s * ℓ)**2)
        Rid = cdist(Xi, Xd, 'seuclidean', V=(s * ℓ)**2)
        Kdd = w**2 * exp(-0.5 * Rdd**2)
        Kdd[ind_d, ind_d] += σd**2
        Kid = w**2 * exp(-0.5 * Rid**2)
        # Inference
        μZi = (Kid @ solve(Kdd, Zd))
        if untrans is False:
            return μZi.reshape(-1)
        else:
            # Return from the transformed space
            μYi = inv_trans(μZi)
            return μYi.reshape(-1)
    elif not trans_type and f_mean and not basis_type and not ret_std and not ret_grad:
        # Evaluate prior mean at Xd & Xi
        μYd_prior = f_mean(Xd)
        μYi_prior = f_mean(Xi)
        # Distance & auto-covariance
        σd, w, ℓ = φ['σd'], φ['w'], φ['ℓ']
        Rdd = cdist(Xd, Xd, 'seuclidean', V=(s * ℓ)**2)
        Rid = cdist(Xi, Xd, 'seuclidean', V=(s * ℓ)**2)
        Kdd = w**2 * exp(-0.5 * Rdd**2)
        Kdd[ind_d, ind_d] += σd**2
        Kid = w**2 * exp(-0.5 * Rid**2)
        # Inference
        if exclude_mean is False:
            μYi = (μYi_prior + Kid @ solve(Kdd, Yd - μYd_prior))
        else:
            μYi = (Kid @ solve(Kdd, Yd - μYd_prior))
        return μYi.reshape(-1)
    elif trans_type and f_mean and not basis_type and not ret_std and not ret_grad:
        # Evaluate prior mean at Xd & Xi
        μYd_prior = f_mean(Xd)
        μYi_prior = f_mean(Xi)
        # Move to the transformed space
        Zd = trans(Yd)
        μZd_prior = trans(μYd_prior)
        μZi_prior = trans(μYi_prior)
        # Distance & auto-covariance
        σd, w, ℓ = φ['σd'], φ['w'], φ['ℓ']
        Rdd = cdist(Xd, Xd, 'seuclidean', V=(s * ℓ)**2)
        Rid = cdist(Xi, Xd, 'seuclidean', V=(s * ℓ)**2)
        Kdd = w**2 * exp(-0.5 * Rdd**2)
        Kdd[ind_d, ind_d] += σd**2
        Kid = w**2 * exp(-0.5 * Rid**2)
        # Inference
        if exclude_mean is False:
            μZi = (μZi_prior + Kid @ solve(Kdd, Zd - μZd_prior))
        else:
            μZi = (Kid @ solve(Kdd, Zd - μZd_prior))
        if untrans is False:
            return μZi.reshape(-1)
        else:
            # Return from the transformed space
            μYi = inv_trans(μZi)
            return μYi.reshape(-1)
        # Note: cannot untransform when (f_mean or basis_type) and exclude_mean
    elif not trans_type and not f_mean and basis_type == "planar" and not ret_std and not ret_grad:
        # Explicit bases
        Hd = empty((nd, (1 + n_xdims)), dtype='float64')
        Hd[:, 0] = 1
        Hd[:, 1:] = Xd
        Hi = empty((ni, (1 + n_xdims)), dtype='float64')
        Hi[:, 0] = 1
        Hi[:, 1:] = Xi
        # Distance & auto-covariance
        σd, w, ℓ = φ['σd'], φ['w'], φ['ℓ']
        Rdd = cdist(Xd, Xd, 'seuclidean', V=(s * ℓ)**2)
        Rid = cdist(Xi, Xd, 'seuclidean', V=(s * ℓ)**2)
        Kdd = w**2 * exp(-0.5 * Rdd**2)
        Kdd[ind_d, ind_d] += σd**2
        Kid = w**2 * exp(-0.5 * Rid**2)
        # Inference
        Σβ_inv = Hd.T @ solve(Kdd, Hd)
        μβ = solve(Σβ_inv, Hd.T @ solve(Kdd, Yd))
        μYi = (Hi @ μβ + Kid @ solve(Kdd, Yd - Hd @ μβ))
        return μYi.reshape(-1)
    elif trans_type and not f_mean and basis_type == "planar" and not ret_std and not ret_grad:
        # Move to the transformed space
        Zd = trans(Yd)
        # Explicit bases
        Hd = empty((nd, (1 + n_xdims)), dtype='float64')
        Hd[:, 0] = 1
        Hd[:, 1:] = Xd
        Hi = empty((ni, (1 + n_xdims)), dtype='float64')
        Hi[:, 0] = 1
        Hi[:, 1:] = Xi
        # Distance & auto-covariance
        σd, w, ℓ = φ['σd'], φ['w'], φ['ℓ']
        Rdd = cdist(Xd, Xd, 'seuclidean', V=(s * ℓ)**2)
        Rid = cdist(Xi, Xd, 'seuclidean', V=(s * ℓ)**2)
        Kdd = w**2 * exp(-0.5 * Rdd**2)
        Kdd[ind_d, ind_d] += σd**2
        Kid = w**2 * exp(-0.5 * Rid**2)
        # Inference
        Σβ_inv = Hd.T @ solve(Kdd, Hd)
        μβ = solve(Σβ_inv, Hd.T @ solve(Kdd, Zd))
        μZi = (Hi @ μβ + Kid @ solve(Kdd, Zd - Hd @ μβ))
        if untrans is False:
            return μZi.reshape(-1)
        else:
            # Return from the transformed space
            μYi = inv_trans(μZi)
            return μYi.reshape(-1)
        # Note: cannot untransform when (f_mean or basis_type) and exclude_mean
    elif not trans_type and f_mean and basis_type == "planar" and not ret_std and not ret_grad:
        # Evaluate prior mean at Xd & Xi
        μYd_prior = f_mean(Xd)
        μYi_prior = f_mean(Xi)
        # Explicit bases
        Hd = empty((nd, (1 + n_xdims)), dtype='float64')
        Hd[:, 0] = 1
        Hd[:, 1:] = Xd
        Hi = empty((ni, (1 + n_xdims)), dtype='float64')
        Hi[:, 0] = 1
        Hi[:, 1:] = Xi
        # Distance & auto-covariance
        σd, w, ℓ = φ['σd'], φ['w'], φ['ℓ']
        Rdd = cdist(Xd, Xd, 'seuclidean', V=(s * ℓ)**2)
        Rid = cdist(Xi, Xd, 'seuclidean', V=(s * ℓ)**2)
        Kdd = w**2 * exp(-0.5 * Rdd**2)
        Kdd[ind_d, ind_d] += σd**2
        Kid = w**2 * exp(-0.5 * Rid**2)
        # Inference
        Σβ_inv = Hd.T @ solve(Kdd, Hd)
        μβ = solve(Σβ_inv, Hd.T @ solve(Kdd, Yd - μYd_prior))
        if exclude_mean is False:
            μYi = (μYi_prior + Hi @ μβ + Kid @ solve(Kdd, Yd - (μYd_prior + Hd @ μβ)))
        else:
            μYi = (Kid @ solve(Kdd, Yd - (μYd_prior + Hd @ μβ)))
        return μYi.reshape(-1)
    elif trans_type and f_mean and basis_type == "planar" and not ret_std and not ret_grad:
        # Evaluate prior mean at Xd & Xi
        μYd_prior = f_mean(Xd)
        μYi_prior = f_mean(Xi)
        # Move to the transformed space
        Zd = trans(Yd)
        μZd_prior = trans(μYd_prior)
        μZi_prior = trans(μYi_prior)
        # Explicit bases
        Hd = empty((nd, (1 + n_xdims)), dtype='float64')
        Hd[:, 0] = 1
        Hd[:, 1:] = Xd
        Hi = empty((ni, (1 + n_xdims)), dtype='float64')
        Hi[:, 0] = 1
        Hi[:, 1:] = Xi
        # Distance & auto-covariance
        σd, w, ℓ = φ['σd'], φ['w'], φ['ℓ']
        Rdd = cdist(Xd, Xd, 'seuclidean', V=(s * ℓ)**2)
        Rid = cdist(Xi, Xd, 'seuclidean', V=(s * ℓ)**2)
        Kdd = w**2 * exp(-0.5 * Rdd**2)
        Kdd[ind_d, ind_d] += σd**2
        Kid = w**2 * exp(-0.5 * Rid**2)
        # Inference
        Σβ_inv = Hd.T @ solve(Kdd, Hd)
        μβ = solve(Σβ_inv, Hd.T @ solve(Kdd, Zd - μZd_prior))
        if exclude_mean is False:
            μZi = (μZi_prior + Hi @ μβ + Kid @ solve(Kdd, Zd - (μZd_prior + Hd @ μβ)))
        else:
            μZi = (Kid @ solve(Kdd, Zd - (μZd_prior + Hd @ μβ)))
        if untrans is False:
            return μZi.reshape(-1)
        else:
            # Return from the transformed space
            μYi = inv_trans(μZi)
            return μYi.reshape(-1)
        # Note: cannot untransform when (f_mean or basis_type) and exclude_mean
    elif not trans_type and not f_mean and not basis_type and ret_std and not ret_grad:
        # Distance & auto-covariance
        σd, w, ℓ = φ['σd'], φ['w'], φ['ℓ']
        Rdd = cdist(Xd, Xd, 'seuclidean', V=(s * ℓ)**2)
        Rid = cdist(Xi, Xd, 'seuclidean', V=(s * ℓ)**2)
        Rii = cdist(Xi, Xi, 'seuclidean', V=(s * ℓ)**2)
        Kdd = w**2 * exp(-0.5 * Rdd**2)
        Kdd[ind_d, ind_d] += σd**2
        Kid = w**2 * exp(-0.5 * Rid**2)
        Kii = w**2 * exp(-0.5 * Rii**2)
        # Inference
        μYi = (Kid @ solve(Kdd, Yd))
        Σii = Kii - Kid @ solve(Kdd, Kid.T)
        if ret_std is True:
            σYi = sqrt(Σii[ind_i, ind_i]).reshape((-1, 1))
            return μYi.reshape(-1), σYi.reshape(-1)
        elif ret_std == "covar":
            return μYi.reshape(-1), Σii
    elif trans_type and not f_mean and not basis_type and ret_std and not ret_grad:
        # Move to the transformed space
        Zd = trans(Yd)
        # Distance & auto-covariance
        σd, w, ℓ = φ['σd'], φ['w'], φ['ℓ']
        Rdd = cdist(Xd, Xd, 'seuclidean', V=(s * ℓ)**2)
        Rid = cdist(Xi, Xd, 'seuclidean', V=(s * ℓ)**2)
        Rii = cdist(Xi, Xi, 'seuclidean', V=(s * ℓ)**2)
        Kdd = w**2 * exp(-0.5 * Rdd**2)
        Kdd[ind_d, ind_d] += σd**2
        Kid = w**2 * exp(-0.5 * Rid**2)
        Kii = w**2 * exp(-0.5 * Rii**2)
        # Inference
        μZi = (Kid @ solve(Kdd, Zd))
        Σii = Kii - Kid @ solve(Kdd, Kid.T)
        if ret_std is True:
            σZi = sqrt(Σii[ind_i, ind_i]).reshape((-1, 1))
            if untrans is False:
                return μZi.reshape(-1), σZi.reshape(-1)
            else:
                Zi_lohi = array([(μZi - σZi).reshape(-1), (μZi + σZi).reshape(-1)])
                # Return from the transformed space
                μYi = inv_trans(μZi)
                Yi_lohi = inv_trans(Zi_lohi)
                return μYi.reshape(-1), Yi_lohi - μYi.reshape(-1)
        elif ret_std == "covar":
            if untrans is False:
                return μZi.reshape(-1), Σii
            # Note: cannot untransform when ret_std == "covar"
    elif not trans_type and f_mean and not basis_type and ret_std and not ret_grad:
        # Evaluate prior mean at Xd & Xi
        μYd_prior = f_mean(Xd)
        μYi_prior = f_mean(Xi)
        # Distance & auto-covariance
        σd, w, ℓ = φ['σd'], φ['w'], φ['ℓ']
        Rdd = cdist(Xd, Xd, 'seuclidean', V=(s * ℓ)**2)
        Rid = cdist(Xi, Xd, 'seuclidean', V=(s * ℓ)**2)
        Rii = cdist(Xi, Xi, 'seuclidean', V=(s * ℓ)**2)
        Kdd = w**2 * exp(-0.5 * Rdd**2)
        Kdd[ind_d, ind_d] += σd**2
        Kid = w**2 * exp(-0.5 * Rid**2)
        Kii = w**2 * exp(-0.5 * Rii**2)
        # Inference
        if exclude_mean is False:
            μYi = (μYi_prior + Kid @ solve(Kdd, Yd - μYd_prior))
        else:
            μYi = (Kid @ solve(Kdd, Yd - μYd_prior))
        Σii = Kii - Kid @ solve(Kdd, Kid.T)
        if ret_std is True:
            σYi = sqrt(Σii[ind_i, ind_i]).reshape((-1, 1))
            return μYi.reshape(-1), σYi.reshape(-1)
        elif ret_std == "covar":
            return μYi.reshape(-1), Σii
    elif trans_type and f_mean and not basis_type and ret_std and not ret_grad:
        # Evaluate prior mean at Xd & Xi
        μYd_prior = f_mean(Xd)
        μYi_prior = f_mean(Xi)
        # Move to the transformed space
        Zd = trans(Yd)
        μZd_prior = trans(μYd_prior)
        μZi_prior = trans(μYi_prior)
        # Distance & auto-covariance
        σd, w, ℓ = φ['σd'], φ['w'], φ['ℓ']
        Rdd = cdist(Xd, Xd, 'seuclidean', V=(s * ℓ)**2)
        Rid = cdist(Xi, Xd, 'seuclidean', V=(s * ℓ)**2)
        Rii = cdist(Xi, Xi, 'seuclidean', V=(s * ℓ)**2)
        Kdd = w**2 * exp(-0.5 * Rdd**2)
        Kdd[ind_d, ind_d] += σd**2
        Kid = w**2 * exp(-0.5 * Rid**2)
        Kii = w**2 * exp(-0.5 * Rii**2)
        # Inference
        if exclude_mean is False:
            μZi = (μZi_prior + Kid @ solve(Kdd, Zd - μZd_prior))
        else:
            μZi = (Kid @ solve(Kdd, Zd - μZd_prior))
        Σii = Kii - Kid @ solve(Kdd, Kid.T)
        if ret_std is True:
            σZi = sqrt(Σii[ind_i, ind_i]).reshape((-1, 1))
            if untrans is False:
                return μZi.reshape(-1), σZi.reshape(-1)
            else:
                Zi_lohi = array([(μZi - σZi).reshape(-1), (μZi + σZi).reshape(-1)])
                # Return from the transformed space
                μYi = inv_trans(μZi)
                Yi_lohi = inv_trans(Zi_lohi)
                return μYi.reshape(-1), Yi_lohi - μYi.reshape(-1)
                # Note: cannot untransform when (f_mean or basis_type) and exclude_mean
        elif ret_std == "covar":
            if untrans is False:
                return μZi.reshape(-1), Σii
            # Note: cannot untransform when ret_std == "covar"
    elif not trans_type and not f_mean and basis_type == "planar" and ret_std and not ret_grad:
        # Explicit bases
        Hd = empty((nd, (1 + n_xdims)), dtype='float64')
        Hd[:, 0] = 1
        Hd[:, 1:] = Xd
        Hi = empty((ni, (1 + n_xdims)), dtype='float64')
        Hi[:, 0] = 1
        Hi[:, 1:] = Xi
        # Distance & auto-covariance
        σd, w, ℓ = φ['σd'], φ['w'], φ['ℓ']
        Rdd = cdist(Xd, Xd, 'seuclidean', V=(s * ℓ)**2)
        Rid = cdist(Xi, Xd, 'seuclidean', V=(s * ℓ)**2)
        Rii = cdist(Xi, Xi, 'seuclidean', V=(s * ℓ)**2)
        Kdd = w**2 * exp(-0.5 * Rdd**2)
        Kdd[ind_d, ind_d] += σd**2
        Kid = w**2 * exp(-0.5 * Rid**2)
        Kii = w**2 * exp(-0.5 * Rii**2)
        # Inference
        Σβ_inv = Hd.T @ solve(Kdd, Hd)
        μβ = solve(Σβ_inv, Hd.T @ solve(Kdd, Yd))
        μYi = (Hi @ μβ + Kid @ solve(Kdd, Yd - Hd @ μβ))
        tmp = Hi - Kid @ solve(Kdd, Hd)
        Σii = Kii - Kid @ solve(Kdd, Kid.T) + tmp @ solve(Σβ_inv, tmp.T)
        if ret_std is True:
            σYi = sqrt(Σii[ind_i, ind_i]).reshape((-1, 1))
            return μYi.reshape(-1), σYi.reshape(-1)
        elif ret_std == "covar":
            return μYi.reshape(-1), Σii
    elif trans_type and not f_mean and basis_type == "planar" and ret_std and not ret_grad:
        # Move to the transformed space
        Zd = trans(Yd)
        # Explicit bases
        Hd = empty((nd, (1 + n_xdims)), dtype='float64')
        Hd[:, 0] = 1
        Hd[:, 1:] = Xd
        Hi = empty((ni, (1 + n_xdims)), dtype='float64')
        Hi[:, 0] = 1
        Hi[:, 1:] = Xi
        # Distance & auto-covariance
        σd, w, ℓ = φ['σd'], φ['w'], φ['ℓ']
        Rdd = cdist(Xd, Xd, 'seuclidean', V=(s * ℓ)**2)
        Rid = cdist(Xi, Xd, 'seuclidean', V=(s * ℓ)**2)
        Rii = cdist(Xi, Xi, 'seuclidean', V=(s * ℓ)**2)
        Kdd = w**2 * exp(-0.5 * Rdd**2)
        Kdd[ind_d, ind_d] += σd**2
        Kid = w**2 * exp(-0.5 * Rid**2)
        Kii = w**2 * exp(-0.5 * Rii**2)
        # Inference
        Σβ_inv = Hd.T @ solve(Kdd, Hd)
        μβ = solve(Σβ_inv, Hd.T @ solve(Kdd, Zd))
        μZi = (Hi @ μβ + Kid @ solve(Kdd, Zd - Hd @ μβ))
        tmp = Hi - Kid @ solve(Kdd, Hd)
        Σii = Kii - Kid @ solve(Kdd, Kid.T) + tmp @ solve(Σβ_inv, tmp.T)
        if ret_std is True:
            σZi = sqrt(Σii[ind_i, ind_i]).reshape((-1, 1))
            if untrans is False:
                return μZi.reshape(-1), σZi.reshape(-1)
            else:
                Zi_lohi = array([(μZi - σZi).reshape(-1), (μZi + σZi).reshape(-1)])
                # Return from the transformed space
                μYi = inv_trans(μZi)
                Yi_lohi = inv_trans(Zi_lohi)
                return μYi.reshape(-1), Yi_lohi - μYi.reshape(-1)
                # Note: cannot untransform when (f_mean or basis_type) and exclude_mean
        elif ret_std == "covar":
            if untrans is False:
                return μZi.reshape(-1), Σii
            # Note: cannot untransform when ret_std == "covar"
    elif not trans_type and f_mean and basis_type == "planar" and ret_std and not ret_grad:
        # Evaluate prior mean at Xd & Xi
        μYd_prior = f_mean(Xd)
        μYi_prior = f_mean(Xi)
        # Explicit bases
        Hd = empty((nd, (1 + n_xdims)), dtype='float64')
        Hd[:, 0] = 1
        Hd[:, 1:] = Xd
        Hi = empty((ni, (1 + n_xdims)), dtype='float64')
        Hi[:, 0] = 1
        Hi[:, 1:] = Xi
        # Distance & auto-covariance
        σd, w, ℓ = φ['σd'], φ['w'], φ['ℓ']
        Rdd = cdist(Xd, Xd, 'seuclidean', V=(s * ℓ)**2)
        Rid = cdist(Xi, Xd, 'seuclidean', V=(s * ℓ)**2)
        Rii = cdist(Xi, Xi, 'seuclidean', V=(s * ℓ)**2)
        Kdd = w**2 * exp(-0.5 * Rdd**2)
        Kdd[ind_d, ind_d] += σd**2
        Kid = w**2 * exp(-0.5 * Rid**2)
        Kii = w**2 * exp(-0.5 * Rii**2)
        # Inference
        Σβ_inv = Hd.T @ solve(Kdd, Hd)
        μβ = solve(Σβ_inv, Hd.T @ solve(Kdd, Yd - μYd_prior))
        if exclude_mean is False:
            μYi = (μYi_prior + Hi @ μβ + Kid @ solve(Kdd, Yd - (μYd_prior + Hd @ μβ)))
        else:
            μYi = (Kid @ solve(Kdd, Yd - (μYd_prior + Hd @ μβ)))
        tmp = Hi - Kid @ solve(Kdd, Hd)
        Σii = Kii - Kid @ solve(Kdd, Kid.T) + tmp @ solve(Σβ_inv, tmp.T)
        if ret_std is True:
            σYi = sqrt(Σii[ind_i, ind_i]).reshape((-1, 1))
            return μYi.reshape(-1), σYi.reshape(-1)
        elif ret_std == "covar":
            return μYi.reshape(-1), Σii
    elif trans_type and f_mean and basis_type == "planar" and ret_std and not ret_grad:
        # Evaluate prior mean at Xd & Xi
        μYd_prior = f_mean(Xd)
        μYi_prior = f_mean(Xi)
        # Move to the transformed space
        Zd = trans(Yd)
        μZd_prior = trans(μYd_prior)
        μZi_prior = trans(μYi_prior)
        # Explicit bases
        Hd = empty((nd, (1 + n_xdims)), dtype='float64')
        Hd[:, 0] = 1
        Hd[:, 1:] = Xd
        Hi = empty((ni, (1 + n_xdims)), dtype='float64')
        Hi[:, 0] = 1
        Hi[:, 1:] = Xi
        # Distance & auto-covariance
        σd, w, ℓ = φ['σd'], φ['w'], φ['ℓ']
        Rdd = cdist(Xd, Xd, 'seuclidean', V=(s * ℓ)**2)
        Rid = cdist(Xi, Xd, 'seuclidean', V=(s * ℓ)**2)
        Rii = cdist(Xi, Xi, 'seuclidean', V=(s * ℓ)**2)
        Kdd = w**2 * exp(-0.5 * Rdd**2)
        Kdd[ind_d, ind_d] += σd**2
        Kid = w**2 * exp(-0.5 * Rid**2)
        Kii = w**2 * exp(-0.5 * Rii**2)
        # Inference
        Σβ_inv = Hd.T @ solve(Kdd, Hd)
        μβ = solve(Σβ_inv, Hd.T @ solve(Kdd, Zd - μZd_prior))
        if exclude_mean is False:
            μZi = (μZi_prior + Hi @ μβ + Kid @ solve(Kdd, Zd - (μZd_prior + Hd @ μβ)))
        else:
            μZi = (Kid @ solve(Kdd, Zd - (μZd_prior + Hd @ μβ)))
        tmp = Hi - Kid @ solve(Kdd, Hd)
        Σii = Kii - Kid @ solve(Kdd, Kid.T) + tmp @ solve(Σβ_inv, tmp.T)
        if ret_std is True:
            σZi = sqrt(Σii[ind_i, ind_i]).reshape((-1, 1))
            if untrans is False:
                return μZi.reshape(-1), σZi.reshape(-1)
            else:
                Zi_lohi = array([(μZi - σZi).reshape(-1), (μZi + σZi).reshape(-1)])
                # Return from the transformed space
                μYi = inv_trans(μZi)
                Yi_lohi = inv_trans(Zi_lohi)
                return μYi.reshape(-1), Yi_lohi - μYi.reshape(-1)
                # Note: cannot untransform when (f_mean or basis_type) and exclude_mean
        elif ret_std == "covar":
            if untrans is False:
                return μZi.reshape(-1), Σii
            # Note: cannot untransform when ret_std == "covar"
    elif not trans_type and not f_mean and not basis_type and not ret_std and ret_grad:
        # Distance & auto-covariance
        σd, w, ℓ = φ['σd'], φ['w'], φ['ℓ']
        Rdd = radius(Xd,  Xd, s)
        Rid = radius(Xi, Xd, s)  # will be use repeatedly for Y & each gradient direction
        Rii = radius(Xi, Xi, s)  # will be use repeatedly for Y & each gradient direction
        Kdd = w**2 * exp(-0.5 * ((Rdd / ℓ)**2).sum(axis=2))
        Kdd[ind_d, ind_d] += σd**2
        Kid = empty((ni * (1 + n_xdims), nd))
        Kid[:ni] = w**2 * exp(-0.5 * ((Rid / ℓ)**2).sum(axis=2))
        for k in range(n_xdims):
            lo, hi = ni * (k + 1), ni * (k + 2)
            Kid[lo:hi] = -Rid[:, :, k] / (s[k] * ℓ[k]**2) * Kid[:ni]
        # Inference
        Kinv_Zd = solve(Kdd, Yd)
        μYi = (Kid[:ni] @ Kinv_Zd).reshape(-1)
        μYg = (Kid[ni:] @ Kinv_Zd).reshape((n_xdims, ni)).T
        return μYi, μYg
    elif trans_type and not f_mean and not basis_type and not ret_std and ret_grad:
        # Move to the transformed space
        Zd = trans(Yd)
        # Distance & auto-covariance
        σd, w, ℓ = φ['σd'], φ['w'], φ['ℓ']
        Rdd = radius(Xd,  Xd, s)
        Rid = radius(Xi, Xd, s)  # will be use repeatedly for Y & each gradient direction
        Rii = radius(Xi, Xi, s)  # will be use repeatedly for Y & each gradient direction
        Kdd = w**2 * exp(-0.5 * ((Rdd / ℓ)**2).sum(axis=2))
        Kdd[ind_d, ind_d] += σd**2
        Kid = empty((ni * (1 + n_xdims), nd))
        Kid[:ni] = w**2 * exp(-0.5 * ((Rid / ℓ)**2).sum(axis=2))
        for k in range(n_xdims):
            lo, hi = ni * (k + 1), ni * (k + 2)
            Kid[lo:hi] = -Rid[:, :, k] / (s[k] * ℓ[k]**2) * Kid[:ni]
        # Inference
        Kinv_Zd = solve(Kdd, Zd)
        μZi = (Kid[:ni] @ Kinv_Zd)
        μZg = (Kid[ni:] @ Kinv_Zd).reshape((n_xdims, ni)).T
        if untrans is False:
            return μZi.reshape(-1), μZg
        else:
            # Return from the transformed space
            μYi, μYg = inv_trans(μZi, μZg)
            return μYi.reshape(-1), μYg
    elif not trans_type and f_mean and not basis_type and not ret_std and ret_grad:
        # Evaluate prior mean at Xd & Xi
        μYd_prior = f_mean(Xd)
        μYi_prior, μYg_prior = f_mean(Xi, grad=True)
        # Distance & auto-covariance
        σd, w, ℓ = φ['σd'], φ['w'], φ['ℓ']
        Rdd = radius(Xd,  Xd, s)
        Rid = radius(Xi, Xd, s)  # will be use repeatedly for Y & each gradient direction
        Rii = radius(Xi, Xi, s)  # will be use repeatedly for Y & each gradient direction
        Kdd = w**2 * exp(-0.5 * ((Rdd / ℓ)**2).sum(axis=2))
        Kdd[ind_d, ind_d] += σd**2
        Kid = empty((ni * (1 + n_xdims), nd))
        Kid[:ni] = w**2 * exp(-0.5 * ((Rid / ℓ)**2).sum(axis=2))
        for k in range(n_xdims):
            lo, hi = ni * (k + 1), ni * (k + 2)
            Kid[lo:hi] = -Rid[:, :, k] / (s[k] * ℓ[k]**2) * Kid[:ni]
        # Inference
        Kinv_Zd = solve(Kdd, Yd - μYd_prior)
        if exclude_mean is False:
            μYi = (μYi_prior + Kid[:ni] @ Kinv_Zd)
            μYg = μYg_prior + (Kid[ni:] @ Kinv_Zd).reshape((n_xdims, ni)).T
        else:
            μYi = (Kid[:ni] @ Kinv_Zd)
            μYg = (Kid[ni:] @ Kinv_Zd).reshape((n_xdims, ni)).T
        return μYi.reshape(-1), μYg
    elif trans_type and f_mean and not basis_type and not ret_std and ret_grad:
        # Evaluate prior mean at Xd & Xi
        μYd_prior = f_mean(Xd)
        μYi_prior, μYg_prior = f_mean(Xi, grad=True)
        # Move to the transformed space
        Zd = trans(Yd)
        μZd_prior = trans(μYd_prior)
        μZi_prior, μZg_prior = trans(μYi_prior, μYg_prior)
        # Distance & auto-covariance
        σd, w, ℓ = φ['σd'], φ['w'], φ['ℓ']
        Rdd = radius(Xd,  Xd, s)
        Rid = radius(Xi, Xd, s)  # will be use repeatedly for Y & each gradient direction
        Rii = radius(Xi, Xi, s)  # will be use repeatedly for Y & each gradient direction
        Kdd = w**2 * exp(-0.5 * ((Rdd / ℓ)**2).sum(axis=2))
        Kdd[ind_d, ind_d] += σd**2
        Kid = empty((ni * (1 + n_xdims), nd))
        Kid[:ni] = w**2 * exp(-0.5 * ((Rid / ℓ)**2).sum(axis=2))
        for k in range(n_xdims):
            lo, hi = ni * (k + 1), ni * (k + 2)
            Kid[lo:hi] = -Rid[:, :, k] / (s[k] * ℓ[k]**2) * Kid[:ni]
        # Inference
        Kinv_Zd = solve(Kdd, Zd - μZd_prior)
        if exclude_mean is False:
            μZi = (μZi_prior + Kid[:ni] @ Kinv_Zd)
            μZg = μZg_prior + (Kid[ni:] @ Kinv_Zd).reshape((n_xdims, ni)).T
        else:
            μZi = (Kid[:ni] @ Kinv_Zd)
            μZg = (Kid[ni:] @ Kinv_Zd).reshape((n_xdims, ni)).T
        if untrans is False:
            return μZi.reshape(-1), μZg
        else:
            # Return from the transformed space
            μYi, μYg = inv_trans(μZi, μZg)
            return μYi.reshape(-1), μYg
        # Note: cannot untransform when (f_mean or basis_type) and exclude_mean
    elif not trans_type and not f_mean and basis_type == "planar" and not ret_std and ret_grad:
        # Explicit bases
        Hd = empty((nd, (1 + n_xdims)), dtype='float64')
        Hd[:, 0] = 1
        Hd[:, 1:] = Xd
        Hig = full((ni, (1 + n_xdims), (1 + n_xdims)), 0, dtype='float64')
        Hi = Hig[:, 0, :]
        Hi[:, 0] = 1
        Hi[:, 1:] = Xi
        Hg = Hig[:, 1:, :]
        for i in range(n_xdims):
            Hg[:, i, i + 1] = 1
        # Distance & auto-covariance
        σd, w, ℓ = φ['σd'], φ['w'], φ['ℓ']
        Rdd = radius(Xd,  Xd, s)
        Rid = radius(Xi, Xd, s)  # will be use repeatedly for Y & each gradient direction
        Rii = radius(Xi, Xi, s)  # will be use repeatedly for Y & each gradient direction
        Kdd = w**2 * exp(-0.5 * ((Rdd / ℓ)**2).sum(axis=2))
        Kdd[ind_d, ind_d] += σd**2
        Kid = empty((ni * (1 + n_xdims), nd))
        Kid[:ni] = w**2 * exp(-0.5 * ((Rid / ℓ)**2).sum(axis=2))
        for k in range(n_xdims):
            lo, hi = ni * (k + 1), ni * (k + 2)
            Kid[lo:hi] = -Rid[:, :, k] / (s[k] * ℓ[k]**2) * Kid[:ni]
        # Inference
        Σβ_inv = Hd.T @ solve(Kdd, Hd)
        μβ = solve(Σβ_inv, Hd.T @ solve(Kdd, Yd))
        Kinv_Zd = solve(Kdd, Yd - Hd @ μβ)
        if exclude_mean is False:
            μYi = (Hi @ μβ + Kid[:ni] @ Kinv_Zd)
            μYg = Hg @ μβ.reshape(-1) + (Kid[ni:] @ Kinv_Zd).reshape((n_xdims, ni)).T
        else:
            μYi = (Kid[:ni] @ Kinv_Zd)
            μYg = (Kid[ni:] @ Kinv_Zd).reshape((n_xdims, ni)).T
        return μYi.reshape(-1), μYg
    elif trans_type and not f_mean and basis_type == "planar" and not ret_std and ret_grad:
        # Move to the transformed space
        Zd = trans(Yd)
        # Explicit bases
        Hd = empty((nd, (1 + n_xdims)), dtype='float64')
        Hd[:, 0] = 1
        Hd[:, 1:] = Xd
        Hig = full((ni, (1 + n_xdims), (1 + n_xdims)), 0, dtype='float64')
        Hi = Hig[:, 0, :]
        Hi[:, 0] = 1
        Hi[:, 1:] = Xi
        Hg = Hig[:, 1:, :]
        for i in range(n_xdims):
            Hg[:, i, i + 1] = 1
        # Distance & auto-covariance
        σd, w, ℓ = φ['σd'], φ['w'], φ['ℓ']
        Rdd = radius(Xd,  Xd, s)
        Rid = radius(Xi, Xd, s)  # will be use repeatedly for Y & each gradient direction
        Rii = radius(Xi, Xi, s)  # will be use repeatedly for Y & each gradient direction
        Kdd = w**2 * exp(-0.5 * ((Rdd / ℓ)**2).sum(axis=2))
        Kdd[ind_d, ind_d] += σd**2
        Kid = empty((ni * (1 + n_xdims), nd))
        Kid[:ni] = w**2 * exp(-0.5 * ((Rid / ℓ)**2).sum(axis=2))
        for k in range(n_xdims):
            lo, hi = ni * (k + 1), ni * (k + 2)
            Kid[lo:hi] = -Rid[:, :, k] / (s[k] * ℓ[k]**2) * Kid[:ni]
        # Inference
        Σβ_inv = Hd.T @ solve(Kdd, Hd)
        μβ = solve(Σβ_inv, Hd.T @ solve(Kdd, Zd))
        Kinv_Zd = solve(Kdd, Zd - Hd @ μβ)
        if exclude_mean is False:
            μZi = (Hi @ μβ + Kid[:ni] @ Kinv_Zd)
            μZg = Hg @ μβ.reshape(-1) + (Kid[ni:] @ Kinv_Zd).reshape((n_xdims, ni)).T
        else:
            μZi = (Kid[:ni] @ Kinv_Zd)
            μZg = (Kid[ni:] @ Kinv_Zd).reshape((n_xdims, ni)).T
        if untrans is False:
            return μZi.reshape(-1), μZg
        else:
            # Return from the transformed space
            μYi, μYg = inv_trans(μZi, μZg)
            return μYi.reshape(-1), μYg
        # Note: cannot untransform when (f_mean or basis_type) and exclude_mean
    elif not trans_type and f_mean and basis_type == "planar" and not ret_std and ret_grad:
        # Evaluate prior mean at Xd & Xi
        μYd_prior = f_mean(Xd)
        μYi_prior, μYg_prior = f_mean(Xi, grad=True)
        # Explicit bases
        Hd = empty((nd, (1 + n_xdims)), dtype='float64')
        Hd[:, 0] = 1
        Hd[:, 1:] = Xd
        Hig = full((ni, (1 + n_xdims), (1 + n_xdims)), 0, dtype='float64')
        Hi = Hig[:, 0, :]
        Hi[:, 0] = 1
        Hi[:, 1:] = Xi
        Hg = Hig[:, 1:, :]
        for i in range(n_xdims):
            Hg[:, i, i + 1] = 1
        # Distance & auto-covariance
        σd, w, ℓ = φ['σd'], φ['w'], φ['ℓ']
        Rdd = radius(Xd,  Xd, s)
        Rid = radius(Xi, Xd, s)  # will be use repeatedly for Y & each gradient direction
        Rii = radius(Xi, Xi, s)  # will be use repeatedly for Y & each gradient direction
        Kdd = w**2 * exp(-0.5 * ((Rdd / ℓ)**2).sum(axis=2))
        Kdd[ind_d, ind_d] += σd**2
        Kid = empty((ni * (1 + n_xdims), nd))
        Kid[:ni] = w**2 * exp(-0.5 * ((Rid / ℓ)**2).sum(axis=2))
        for k in range(n_xdims):
            lo, hi = ni * (k + 1), ni * (k + 2)
            Kid[lo:hi] = -Rid[:, :, k] / (s[k] * ℓ[k]**2) * Kid[:ni]
        # Inference
        Σβ_inv = Hd.T @ solve(Kdd, Hd)
        μβ = solve(Σβ_inv, Hd.T @ solve(Kdd, Yd - μYd_prior))
        Kinv_Zd = solve(Kdd, Yd - (μYd_prior + Hd @ μβ))
        if exclude_mean is False:
            μYi = (μYi_prior + Hi @ μβ + Kid[:ni] @ Kinv_Zd)
            μYg = μYg_prior + Hg @ μβ.reshape(-1) + (Kid[ni:] @ Kinv_Zd).reshape((n_xdims, ni)).T
        else:
            μYi = (Kid[:ni] @ Kinv_Zd)
            μYg = (Kid[ni:] @ Kinv_Zd).reshape((n_xdims, ni)).T
        return μYi.reshape(-1), μYg
    elif trans_type and f_mean and basis_type == "planar" and not ret_std and ret_grad:
        # Evaluate prior mean at Xd & Xi
        μYd_prior = f_mean(Xd)
        μYi_prior, μYg_prior = f_mean(Xi, grad=True)
        # Move to the transformed space
        Zd = trans(Yd)
        μZd_prior = trans(μYd_prior)
        μZi_prior, μZg_prior = trans(μYi_prior, μYg_prior)
        # Explicit bases
        Hd = empty((nd, (1 + n_xdims)), dtype='float64')
        Hd[:, 0] = 1
        Hd[:, 1:] = Xd
        Hig = full((ni, (1 + n_xdims), (1 + n_xdims)), 0, dtype='float64')
        Hi = Hig[:, 0, :]
        Hi[:, 0] = 1
        Hi[:, 1:] = Xi
        Hg = Hig[:, 1:, :]
        for i in range(n_xdims):
            Hg[:, i, i + 1] = 1
        # Distance & auto-covariance
        σd, w, ℓ = φ['σd'], φ['w'], φ['ℓ']
        Rdd = radius(Xd,  Xd, s)
        Rid = radius(Xi, Xd, s)  # will be use repeatedly for Y & each gradient direction
        Rii = radius(Xi, Xi, s)  # will be use repeatedly for Y & each gradient direction
        Kdd = w**2 * exp(-0.5 * ((Rdd / ℓ)**2).sum(axis=2))
        Kdd[ind_d, ind_d] += σd**2
        Kid = empty((ni * (1 + n_xdims), nd))
        Kid[:ni] = w**2 * exp(-0.5 * ((Rid / ℓ)**2).sum(axis=2))
        for k in range(n_xdims):
            lo, hi = ni * (k + 1), ni * (k + 2)
            Kid[lo:hi] = -Rid[:, :, k] / (s[k] * ℓ[k]**2) * Kid[:ni]
        # Inference
        Σβ_inv = Hd.T @ solve(Kdd, Hd)
        μβ = solve(Σβ_inv, Hd.T @ solve(Kdd, Zd - μZd_prior))
        Kinv_Zd = solve(Kdd, Zd - (μZd_prior + Hd @ μβ))
        if exclude_mean is False:
            μZi = (μZi_prior + Hi @ μβ + Kid[:ni] @ Kinv_Zd)
            μZg = μZg_prior + Hg @ μβ.reshape(-1) + (Kid[ni:] @ Kinv_Zd).reshape((n_xdims, ni)).T
        else:
            μZi = (Kid[:ni] @ Kinv_Zd)
            μZg = (Kid[ni:] @ Kinv_Zd).reshape((n_xdims, ni)).T
        if untrans is False:
            return μZi.reshape(-1), μZg
        else:
            # Return from the transformed space
            μYi, μYg = inv_trans(μZi, μZg)
            return μYi.reshape(-1), μYg
        # Note: cannot untransform when (f_mean or basis_type) and exclude_mean
    elif not trans_type and not f_mean and not basis_type and ret_std and ret_grad:
        # Distance & auto-covariance
        σd, w, ℓ = φ['σd'], φ['w'], φ['ℓ']
        Rdd = radius(Xd,  Xd, s)
        Rid = radius(Xi, Xd, s)  # will be use repeatedly for Y & each gradient direction
        Rii = radius(Xi, Xi, s)  # will be use repeatedly for Y & each gradient direction
        Kdd = w**2 * exp(-0.5 * ((Rdd / ℓ)**2).sum(axis=2))
        Kdd[ind_d, ind_d] += σd**2
        Kid = empty((ni * (1 + n_xdims), nd))
        Kid[:ni] = w**2 * exp(-0.5 * ((Rid / ℓ)**2).sum(axis=2))
        for k in range(n_xdims):
            lo, hi = ni * (k + 1), ni * (k + 2)
            Kid[lo:hi] = -Rid[:, :, k] / (s[k] * ℓ[k]**2) * Kid[:ni]
        Kii = empty((ni * (1 + n_xdims), ni * (1 + n_xdims)))
        Kii[:ni, :ni] = w**2 * exp(-0.5 * ((Rii / ℓ)**2).sum(axis=2))
        for k in range(n_xdims):
            lo, hi = ni * (k + 1), ni * (k + 2)
            Kii[lo:hi, :ni] = -Rii[:, :, k] / (s[k] * ℓ[k]**2) * Kii[:ni, :ni]
        Kii[:ni, ni:] = -Kii[ni:, :ni].T
        for ki in range(n_xdims):
            i_lo, i_hi = ni * (ki + 1), ni * (ki + 2)
            for kj in range(n_xdims):
                δ = 1 if ki == kj else 0
                j_lo, j_hi = ni * (kj + 1), ni * (kj + 2)
                Kii[i_lo:i_hi, j_lo:j_hi] = ((δ - Rii[:, :, ki] / ℓ[ki] * Rii[:, :, kj] / ℓ[kj]) *
                                             Kii[:ni, :ni] / (s[ki] * ℓ[ki] * s[kj] * ℓ[kj]))
        # Inference
        Kinv_Zd = solve(Kdd, Yd)
        μYi = (Kid[:ni] @ Kinv_Zd)
        μYg = (Kid[ni:] @ Kinv_Zd).reshape((n_xdims, ni)).T
        Σii = Kii - Kid @ solve(Kdd, Kid.T)
        if ret_std is True:
            σYig = sqrt(Σii[ind_ig, ind_ig]).reshape(-1)
            σYi, σYg = σYig[:ni], σYig[ni:].reshape((n_xdims, ni)).T
            return (μYi.reshape(-1), μYg), (σYi.reshape(-1), σYg)
        elif ret_std == "covar":
            return (μYi.reshape(-1), μYg), Σii
    elif trans_type and not f_mean and not basis_type and ret_std and ret_grad:
        # Move to the transformed space
        Zd = trans(Yd)
        # Distance & auto-covariance
        σd, w, ℓ = φ['σd'], φ['w'], φ['ℓ']
        Rdd = radius(Xd,  Xd, s)
        Rid = radius(Xi, Xd, s)  # will be use repeatedly for Y & each gradient direction
        Rii = radius(Xi, Xi, s)  # will be use repeatedly for Y & each gradient direction
        Kdd = w**2 * exp(-0.5 * ((Rdd / ℓ)**2).sum(axis=2))
        Kdd[ind_d, ind_d] += σd**2
        Kid = empty((ni * (1 + n_xdims), nd))
        Kid[:ni] = w**2 * exp(-0.5 * ((Rid / ℓ)**2).sum(axis=2))
        for k in range(n_xdims):
            lo, hi = ni * (k + 1), ni * (k + 2)
            Kid[lo:hi] = -Rid[:, :, k] / (s[k] * ℓ[k]**2) * Kid[:ni]
        Kii = empty((ni * (1 + n_xdims), ni * (1 + n_xdims)))
        Kii[:ni, :ni] = w**2 * exp(-0.5 * ((Rii / ℓ)**2).sum(axis=2))
        for k in range(n_xdims):
            lo, hi = ni * (k + 1), ni * (k + 2)
            Kii[lo:hi, :ni] = -Rii[:, :, k] / (s[k] * ℓ[k]**2) * Kii[:ni, :ni]
        Kii[:ni, ni:] = -Kii[ni:, :ni].T
        for ki in range(n_xdims):
            i_lo, i_hi = ni * (ki + 1), ni * (ki + 2)
            for kj in range(n_xdims):
                δ = 1 if ki == kj else 0
                j_lo, j_hi = ni * (kj + 1), ni * (kj + 2)
                Kii[i_lo:i_hi, j_lo:j_hi] = ((δ - Rii[:, :, ki] / ℓ[ki] * Rii[:, :, kj] / ℓ[kj]) *
                                             Kii[:ni, :ni] / (s[ki] * ℓ[ki] * s[kj] * ℓ[kj]))
        # Inference
        Kinv_Zd = solve(Kdd, Zd)
        μZi = (Kid[:ni] @ Kinv_Zd)
        μZg = (Kid[ni:] @ Kinv_Zd).reshape((n_xdims, ni)).T
        Σii = Kii - Kid @ solve(Kdd, Kid.T)
        if ret_std is True:
            σZig = sqrt(Σii[ind_ig, ind_ig]).reshape(-1)
            σZi, σZg = σZig[:ni], σZig[ni:].reshape((n_xdims, ni)).T
            return (μZi.reshape(-1), μZg), (σZi, σZg)
        elif ret_std == "covar":
            return (μZi.reshape(-1), μZg), Σii
        # Note: cannot untransform when ret_std and ret_grad
    elif not trans_type and f_mean and not basis_type and ret_std and ret_grad:
        # Evaluate prior mean at Xd & Xi
        μYd_prior = f_mean(Xd)
        μYi_prior, μYg_prior = f_mean(Xi, grad=True)
        # Distance & auto-covariance
        σd, w, ℓ = φ['σd'], φ['w'], φ['ℓ']
        Rdd = radius(Xd,  Xd, s)
        Rid = radius(Xi, Xd, s)  # will be use repeatedly for Y & each gradient direction
        Rii = radius(Xi, Xi, s)  # will be use repeatedly for Y & each gradient direction
        Kdd = w**2 * exp(-0.5 * ((Rdd / ℓ)**2).sum(axis=2))
        Kdd[ind_d, ind_d] += σd**2
        Kid = empty((ni * (1 + n_xdims), nd))
        Kid[:ni] = w**2 * exp(-0.5 * ((Rid / ℓ)**2).sum(axis=2))
        for k in range(n_xdims):
            lo, hi = ni * (k + 1), ni * (k + 2)
            Kid[lo:hi] = -Rid[:, :, k] / (s[k] * ℓ[k]**2) * Kid[:ni]
        Kii = empty((ni * (1 + n_xdims), ni * (1 + n_xdims)))
        Kii[:ni, :ni] = w**2 * exp(-0.5 * ((Rii / ℓ)**2).sum(axis=2))
        for k in range(n_xdims):
            lo, hi = ni * (k + 1), ni * (k + 2)
            Kii[lo:hi, :ni] = -Rii[:, :, k] / (s[k] * ℓ[k]**2) * Kii[:ni, :ni]
        Kii[:ni, ni:] = -Kii[ni:, :ni].T
        for ki in range(n_xdims):
            i_lo, i_hi = ni * (ki + 1), ni * (ki + 2)
            for kj in range(n_xdims):
                δ = 1 if ki == kj else 0
                j_lo, j_hi = ni * (kj + 1), ni * (kj + 2)
                Kii[i_lo:i_hi, j_lo:j_hi] = ((δ - Rii[:, :, ki] / ℓ[ki] * Rii[:, :, kj] / ℓ[kj]) *
                                             Kii[:ni, :ni] / (s[ki] * ℓ[ki] * s[kj] * ℓ[kj]))
        # Inference
        Kinv_Zd = solve(Kdd, Yd - μYd_prior)
        if exclude_mean is False:
            μYi = (μYi_prior + Kid[:ni] @ Kinv_Zd)
            μYg = μYg_prior + (Kid[ni:] @ Kinv_Zd).reshape((n_xdims, ni)).T
        else:
            μYi = (Kid[:ni] @ Kinv_Zd)
            μYg = (Kid[ni:] @ Kinv_Zd).reshape((n_xdims, ni)).T
        Σii = Kii - Kid @ solve(Kdd, Kid.T)
        if ret_std is True:
            σYig = sqrt(Σii[ind_ig, ind_ig]).reshape(-1)
            σYi, σYg = σYig[:ni], σYig[ni:].reshape((n_xdims, ni)).T
            return (μYi.reshape(-1), μYg), (σYi.reshape(-1), σYg)
        elif ret_std == "covar":
            return (μYi.reshape(-1), μYg), Σii
    elif trans_type and f_mean and not basis_type and ret_std and ret_grad:
        # Evaluate prior mean at Xd & Xi
        μYd_prior = f_mean(Xd)
        μYi_prior, μYg_prior = f_mean(Xi, grad=True)
        # Move to the transformed space
        Zd = trans(Yd)
        μZd_prior = trans(μYd_prior)
        μZi_prior, μZg_prior = trans(μYi_prior, μYg_prior)
        # Distance & auto-covariance
        σd, w, ℓ = φ['σd'], φ['w'], φ['ℓ']
        Rdd = radius(Xd,  Xd, s)
        Rid = radius(Xi, Xd, s)  # will be use repeatedly for Y & each gradient direction
        Rii = radius(Xi, Xi, s)  # will be use repeatedly for Y & each gradient direction
        Kdd = w**2 * exp(-0.5 * ((Rdd / ℓ)**2).sum(axis=2))
        Kdd[ind_d, ind_d] += σd**2
        Kid = empty((ni * (1 + n_xdims), nd))
        Kid[:ni] = w**2 * exp(-0.5 * ((Rid / ℓ)**2).sum(axis=2))
        for k in range(n_xdims):
            lo, hi = ni * (k + 1), ni * (k + 2)
            Kid[lo:hi] = -Rid[:, :, k] / (s[k] * ℓ[k]**2) * Kid[:ni]
        Kii = empty((ni * (1 + n_xdims), ni * (1 + n_xdims)))
        Kii[:ni, :ni] = w**2 * exp(-0.5 * ((Rii / ℓ)**2).sum(axis=2))
        for k in range(n_xdims):
            lo, hi = ni * (k + 1), ni * (k + 2)
            Kii[lo:hi, :ni] = -Rii[:, :, k] / (s[k] * ℓ[k]**2) * Kii[:ni, :ni]
        Kii[:ni, ni:] = -Kii[ni:, :ni].T
        for ki in range(n_xdims):
            i_lo, i_hi = ni * (ki + 1), ni * (ki + 2)
            for kj in range(n_xdims):
                δ = 1 if ki == kj else 0
                j_lo, j_hi = ni * (kj + 1), ni * (kj + 2)
                Kii[i_lo:i_hi, j_lo:j_hi] = ((δ - Rii[:, :, ki] / ℓ[ki] * Rii[:, :, kj] / ℓ[kj]) *
                                             Kii[:ni, :ni] / (s[ki] * ℓ[ki] * s[kj] * ℓ[kj]))
        # Inference
        Kinv_Zd = solve(Kdd, Zd - μZd_prior)
        if exclude_mean is False:
            μZi = (μZi_prior + Kid[:ni] @ Kinv_Zd)
            μZg = μZg_prior + (Kid[ni:] @ Kinv_Zd).reshape((n_xdims, ni)).T
        else:
            μZi = (Kid[:ni] @ Kinv_Zd)
            μZg = (Kid[ni:] @ Kinv_Zd).reshape((n_xdims, ni)).T
        Σii = Kii - Kid @ solve(Kdd, Kid.T)
        if ret_std is True:
            σZig = sqrt(Σii[ind_ig, ind_ig]).reshape(-1)
            σZi, σZg = σZig[:ni], σZig[ni:].reshape((n_xdims, ni)).T
            return (μZi.reshape(-1), μZg), (σZi.reshape(-1), σZg)
        elif ret_std == "covar":
            return (μZi.reshape(-1), μZg), Σii
            # Note: cannot untransform when ret_std and ret_grad
    elif not trans_type and not f_mean and basis_type == "planar" and ret_std and ret_grad:
        # Explicit bases
        Hd = empty((nd, (1 + n_xdims)), dtype='float64')
        Hd[:, 0] = 1
        Hd[:, 1:] = Xd
        Hig = full((ni, (1 + n_xdims), (1 + n_xdims)), 0, dtype='float64')
        Hi = Hig[:, 0, :]
        Hi[:, 0] = 1
        Hi[:, 1:] = Xi
        Hg = Hig[:, 1:, :]
        for i in range(n_xdims):
            Hg[:, i, i + 1] = 1
        # Distance & auto-covariance
        σd, w, ℓ = φ['σd'], φ['w'], φ['ℓ']
        Rdd = radius(Xd,  Xd, s)
        Rid = radius(Xi, Xd, s)  # will be use repeatedly for Y & each gradient direction
        Rii = radius(Xi, Xi, s)  # will be use repeatedly for Y & each gradient direction
        Kdd = w**2 * exp(-0.5 * ((Rdd / ℓ)**2).sum(axis=2))
        Kdd[ind_d, ind_d] += σd**2
        Kid = empty((ni * (1 + n_xdims), nd))
        Kid[:ni] = w**2 * exp(-0.5 * ((Rid / ℓ)**2).sum(axis=2))
        for k in range(n_xdims):
            lo, hi = ni * (k + 1), ni * (k + 2)
            Kid[lo:hi] = -Rid[:, :, k] / (s[k] * ℓ[k]**2) * Kid[:ni]
        Kii = empty((ni * (1 + n_xdims), ni * (1 + n_xdims)))
        Kii[:ni, :ni] = w**2 * exp(-0.5 * ((Rii / ℓ)**2).sum(axis=2))
        for k in range(n_xdims):
            lo, hi = ni * (k + 1), ni * (k + 2)
            Kii[lo:hi, :ni] = -Rii[:, :, k] / (s[k] * ℓ[k]**2) * Kii[:ni, :ni]
        Kii[:ni, ni:] = -Kii[ni:, :ni].T
        for ki in range(n_xdims):
            i_lo, i_hi = ni * (ki + 1), ni * (ki + 2)
            for kj in range(n_xdims):
                δ = 1 if ki == kj else 0
                j_lo, j_hi = ni * (kj + 1), ni * (kj + 2)
                Kii[i_lo:i_hi, j_lo:j_hi] = ((δ - Rii[:, :, ki] / ℓ[ki] * Rii[:, :, kj] / ℓ[kj]) *
                                             Kii[:ni, :ni] / (s[ki] * ℓ[ki] * s[kj] * ℓ[kj]))
        # Inference
        Σβ_inv = Hd.T @ solve(Kdd, Hd)
        μβ = solve(Σβ_inv, Hd.T @ solve(Kdd, Yd))
        Kinv_Zd = solve(Kdd, Yd - Hd @ μβ)
        if exclude_mean is False:
            μYi = (Hi @ μβ + Kid[:ni] @ Kinv_Zd)
            μYg = Hg @ μβ.reshape(-1) + (Kid[ni:] @ Kinv_Zd).reshape((n_xdims, ni)).T
        else:
            μYi = (Kid[:ni] @ Kinv_Zd)
            μYg = (Kid[ni:] @ Kinv_Zd).reshape((n_xdims, ni)).T
        tmp = Hig.T.reshape(((1 + n_xdims), -1)).T - Kid @ solve(Kdd, Hd)
        Σii = Kii - Kid @ solve(Kdd, Kid.T) + tmp @ solve(Σβ_inv, tmp.T)
        if ret_std is True:
            σYig = sqrt(Σii[ind_ig, ind_ig]).reshape(-1)
            σYi, σYg = σYig[:ni], σYig[ni:].reshape((n_xdims, ni)).T
            return (μYi.reshape(-1), μYg), (σYi.reshape(-1), σYg)
        elif ret_std == "covar":
            return (μYi.reshape(-1), μYg), Σii
    elif trans_type and not f_mean and basis_type == "planar" and ret_std and ret_grad:
        # Move to the transformed space
        Zd = trans(Yd)
        # Explicit bases
        Hd = empty((nd, (1 + n_xdims)), dtype='float64')
        Hd[:, 0] = 1
        Hd[:, 1:] = Xd
        Hig = full((ni, (1 + n_xdims), (1 + n_xdims)), 0, dtype='float64')
        Hi = Hig[:, 0, :]
        Hi[:, 0] = 1
        Hi[:, 1:] = Xi
        Hg = Hig[:, 1:, :]
        for i in range(n_xdims):
            Hg[:, i, i + 1] = 1
        # Distance & auto-covariance
        σd, w, ℓ = φ['σd'], φ['w'], φ['ℓ']
        Rdd = radius(Xd,  Xd, s)
        Rid = radius(Xi, Xd, s)  # will be use repeatedly for Y & each gradient direction
        Rii = radius(Xi, Xi, s)  # will be use repeatedly for Y & each gradient direction
        Kdd = w**2 * exp(-0.5 * ((Rdd / ℓ)**2).sum(axis=2))
        Kdd[ind_d, ind_d] += σd**2
        Kid = empty((ni * (1 + n_xdims), nd))
        Kid[:ni] = w**2 * exp(-0.5 * ((Rid / ℓ)**2).sum(axis=2))
        for k in range(n_xdims):
            lo, hi = ni * (k + 1), ni * (k + 2)
            Kid[lo:hi] = -Rid[:, :, k] / (s[k] * ℓ[k]**2) * Kid[:ni]
        Kii = empty((ni * (1 + n_xdims), ni * (1 + n_xdims)))
        Kii[:ni, :ni] = w**2 * exp(-0.5 * ((Rii / ℓ)**2).sum(axis=2))
        for k in range(n_xdims):
            lo, hi = ni * (k + 1), ni * (k + 2)
            Kii[lo:hi, :ni] = -Rii[:, :, k] / (s[k] * ℓ[k]**2) * Kii[:ni, :ni]
        Kii[:ni, ni:] = -Kii[ni:, :ni].T
        for ki in range(n_xdims):
            i_lo, i_hi = ni * (ki + 1), ni * (ki + 2)
            for kj in range(n_xdims):
                δ = 1 if ki == kj else 0
                j_lo, j_hi = ni * (kj + 1), ni * (kj + 2)
                Kii[i_lo:i_hi, j_lo:j_hi] = ((δ - Rii[:, :, ki] / ℓ[ki] * Rii[:, :, kj] / ℓ[kj]) *
                                             Kii[:ni, :ni] / (s[ki] * ℓ[ki] * s[kj] * ℓ[kj]))
        # Inference
        Σβ_inv = Hd.T @ solve(Kdd, Hd)
        μβ = solve(Σβ_inv, Hd.T @ solve(Kdd, Zd))
        Kinv_Zd = solve(Kdd, Zd - Hd @ μβ)
        if exclude_mean is False:
            μZi = (Hi @ μβ + Kid[:ni] @ Kinv_Zd)
            μZg = Hg @ μβ.reshape(-1) + (Kid[ni:] @ Kinv_Zd).reshape((n_xdims, ni)).T
        else:
            μZi = (Kid[:ni] @ Kinv_Zd)
            μZg = (Kid[ni:] @ Kinv_Zd).reshape((n_xdims, ni)).T
        tmp = Hig.T.reshape(((1 + n_xdims), -1)).T - Kid @ solve(Kdd, Hd)
        Σii = Kii - Kid @ solve(Kdd, Kid.T) + tmp @ solve(Σβ_inv, tmp.T)
        if ret_std is True:
            σZig = sqrt(Σii[ind_ig, ind_ig]).reshape(-1)
            σZi, σZg = σZig[:ni], σZig[ni:].reshape((n_xdims, ni)).T
            return (μZi.reshape(-1), μZg), (σZi.reshape(-1), σZg)
        elif ret_std == "covar":
            return (μZi.reshape(-1), μZg), Σii
            # Note: cannot untransform when ret_std and ret_grad
    elif not trans_type and f_mean and basis_type == "planar" and ret_std and ret_grad:
        # Evaluate prior mean at Xd & Xi
        μYd_prior = f_mean(Xd)
        μYi_prior, μYg_prior = f_mean(Xi, grad=True)
        # Explicit bases
        Hd = empty((nd, (1 + n_xdims)), dtype='float64')
        Hd[:, 0] = 1
        Hd[:, 1:] = Xd
        Hig = full((ni, (1 + n_xdims), (1 + n_xdims)), 0, dtype='float64')
        Hi = Hig[:, 0, :]
        Hi[:, 0] = 1
        Hi[:, 1:] = Xi
        Hg = Hig[:, 1:, :]
        for i in range(n_xdims):
            Hg[:, i, i + 1] = 1
        # Distance & auto-covariance
        σd, w, ℓ = φ['σd'], φ['w'], φ['ℓ']
        Rdd = radius(Xd,  Xd, s)
        Rid = radius(Xi, Xd, s)  # will be use repeatedly for Y & each gradient direction
        Rii = radius(Xi, Xi, s)  # will be use repeatedly for Y & each gradient direction
        Kdd = w**2 * exp(-0.5 * ((Rdd / ℓ)**2).sum(axis=2))
        Kdd[ind_d, ind_d] += σd**2
        Kid = empty((ni * (1 + n_xdims), nd))
        Kid[:ni] = w**2 * exp(-0.5 * ((Rid / ℓ)**2).sum(axis=2))
        for k in range(n_xdims):
            lo, hi = ni * (k + 1), ni * (k + 2)
            Kid[lo:hi] = -Rid[:, :, k] / (s[k] * ℓ[k]**2) * Kid[:ni]
        Kii = empty((ni * (1 + n_xdims), ni * (1 + n_xdims)))
        Kii[:ni, :ni] = w**2 * exp(-0.5 * ((Rii / ℓ)**2).sum(axis=2))
        for k in range(n_xdims):
            lo, hi = ni * (k + 1), ni * (k + 2)
            Kii[lo:hi, :ni] = -Rii[:, :, k] / (s[k] * ℓ[k]**2) * Kii[:ni, :ni]
        Kii[:ni, ni:] = -Kii[ni:, :ni].T
        for ki in range(n_xdims):
            i_lo, i_hi = ni * (ki + 1), ni * (ki + 2)
            for kj in range(n_xdims):
                δ = 1 if ki == kj else 0
                j_lo, j_hi = ni * (kj + 1), ni * (kj + 2)
                Kii[i_lo:i_hi, j_lo:j_hi] = ((δ - Rii[:, :, ki] / ℓ[ki] * Rii[:, :, kj] / ℓ[kj]) *
                                             Kii[:ni, :ni] / (s[ki] * ℓ[ki] * s[kj] * ℓ[kj]))
        # Inference
        Σβ_inv = Hd.T @ solve(Kdd, Hd)
        μβ = solve(Σβ_inv, Hd.T @ solve(Kdd, Yd - μYd_prior))
        Kinv_Zd = solve(Kdd, Yd - (μYd_prior + Hd @ μβ))
        if exclude_mean is False:
            μYi = (μYi_prior + Hi @ μβ + Kid[:ni] @ Kinv_Zd)
            μYg = μYg_prior + Hg @ μβ.reshape(-1) + (Kid[ni:] @ Kinv_Zd).reshape((n_xdims, ni)).T
        else:
            μYi = (Kid[:ni] @ Kinv_Zd)
            μYg = (Kid[ni:] @ Kinv_Zd).reshape((n_xdims, ni)).T
        tmp = Hig.T.reshape(((1 + n_xdims), -1)).T - Kid @ solve(Kdd, Hd)
        Σii = Kii - Kid @ solve(Kdd, Kid.T) + tmp @ solve(Σβ_inv, tmp.T)
        if ret_std is True:
            σYig = sqrt(Σii[ind_ig, ind_ig]).reshape(-1)
            σYi, σYg = σYig[:ni], σYig[ni:].reshape((n_xdims, ni)).T
            return (μYi.reshape(-1), μYg), (σYi.reshape(-1), σYg)
        elif ret_std == "covar":
            return (μYi.reshape(-1), μYg), Σii
    elif trans_type and f_mean and basis_type == "planar" and ret_std and ret_grad:
        # Evaluate prior mean at Xd & Xi
        μYd_prior = f_mean(Xd)
        μYi_prior, μYg_prior = f_mean(Xi, grad=True)
        # Move to the transformed space
        Zd = trans(Yd)
        μZd_prior = trans(μYd_prior)
        μZi_prior, μZg_prior = trans(μYi_prior, μYg_prior)
        # Explicit bases
        Hd = empty((nd, (1 + n_xdims)), dtype='float64')
        Hd[:, 0] = 1
        Hd[:, 1:] = Xd
        Hig = full((ni, (1 + n_xdims), (1 + n_xdims)), 0, dtype='float64')
        Hi = Hig[:, 0, :]
        Hi[:, 0] = 1
        Hi[:, 1:] = Xi
        Hg = Hig[:, 1:, :]
        for i in range(n_xdims):
            Hg[:, i, i + 1] = 1
        # Distance & auto-covariance
        σd, w, ℓ = φ['σd'], φ['w'], φ['ℓ']
        Rdd = radius(Xd,  Xd, s)
        Rid = radius(Xi, Xd, s)  # will be use repeatedly for Y & each gradient direction
        Rii = radius(Xi, Xi, s)  # will be use repeatedly for Y & each gradient direction
        Kdd = w**2 * exp(-0.5 * ((Rdd / ℓ)**2).sum(axis=2))
        Kdd[ind_d, ind_d] += σd**2
        Kid = empty((ni * (1 + n_xdims), nd))
        Kid[:ni] = w**2 * exp(-0.5 * ((Rid / ℓ)**2).sum(axis=2))
        for k in range(n_xdims):
            lo, hi = ni * (k + 1), ni * (k + 2)
            Kid[lo:hi] = -Rid[:, :, k] / (s[k] * ℓ[k]**2) * Kid[:ni]
        Kii = empty((ni * (1 + n_xdims), ni * (1 + n_xdims)))
        Kii[:ni, :ni] = w**2 * exp(-0.5 * ((Rii / ℓ)**2).sum(axis=2))
        for k in range(n_xdims):
            lo, hi = ni * (k + 1), ni * (k + 2)
            Kii[lo:hi, :ni] = -Rii[:, :, k] / (s[k] * ℓ[k]**2) * Kii[:ni, :ni]
        Kii[:ni, ni:] = -Kii[ni:, :ni].T
        for ki in range(n_xdims):
            i_lo, i_hi = ni * (ki + 1), ni * (ki + 2)
            for kj in range(n_xdims):
                δ = 1 if ki == kj else 0
                j_lo, j_hi = ni * (kj + 1), ni * (kj + 2)
                Kii[i_lo:i_hi, j_lo:j_hi] = ((δ - Rii[:, :, ki] / ℓ[ki] * Rii[:, :, kj] / ℓ[kj]) *
                                             Kii[:ni, :ni] / (s[ki] * ℓ[ki] * s[kj] * ℓ[kj]))
        # Inference
        Σβ_inv = Hd.T @ solve(Kdd, Hd)
        μβ = solve(Σβ_inv, Hd.T @ solve(Kdd, Zd - μZd_prior))
        Kinv_Zd = solve(Kdd, Zd - (μZd_prior + Hd @ μβ))
        if exclude_mean is False:
            μZi = (μZi_prior + Hi @ μβ + Kid[:ni] @ Kinv_Zd)
            μZg = μZg_prior + Hg @ μβ.reshape(-1) + (Kid[ni:] @ Kinv_Zd).reshape((n_xdims, ni)).T
        else:
            μZi = (Kid[:ni] @ Kinv_Zd)
            μZg = (Kid[ni:] @ Kinv_Zd).reshape((n_xdims, ni)).T
        tmp = Hig.T.reshape(((1 + n_xdims), -1)).T - Kid @ solve(Kdd, Hd)
        Σii = Kii - Kid @ solve(Kdd, Kid.T) + tmp @ solve(Σβ_inv, tmp.T)
        if ret_std is True:
            σZig = sqrt(Σii[ind_ig, ind_ig]).reshape(-1)
            σZi, σZg = σZig[:ni], σZig[ni:].reshape((n_xdims, ni)).T
            return (μZi.reshape(-1), μZg), (σZi.reshape(-1), σZg)
        elif ret_std == "covar":
            return (μZi.reshape(-1), μZg), Σii
            # Note: cannot untransform when ret_std and ret_grad

def consolidated(Xd, Yd, Xi, φ, s, ret_std, trans_type, untrans,
                f_mean, exclude_mean, basis_type, ret_grad):
    """
    Return identical output values as `gold_standard_GPs` for all combinations of inputs in
    [trans_type, f_mean, basis_type, ret_std, ret_grad] while eliminating all code redundancy.
    This code provides the reference branching logic for a more general & optimized implementation.
    """
    # Options handling & problem setup
    # <the initial part of this code is analogous to what belongs in `__init__` or `_one_time_prep`>
    nd, n_xdims = Xd.shape
    ni = Xi.shape[0]
    ind_d = arange(Xd.shape[0])
    ind_i = arange(Xi.shape[0])
    ind_ig = arange(Xi.shape[0] * (1 + n_xdims))
    if trans_type == "Log":
        trans = lambda y, dy=None: log(y) if dy is None else (log(y), dy / y.reshape((-1, 1)))
        inv_trans = lambda z, dz=None: exp(z) if dz is None else (exp(z), exp(z) * dz)
    elif trans_type == "Logit":
        trans = lambda y, dy=None: (log(y / (1 - y)) if dy is None else
                                   (log(y / (1 - y)),
                                    ((2 - y) / (y * (1 - y))).reshape((-1, 1)) * dy))
        inv_trans = lambda z, dz=None: (1 / (exp(-z) + 1) if dz is None else
                                       (1 / (exp(-z) + 1), exp(z) / (1 + exp(z))**2 * dz))
    elif trans_type == "Probit":
        trans = lambda y, dy=None: (sqrt(2) * erfinv(2 * y - 1) if dy is None else
                                   (sqrt(2) * erfinv(2 * y - 1),
                                   (sqrt(2 * π) * exp(erfinv(2 * y - 1)**2)).reshape((-1, 1)) * dy))
        inv_trans = lambda z, dz=None: ((1 + erf(z / sqrt(2))) / 2 if dz is None else
                                       ((1 + erf(z / sqrt(2))) / 2,
                                        exp(-0.5 * z**2) / sqrt(2 * π) * dz))

    # Raise errors before any calculations are performed (structure parallels testing)
    # <this next block belongs in `__call__`>
    if trans_type and untrans is True:
        if ret_std == "covar":
            raise InputError(error_trans_covar)# , f"{trans_type = }, {untrans = }, {ret_std = }")
        if ret_std and ret_grad:
            raise InputError(error_trans_grad)# , f"{trans_type = }, {untrans = }, {ret_grad = }, {ret_std = }")
        if (f_mean or basis_type is not None) and exclude_mean:
            raise InputError(error_trans_exclude)# , f"{trans_type = }, {untrans = }, {f_mean = }, {basis_type = }, {exclude_mean = }")

    # Evaluate prior mean at Xd & Xi
    if f_mean:
        μYd_prior = f_mean(Xd)
        # <this next block belongs in `__call__`>
        if not ret_grad:
            μYi_prior = f_mean(Xi)
        else:
            μYi_prior, μYg_prior = f_mean(Xi, grad=True)

    # Shift to the transformed space (or use pointers for the identity transformation)
    Zd = trans(Yd) if trans_type else Yd
    if not f_mean:
        Zd_prime = Zd
    else:
        μZd_prior = trans(μYd_prior) if trans_type else μYd_prior
        # <this next block belongs in `__call__`>
        if not ret_grad:
            μZi_prior = trans(μYi_prior) if trans_type else μYi_prior
        else:
            μZi_prior, μZg_prior = (trans(μYi_prior, μYg_prior) if trans_type
                                  else (μYi_prior, μYg_prior))
        Zd_prime = Zd - μZd_prior

    # Evaluate the explicit bases
    if basis_type is not None:
        Hd = empty((nd, (1 + n_xdims)), dtype='float64')
        Hd[:, 0] = 1
        Hd[:, 1:] = Xd
        # <this next block belongs in `__call__`>
        if not ret_grad:
            Hi = empty((ni, (1 + n_xdims)), dtype='float64')
            Hi[:, 0] = 1
            Hi[:, 1:] = Xi
        else:
            Hig = full((ni, (1 + n_xdims), (1 + n_xdims)), 0, dtype='float64')
            Hi = Hig[:, 0, :]
            Hi[:, 0] = 1
            Hi[:, 1:] = Xi
            Hg = Hig[:, 1:, :]
            for i in range(n_xdims):
                Hg[:, i, i + 1] = 1

    # Distance & auto-covariance
    σd, w, ℓ = φ['σd'], φ['w'], φ['ℓ']
    # <the `_id` and `_ii` parts of this next block belong in `kernels`>
    if not ret_grad:
        Rdd = cdist(Xd, Xd, 'seuclidean', V=(s * ℓ)**2)
        Rid = cdist(Xi, Xd, 'seuclidean', V=(s * ℓ)**2)
        Kdd = w**2 * exp(-0.5 * Rdd**2)
        Kdd[ind_d, ind_d] += σd**2
        Kid = w**2 * exp(-0.5 * Rid**2)
        if ret_std:
            Rii = cdist(Xi, Xi, 'seuclidean', V=(s * ℓ)**2)
            Kii = w**2 * exp(-0.5 * Rii**2)
    else:
        Rdd = radius(Xd,  Xd, s)
        Rid = radius(Xi, Xd, s)  # will be use repeatedly for Y & each gradient direction
        Kdd = w**2 * exp(-0.5 * ((Rdd / ℓ)**2).sum(axis=2))
        Kdd[ind_d, ind_d] += σd**2
        Kid = empty((ni * (1 + n_xdims), nd))
        Kid[:ni] = w**2 * exp(-0.5 * ((Rid / ℓ)**2).sum(axis=2))
        for k in range(n_xdims):
            lo, hi = ni * (k + 1), ni * (k + 2)
            Kid[lo:hi] = -Rid[:, :, k] / (s[k] * ℓ[k]**2) * Kid[:ni]
        if ret_std:
            Rii = radius(Xi, Xi, s)  # will be use repeatedly for Y & each gradient direction
            Kii = empty((ni * (1 + n_xdims), ni * (1 + n_xdims)))
            Kii[:ni, :ni] = w**2 * exp(-0.5 * ((Rii / ℓ)**2).sum(axis=2))
            for k in range(n_xdims):
                lo, hi = ni * (k + 1), ni * (k + 2)
                Kii[lo:hi, :ni] = -Rii[:, :, k] / (s[k] * ℓ[k]**2) * Kii[:ni, :ni]
            Kii[:ni, ni:] = -Kii[ni:, :ni].T
            for ki in range(n_xdims):
                i_lo, i_hi = ni * (ki + 1), ni * (ki + 2)
                for kj in range(n_xdims):
                    δ = 1 if ki == kj else 0
                    j_lo, j_hi = ni * (kj + 1), ni * (kj + 2)
                    Kii[i_lo:i_hi, j_lo:j_hi] = ((δ - Rii[:, :, ki] / ℓ[ki] * Rii[:, :, kj] / ℓ[kj]) *
                                                Kii[:ni, :ni] / (s[ki] * ℓ[ki] * s[kj] * ℓ[kj]))

    # Inference
    # <these calculations belong in `_one_time_prep`>
    if basis_type is not None:
        Σβ_inv = Hd.T @ solve(Kdd, Hd)
        μβ = solve(Σβ_inv, Hd.T @ solve(Kdd, Zd_prime))
        Zd_prime -= Hd @ μβ
    Kinv_Zd = solve(Kdd, Zd_prime)
    # <from here on, everything is analogous to what belongs in `__call__`>
    μZi = (Kid[:ni] @ Kinv_Zd)
    if ret_grad:
        μZg = (Kid[ni:] @ Kinv_Zd).reshape((n_xdims, ni)).T
    if not exclude_mean:
        if f_mean:
            μZi += μZi_prior
            if ret_grad:
                μZg += μZg_prior
        if basis_type is not None:
            μZi += Hi @ μβ
            if ret_grad:
                μZg += Hg @ μβ.reshape(-1)
    if ret_std:
        Σii = Kii - Kid @ solve(Kdd, Kid.T)
        if basis_type is not None:
            if not ret_grad:
                tmp = Hi - Kid @ solve(Kdd, Hd)
            else:
                tmp = Hig.T.reshape(((1 + n_xdims), -1)).T - Kid @ solve(Kdd, Hd)
            Σii += tmp @ solve(Σβ_inv, tmp.T)

    # Untransform and return the requested variables (calculating deviations when necessary)
    if not trans_type or not untrans:  # when `not trans_type`, imply the identiy untransform:
        if ret_std is False:
            if ret_grad is False:
                return μZi.reshape(-1)
            else:
                return μZi.reshape(-1), μZg
        elif ret_std is True:
            if ret_grad is False:
                σZi = sqrt(Σii[ind_i, ind_i]).reshape((-1, 1))
                return μZi.reshape(-1), σZi.reshape(-1)
            else:
                σZig = sqrt(Σii[ind_ig, ind_ig]).reshape(-1)
                σZi, σZg = σZig[:ni], σZig[ni:].reshape((n_xdims, ni)).T
                return (μZi.reshape(-1), μZg), (σZi.reshape(-1), σZg)
        elif ret_std == "covar":
            if ret_grad is False:
                return μZi.reshape(-1), Σii
            else:
                return (μZi.reshape(-1), μZg), Σii
    else:
        if ret_std is False:
            warn(warn_trans_μ, RuntimeWarning)
            if ret_grad is False:
                μYi = inv_trans(μZi)
                return μYi.reshape(-1)
            else:
                μYi, μYg = inv_trans(μZi, μZg)
                return μYi.reshape(-1), μYg
        elif ret_std is True:
            warn(warn_trans_μ + warn_trans_σ, RuntimeWarning)
            if ret_grad is False:
                μYi = inv_trans(μZi)
                σZi = sqrt(Σii[ind_i, ind_i]).reshape((-1, 1))
                Zi_lohi = array([(μZi - σZi).reshape(-1), (μZi + σZi).reshape(-1)])
                Yi_lohi = inv_trans(Zi_lohi)
                return μYi.reshape(-1), Yi_lohi - μYi.reshape(-1)
            else:
                raise InputError(error_trans_grad)  # raised above, noted here for logical symmetry
        elif ret_std == "covar":
            raise InputError(error_trans_covar)  # raised above, noted here for logical symmetry


if __name__ == "__main__":
    example_1D(plot=True)
    example_nD(plot=True)
