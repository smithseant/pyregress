from numpy import (array, empty, full, ones, eye, arange, outer, sqrt)
from numpy.random import default_rng
my_rng = default_rng()
uniform, std_norm, gamma, beta = my_rng.random, my_rng.standard_normal, my_rng.gamma, my_rng.beta
from scipy.special import erfinv
from scipy.stats import ortho_group as orthonormal
from scipy.stats.qmc import Sobol
from scipy.linalg import cho_solve
from scipy.linalg.lapack import dtrtri

import matplotlib.pyplot as plt

from pyregress import OrdLinRegress, PolySet

def normal_affine(z, μ, Λ, V):
    return ((V * sqrt(Λ)) @ z.T).T + μ

# Specify the problem & design type:
n_xdims = 25  # number of dimensions
by2_steps = 10
xd_pdf = 'normal'  # 'uniform' or 'normal'
SobolSampler = Sobol(d=n_xdims, scramble=True, optimization='random-cd')
if xd_pdf == 'normal':
    Vx = orthonormal.rvs(n_xdims)
    Λx = beta(0.5, 2, n_xdims)
    Λx[0] = 1.0
    Λx.sort()
    Λx /= 2

# Specify 'unknown' parameters (for generating expt. data):
_H = PolySet(n_xdims, 1)  # list of basis functions
_nH = _H.n_bases  # number of bases
_ϵ_pdf = 'normal'  # 'normal' or 'student-t'
_β = std_norm(_nH)
if _ϵ_pdf == 'normal':
    _σ = 1.4e-3
elif y_pdf == 'student-t':
    _s2 = 0.14**2
    _ν = 4

# Specify the inference model:
H = PolySet(n_xdims, 1)
nH = H.n_bases
n_σ2 = 10_000
p = array([0.05, 0.50, 0.95])

# Generate the design:
start_log2_nd = 1  # round(log2(nH) + 0.5)
end_log2_nd = int(start_log2_nd + by2_steps)
nd = 2**(arange(start_log2_nd, end_log2_nd + 1))
Xd_all = SobolSampler.random_base2(end_log2_nd)

# Generate the experimental data:
if _ϵ_pdf == 'student-t':
    _σ = sqrt(1 / gamma(_ν / 2, 2 / (_ν * _s2), nd[-1]))
Yd_all = _H(Xd_all) @ _β + _σ * std_norm(nd[-1])

# Generate design & data for blind comparisons:
SobolSampler.reset()
Xi = SobolSampler.random_base2(8)
if xd_pdf == 'normal':
    Xi = normal_affine(Xi, full(n_xdims, 0.5), Λx, Vx)
_Ht = _H(Xi)
if _ϵ_pdf == 'student-t':
    _σ = sqrt(1 / gamma(_ν / 2, 2 / (_ν * _s2), nd[-1]))
Yt = _Ht @ _β + _σ * std_norm(Xi.shape[0])

# Setup for calculating the inferred regression error, the L2 norm & blind comparison:
σ_infer = empty(((by2_steps + 1), 3))
σ_flag = full(by2_steps + 1, False)
if xd_pdf == 'uniform':
    int_coefs = ones((nH, nH))
    int_coefs[1:, 1:] /= 4
    int_coefs[arange(1, nH), arange(1, nH)] *= 4 / 3
    int_coefs[0, 1:] /= 2
    int_coefs[1:, 0] /= 2
    nf_err_infer = empty(by2_steps)
    nf_err_L2 = empty(by2_steps)
err_blind = empty(by2_steps + 1)
nf_err_blind = empty(by2_steps + 1)

regressions = []
for i in range(by2_steps + 1):
    # subset the 'observed' data:
    Xd = Xd_all[:nd[i]]
    Yd = Yd_all[:nd[i]]
    # calculate the linear regression:
    my_regress = OrdLinRegress(Xd, Yd, H, λ=(1e-7 / _σ**2))
    regressions.append(my_regress)
    Yi = my_regress(Xi)
    # σ_infer[i, 1] = sqrt(my_regress.σ2_mean) if i > 0 else 0
    err_blind[i] = (Yt - Yi).std()
    # infer the experimental error (by sampling):
    if my_regress.ν > 0:
        σ_flag[i] = True
        _, σ2 = my_regress.sample_βσ2(n_σ2)
        σ2.sort()
        σ_infer[i] = sqrt(σ2[int(n_σ2 * p[0])]), sqrt(σ2[int(n_σ2 * p[1])]), sqrt(σ2[int(n_σ2 * p[2])])
        σ_infer[i, 1] = sqrt(my_regress.σ2_median)
    # calculate the noise-free regression errors:
    if hasattr(my_regress, "eigΛV"):  # ...using Hermitian eigendecomposition
        Λ, V = my_regress.eigΛV
        Vβ = (V / Λ) @ V.T
    elif hasattr(my_regress, "choL"):  # ...using Cholesky decomposition
        Vβ = cho_solve(my_regress.choL, eye(nH), check_finite=False)
    elif hasattr(my_regress, "QR"):  # ...using QR decomposition
        Q, R = my_regress.QR
        Rinv = dtrtri(R)[0]
        Vβ = Rinv @ Rinv.T
    elif hasattr(my_regress, "SVD"):  # ...using SVD
        U, s, V = my_regress.SVD
        VTSinv = V.T / s
        Vβ = VTSinv @ VTSinv.T
    nf_err_blind[i] = sqrt(((_Ht @ (_β - my_regress.μβ))**2).mean())
    if xd_pdf == 'uniform':    
        nf_err_infer[i] = sqrt((my_regress.s2 * Vβ * int_coefs).sum())  # TODO: include the student-t quantile factor
        nf_err_L2[i] = sqrt((outer(my_regress.μβ - _β, my_regress.μβ - _β) * int_coefs).sum())

plt.figure(figsize=(8, 6))
plt.loglog(nd, err_blind, color='tab:blue', label='RMS of blind comparison')
plt.loglog(nd, nf_err_blind, '--', color='tab:blue', label='RMS noise-free comparison')
if xd_pdf == 'uniform':
    plt.loglog(nd, err_infer, '--', color='tab:green', label='inferred noise-free error')
    plt.loglog(nd, err_L2, '--', color='black', linestyle='--', label='L$_2$ noise-free error')
y_lim = plt.ylim()
if _ϵ_pdf == 'normal':
    y_lim = (min(y_lim[0], 0.5 * _σ), y_lim[1])
plt.plot([nH, nH], y_lim, color='tab:red', linewidth=1, label='saturation', zorder=-10)
plt.loglog(nd[σ_flag], σ_infer[σ_flag, 1], color='tab:green', linewidth=0.5, label='inferred expt. error')
plt.fill_between(nd[σ_flag], σ_infer[σ_flag, 0], σ_infer[σ_flag, 2], color='tab:green', alpha=0.35,
                 label=f'inferred expt. error, {100 * (p[2] - p[0]):.0f}%')
if _ϵ_pdf == 'normal':
    plt.plot([nd[0], nd[-1]], [_σ, _σ], color='black', linewidth=1, label='expt. error, $\sigma$ (unknown)', zorder=-9)
plt.xlim(nd[0], nd[-1])
plt.ylim(y_lim)
plt.grid(True, alpha=0.2)
plt.xlabel('No. of training points', fontsize=16)
plt.ylabel('Error', fontsize=16)
plt.legend(fontsize=12)

plt.show()