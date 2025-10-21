import pytest
pytest_param = pytest.mark.parametrize

from numpy import array, empty, full, linspace, logspace, sqrt, exp, log, pi as π
from numpy.random import default_rng
from numpy.testing import assert_allclose
from scipy.special import erfinv
from scipy.stats.qmc import Sobol

from pyregress import (Constant, Normal, Jeffreys, LogNormal, Gamma, Uniform, Beta,
                       radius, SquareExp, Noise, GPI)
from gp_test_utils import example_1D, example_nD

# @pytest_param("prior_type",[Constant, Normal, Jeffreys, LogNormal, Gamma, Uniform, Beta])
@pytest_param("prior_type", [Constant, Normal, Jeffreys, LogNormal])
def test_hyper_priors(prior_type):
    if prior_type == Constant:
        n = 17  # must be odd
        half = logspace(-2, 5, int((n - 1) / 2))
        φ = array([*(-half[::-1]), 0, *half])
        lnp_gs = full(φ.shape, 1)
        prior = prior_type(guess=1)
    elif prior_type == Normal:
        μ, σ = 1.1, 0.65
        n = 11
        φ = μ + σ * sqrt(2) * erfinv(2 * linspace(1 / (2 * n), 1 - 1 / (2 * n), n) - 1)
        lnp_gs = log(1 / sqrt(2 * π * σ**2) * exp(-(φ - μ)**2 / (2 * σ**2)))
        prior = prior_type(μ=μ, σ=σ, guess=μ)
    elif prior_type == Jeffreys:
        n = 9
        φ = logspace(-4, 4, n)
        lnp_gs = log(1 / φ)
        prior = prior_type(guess=1)
    elif prior_type == LogNormal:
        μ, σ = 1.1, 0.65
        n = 11
        φ = exp(μ + σ * sqrt(2) * erfinv(2 * linspace(1 / (2 * n), 1 - 1 / (2 * n), n) - 1))
        lnp_gs = log(1 / (φ * sqrt(2 * π * σ**2)) * exp(-(log(φ) - μ)**2 / (2 * σ**2)))
        prior = prior_type(μ=μ, σ=σ, guess=exp(μ))
    
    lnp = empty(n)
    for i in range(n):
        lnp[i] = prior(φ[i])
    assert_allclose(lnp, lnp_gs)

@pytest_param("kernel_opts", [
    dict(params=dict(w=2, l=0.5),
        kernel=(lambda w, l: SquareExp(w=w, l=l))),
    dict(params=dict(w=2, l=0.5),
        kernel=(lambda w, l: SquareExp(w=Jeffreys(guess=w), l=l))),
    dict(params=dict(w=2, l=0.5),
        kernel=(lambda w, l: SquareExp(w=w, l=Jeffreys(guess=l)))),
    dict(params=dict(w=2, l=0.5),
        kernel=(lambda w, l: SquareExp(w=Jeffreys(guess=w), l=Jeffreys(guess=l)))),
    dict(params=dict(w=2, l=[0.3, 0.8]),
         kernel=(lambda w, l: SquareExp(w=w, l=l))),
    dict(params=dict(w=2, l=[0.3, 0.8]),
         kernel=(lambda w, l: SquareExp(w=w, l=[Jeffreys(guess=l[0]), l[1]]))),
    dict(params=dict(w=2, l=[0.3, 0.8]),
         kernel=(lambda w, l: SquareExp(w=w, l=[Jeffreys(guess=l[0]), Jeffreys(guess=l[1])]))),
    # dict(params=dict(Noise=dict(w=0.03), SquareExp=dict(w=2, l=0.5)),
    #      kernel=(lambda σ, w, l: Noise(w=) + SquareExp(w=w, l=l)))
    ])
def test_kernel_params(kernel_opts):
    # Test that kernel params are initialized correctly:
    kernel_params = kernel_opts['params']
    my_kernel = kernel_opts['kernel'](**kernel_params)
    assert kernel_params == my_kernel.param_vals
    # # Test that the kernel params can be optimized:
    # if my_kernel.n_φ > 0:
    #     ln_n = 6
    #     Xd, Yd, Xi, *_ = example_nD(dims=2, log2_n_pts=ln_n, ni=(4, 5))
    #     my_gpi = GPI(Xd, Yd, my_kernel, optimize=False)
    #     my_opt = GPI(Xd, Yd, my_kernel, optimize=True)
    #     assert my_opt.posterior_φ(my_opt.kernel.get_φ()) < my_gpi.posterior_φ(my_gpi.kernel.get_φ())

