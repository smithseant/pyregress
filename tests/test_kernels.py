import pytest
pytest_param = pytest.mark.parametrize
assert_close = pytest.approx

from numpy import array, empty, full, linspace, logspace, sqrt, exp, log, pi as π, inf
from numpy.random import default_rng
from numpy.testing import assert_allclose
from scipy.special import erfinv
from scipy.stats.qmc import Sobol

from pyregress import (Uniform, Jeffreys, Normal, LogNormal, Gamma, Beta,
                       radius, SquareExp, Noise, GPI)
from gp_test_utils import example_1D, example_nD

def arg_helper(kw, pos, default, *args, **kwargs):
    if kw in kwargs:
        val = kwargs[kw]
    elif len(args) - 1 >= pos:
        val = args[pos]
    else:
        val = default
    return val

# hint: `uv run --extra dev pytest -x tests/test_kernels.py::test_hyper_priors`
@pytest_param("prior_cls,args,kwargs", [
    # test cases:
    pytest.param(Uniform, (), {}, id="Uniform()"),
    pytest.param(Uniform, (), dict(guess=0.1), id="Uniform(0.1)"),
    pytest.param(Uniform, (), dict(bounds=(-2, 3), guess=0.9), id="Uniform((-2, 3), 0.9)"),
    pytest.param(Uniform, (), dict(bounds=(0, inf), guess=1.1), id="Uniform((0, inf), 1.1)"),
    # pytest.param(Uniform, (2, 'probit'), dict(bounds=(1, 3)), id="Uniform(2, 'probit', (1, 3))"),
    pytest.param(Jeffreys, (), dict(trans=None), id="Jeffreys()"),
    pytest.param(Jeffreys, (2,), dict(trans=None), id="Jeffreys(2)"),
    pytest.param(Jeffreys, (2, 'log'), {}, id="Jeffreys(2, 'log')"),
    pytest.param(Normal, (), {}, id="Normal()"),
    pytest.param(Normal, (), dict(guess=10), id="Normal(10)"),
    pytest.param(Normal, (), dict(guess=10, μ=9, σ=4), id="Normal(10, 9, 4)"),
    pytest.param(LogNormal, (), dict(trans=None), id="LogNormal()"),
    pytest.param(LogNormal, (), dict(guess=0.1, trans=None), id="LogNormal(0.1)"),
    pytest.param(LogNormal, (), dict(guess=0.1, μ=-2, σ=0.65, trans=None), id="LogNormal(0.1, -2, 0.65)"),
    # pytest.param(Gamma, (), {}, id="Gamma()"),
    # pytest.param(Beta, (), {}, id="Beta()"),
    ])
def test_hyper_priors(prior_cls, args, kwargs):
    """
    Test whether hyper-parameter priors created manually behave as expected
    (for `__init__` and `ln_pdf`).
    """
    if prior_cls == Uniform:
        φ_lo, φ_hi = arg_helper("bounds", 0, (0, 1), *args, **kwargs)
        φ_lo = max(φ_lo, -9e15)
        φ_hi = min(φ_hi, +9e15)
        guess = arg_helper("guess", 1, 0, *args, **kwargs)
        mid_point = (φ_lo + φ_hi) / 2
        half_width = (φ_hi - φ_lo) / 2
        half_grid = array([*logspace(-2, 0, 5), 1.1])
        φ = half_width * array([*(-half_grid[::-1]), 0, *half_grid]) + mid_point
        lnp_gs = full(φ.shape, -log(φ_hi - φ_lo))
        lnp_gs[[0, -1]] = -inf
        prior = prior_cls(*args, **kwargs)
    elif prior_cls == Jeffreys:
        guess = arg_helper("guess", 0, 1, *args, **kwargs)
        trans = arg_helper("trans", 1, "log", *args, **kwargs)
        n = 9
        if trans is None:
            φ = logspace(-4, 4, n)
            lnp_gs = -log(φ)
        elif trans == "log":
            half_grid = logspace(-2, 0, 6)
            φ = 9e15 * array([*(-half_grid[::-1]), 0, *half_grid])
            lnp_gs = full(φ.shape, -log(18e15))
        prior = prior_cls(*args, **kwargs)
    elif prior_cls == Normal:
        guess = arg_helper("guess", 0, 0, *args, **kwargs)
        μ = arg_helper("μ", 1, 0, *args, **kwargs)
        σ = arg_helper("σ", 2, 1, *args, **kwargs)
        trans = arg_helper("trans", 3, None, *args, **kwargs)
        n = 11
        φ = μ + σ * sqrt(2) * erfinv(2 * linspace(1 / (2 * n), 1 - 1 / (2 * n), n) - 1)
        lnp_gs = log(1 / sqrt(2 * π * σ**2) * exp(-(φ - μ)**2 / (2 * σ**2)))
        prior = prior_cls(μ=μ, σ=σ, guess=μ)
    elif prior_cls == LogNormal:
        guess = arg_helper("guess", 0, 0, *args, **kwargs)
        μ = arg_helper("μ", 1, 0, *args, **kwargs)
        σ = arg_helper("σ", 2, 1, *args, **kwargs)
        trans = arg_helper("trans", 3, "log", *args, **kwargs)
        n = 11
        φ = exp(μ + σ * sqrt(2) * erfinv(2 * linspace(1 / (2 * n), 1 - 1 / (2 * n), n) - 1))
        lnp_gs = log(1 / (φ * sqrt(2 * π * σ**2)) * exp(-(log(φ) - μ)**2 / (2 * σ**2)))
        prior = prior_cls(guess=guess, μ=μ, σ=σ, trans=trans)
    
    assert_close(guess, prior.guess)
    lnp = empty(φ.shape)
    for i in range(φ.size):
        lnp[i] = prior.ln_pdf(φ[i])
    assert_allclose(lnp, lnp_gs)

@pytest_param("kernel_opts", [
    dict(params=dict(σ=2, l=0.5),
         kernel=(lambda σ, l: SquareExp(σ=σ, l=l))),
    dict(params=dict(σ=2, l=0.5),
         kernel=(lambda σ, l: SquareExp(σ=Jeffreys(guess=σ), l=l))),
    dict(params=dict(σ=2, l=0.5),
         kernel=(lambda σ, l: SquareExp(σ=σ, l=Jeffreys(guess=l)))),
    dict(params=dict(σ=2, l=0.5),
         kernel=(lambda σ, l: SquareExp(σ=Jeffreys(guess=σ), l=Jeffreys(guess=l)))),
    dict(params=dict(σ=2, l=[0.3, 0.8]),
         kernel=(lambda σ, l: SquareExp(σ=σ, l=l))),
    dict(params=dict(σ=2, l=[0.3, 0.8]),
         kernel=(lambda σ, l: SquareExp(σ=σ, l=[Jeffreys(guess=l[0]), l[1]]))),
    dict(params=dict(σ=2, l=[0.3, 0.8]),
         kernel=(lambda σ, l: SquareExp(σ=σ, l=[Jeffreys(guess=l[0]), Jeffreys(guess=l[1])]))),
    # dict(params=dict(Noise=dict(σ=0.03), SquareExp=dict(σ=2, l=0.5)),
    #      kernel=(lambda σ, σ, l: Noise(σ=) + SquareExp(σ=σ, l=l)))
    ])
def test_kernel_params(kernel_opts):
    """Test whether known kernel params are initialized correctly."""
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

@pytest_param("prior", [])
def test_prior_within_kernel(prior, kernel_type):
    """Test whether priors for unknown kernel params are created correctly."""
    pass

