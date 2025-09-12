"""
Use the gold-standards in gp_test_utils to perform comprehensive system-coverage regression testing
and verification of `GPI.__call__` can be compared.
"""
import re
import pytest
pytest_param = pytest.mark.parametrize
from numpy import array, full
from numpy.testing import assert_allclose

from gp_test_utils import example_1D, example_nD, gold_standard_GPs, consolidated
from pyregress import GPI, Noise, SquareExp, Logarithm, Logit, Probit, InputError
from pyregress.gaussian_processes.gaussian_process import (warn_trans_μ, warn_trans_σ,
                                        error_trans_exclude, error_trans_covar, error_trans_grad)


# Task list:
# - consider adding ranges to scale the transformations,
# - add unit testing for the transformations,
# - consider adding an inner-percentile option to `infer_std` (for the `untranspose` case),
# - consider adding gradients to the bases in lin_regress incorporating those as explicit_bases in gaussian_process,
# - add unit testing for basis functions,
# - consider adding a parametrization for `kernel_terms`,
# - add unit testing for the kernels & gradients,
# - consider re-adding the basis when excluding the mean (question why that option was available in the first place),


rel_tol = 1e-6
abs_tol = 1e-6

@pytest_param("example_type", ["1D", "2D"])
@pytest_param("scale_type", [None, "manual"])
@pytest_param("trans_ret", [(None, True), ("Log", True), ("Log", False)])  # (trans., untrans.)
@pytest_param("mean_flags", [(None, False), (True, False), (True, True)])  # (prior, exclude_mean)
@pytest_param("basis_type", [None, "planar"])  # TODO: needs work (pull from linregress.py)
@pytest_param("ret_std", [False, True, "covar"])
@pytest_param("ret_grad", [False, True])
def test_GPI(example_type, scale_type, trans_ret, mean_flags, basis_type, ret_std, ret_grad):
    """
    Provide comprehensive regression testing by checking coverage of all combinations of input
    option groups, by checking output types (inc. warnings & errors), and by verifying values of
    all outputs against manually coded gold standards of the full system.
    """
    # Options handling & problem setup
    if example_type == "1D":
        example_generator = example_1D
        ni = 7
    elif example_type == "2D":
        example_generator = example_nD
        ni = (3, 5)
    Xd, Yd, Xi, φ, f_mean = example_generator(ni=ni)
    nd, n_xdims = Xd.shape
    if scale_type is None:
        scale = full(n_xdims, 1)
    else:
        if scale_type == "range":
            scale = Xd.max(axis=0) - Xd.min(axis=0)
        elif scale_type == "std":
            scale = Xd.std(axis=0)
        elif scale_type == "manual":
            if example_type == "1D":
                scale_type = scale = array([20.5])
            elif example_type == "2D":
                scale_type = scale = full(n_xdims, 1.01)
        φ['ℓ'] = φ['ℓ'] / scale
    trans_type, untrans = trans_ret
    trans = {None:None, "Log":Logarithm(), "Logit":Logit(), "Probit":Probit()}[trans_type]
    prior_flag, exclude_mean = mean_flags
    if prior_flag is None:
        f_mean = None
    bases = {None:None, "planar":[0, 1]}[basis_type]
    K = Noise(φ['σd']) + SquareExp(w=φ['w'], l=φ['ℓ'])

    # Retrieve the gold standard for these options & setup for the routine being tested
    test_opts = (Xd, Yd, Xi, φ, scale, ret_std, trans_type, untrans,
                     f_mean, exclude_mean, basis_type, ret_grad)
    out_gs = gold_standard_GPs(*test_opts)
    my_gpi = GPI(Xd, Yd, K, scale_type, f_mean, bases, trans)

    # Compare
    if trans_type is not None and untrans is True:
        if ret_std == "covar":
            with pytest.raises(InputError, match=error_trans_covar):
                # consolidated(*test_opts)
                my_gpi(Xi, 'noisefree', ret_std, exclude_mean, untrans, ret_grad)
            out = None
        elif ret_std is not False and ret_grad:
            with pytest.raises(InputError, match=error_trans_grad):
                # consolidated(*test_opts)
                my_gpi(Xi, 'noisefree', ret_std, exclude_mean, untrans, ret_grad)
            out = None
        elif (prior_flag is not None or basis_type is not None) and exclude_mean:
            with pytest.raises(InputError, match=error_trans_exclude):
                # consolidated(*test_opts)
                my_gpi(Xi, 'noisefree', ret_std, exclude_mean, untrans, ret_grad)
            out = None
        elif ret_std is False:
            with pytest.warns(RuntimeWarning, match=re.escape(warn_trans_μ)):
                # out = consolidated(*test_opts)
                out = my_gpi(Xi, 'noisefree', ret_std, exclude_mean, untrans, ret_grad)
        elif ret_std is True:
            with pytest.warns(RuntimeWarning, match=re.escape(warn_trans_μ + warn_trans_σ)):
                # out = consolidated(*test_opts)
                out = my_gpi(Xi, 'noisefree', ret_std, exclude_mean, untrans, ret_grad)
    else:
        # out = consolidated(*test_opts)
        out = my_gpi(Xi, 'noisefree', ret_std, exclude_mean, untrans, ret_grad)
    if out is not None:
        assert isinstance(out, type(out_gs))
        if ret_std is False:
            if ret_grad is False:
                μYi, μYi_gs = out, out_gs
                assert μYi.shape == μYi_gs.shape
                assert_allclose(μYi, μYi_gs, rtol=rel_tol, atol=abs_tol)
            else:
                μYi, μδYi = out
                μYi_gs, μδYi_gs = out_gs
                assert isinstance(μYi, type(μYi_gs))
                assert isinstance(μδYi, type(μδYi_gs))
                assert μYi.shape == μYi_gs.shape
                assert μδYi.shape == μδYi_gs.shape
                assert_allclose(μYi, μYi_gs, rtol=rel_tol, atol=abs_tol)
                assert_allclose(μδYi, μδYi_gs, rtol=rel_tol, atol=abs_tol)
        elif ret_std is True:
            if ret_grad is False:
                μYi, σYi = out
                μYi_gs, σYi_gs = out_gs
                assert isinstance(μYi, type(μYi_gs))
                assert isinstance(σYi, type(σYi_gs))
                assert μYi.shape == μYi_gs.shape
                assert σYi.shape == σYi_gs.shape
                assert_allclose(μYi, μYi_gs, rtol=rel_tol, atol=abs_tol)
                assert_allclose(σYi, σYi_gs, rtol=rel_tol, atol=abs_tol)
            else:
                (μYi, μδYi), (σYi, σδYi) = out
                (μYi_gs, μδYi_gs), (σYi_gs, σδYi_gs) = out_gs
                assert isinstance(μYi, type(μYi_gs))
                assert isinstance(μδYi, type(μδYi_gs))
                assert isinstance(σYi, type(σYi_gs))
                assert isinstance(σδYi, type(σδYi_gs))
                assert μYi.shape == μYi_gs.shape
                assert μδYi.shape == μδYi_gs.shape
                assert σYi.shape == σYi_gs.shape
                assert σδYi.shape == σδYi_gs.shape
                assert_allclose(μYi, μYi_gs, rtol=rel_tol, atol=abs_tol)
                assert_allclose(μδYi, μδYi_gs, rtol=rel_tol, atol=abs_tol)
                assert_allclose(σYi, σYi_gs, rtol=rel_tol, atol=abs_tol)
                assert_allclose(σδYi, σδYi_gs, rtol=rel_tol, atol=abs_tol)
        elif ret_std == "covar":
            if ret_grad is False:
                μYi, ΣYi = out
                μYi_gs, ΣYi_gs = out_gs
                assert isinstance(μYi, type(μYi_gs))
                assert isinstance(ΣYi, type(ΣYi_gs))
                assert μYi.shape == μYi_gs.shape
                assert ΣYi.shape == ΣYi_gs.shape
                assert_allclose(μYi, μYi_gs, rtol=rel_tol, atol=abs_tol)
                assert_allclose(ΣYi, ΣYi_gs, rtol=rel_tol, atol=abs_tol)
            else:
                (μYi, μδYi), ΣYi = out
                (μYi_gs, μδYi_gs), ΣYi_gs = out_gs
                assert isinstance(μYi, type(μYi_gs))
                assert isinstance(μδYi, type(μδYi_gs))
                assert isinstance(ΣYi, type(ΣYi_gs))
                assert μYi.shape == μYi_gs.shape
                assert μδYi.shape == μδYi_gs.shape
                assert ΣYi.shape == ΣYi_gs.shape
                assert_allclose(μYi, μYi_gs, rtol=rel_tol, atol=abs_tol)
                assert_allclose(μδYi, μδYi_gs, rtol=rel_tol, atol=abs_tol)
                assert_allclose(ΣYi, ΣYi_gs, rtol=rel_tol, atol=abs_tol)
