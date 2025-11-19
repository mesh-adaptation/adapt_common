"""Test the conversions between Cofunctions and Functions in the utility module."""

import firedrake as fd
import numpy as np
import pytest

from adapt_common.utility import cofunction2function, function2cofunction


@pytest.fixture(params=["simple", "mixed"])
def function_space(request):
    """Return a function space for testing."""
    V = fd.FunctionSpace(fd.UnitSquareMesh(12, 12), "CG", 1)
    return V if request.param == "simple" else fd.MixedFunctionSpace([V, V])


def set_random_data(function_or_cofunction):
    """Set random data in the given Function or Cofunction."""
    rng = np.random.default_rng()
    if isinstance(function_or_cofunction.function_space().dof_count, np.int64):
        size = len(function_or_cofunction.dat.data_with_halos)
        function_or_cofunction.dat.data_with_halos[:] = rng.random(size)
    else:
        for i, data in enumerate(function_or_cofunction.dat.data_with_halos):
            function_or_cofunction.dat.data_with_halos[i][:] = rng.random(len(data))
    assert not np.allclose(function_or_cofunction.dat.data_with_halos, 0)


def test_cofunction2function_function_space(function_space):
    """Test that the converted Function has the expected function space."""
    cofunc = fd.Cofunction(function_space.dual())
    func = cofunction2function(cofunc)
    assert func.function_space() == function_space
    assert func.function_space().dual() == cofunc.function_space()


def test_cofunction2function_data(function_space):
    """Test that the data in the Cofunction is correctly transferred to the Function."""
    cofunc = fd.Cofunction(function_space.dual())
    set_random_data(cofunc)
    func = cofunction2function(cofunc)
    assert np.allclose(func.dat.data, cofunc.dat.data)


@pytest.mark.parallel(nprocs=[2, 3])
def test_cofunction2function_data_with_halos(function_space):
    """Test that the Cofunction data with halos is correctly transferred."""
    cofunc = fd.Cofunction(function_space.dual())
    set_random_data(cofunc)
    func = cofunction2function(cofunc)
    assert np.allclose(func.dat.data_with_halos, cofunc.dat.data_with_halos)


def test_function2cofunction_function_space(function_space):
    """Test that the converted Cofunction has the expected function space."""
    cofunc = function2cofunction(fd.Function(function_space))
    assert function_space.dual() == cofunc.function_space()


def test_function2cofunction_data(function_space):
    """Test that the data in the Function is correctly transferred to the Cofunction."""
    func = fd.Function(function_space)
    set_random_data(func)
    cofunc = function2cofunction(func)
    assert np.allclose(func.dat.data, cofunc.dat.data)


@pytest.mark.parallel(nprocs=[2, 3])
def test_function2cofunction_data_with_halos(function_space):
    """Test that the Function data with halos is correctly transferred."""
    func = fd.Function(function_space)
    set_random_data(func)
    cofunc = function2cofunction(func)
    assert np.allclose(func.dat.data_with_halos, cofunc.dat.data_with_halos)
