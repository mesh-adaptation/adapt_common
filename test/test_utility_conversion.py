"""Test the conversions between Cofunctions and Functions in the utility module."""

import firedrake as fd
import numpy as np
import pytest

from adapt_common.utility import cofunction2function, function2cofunction


def test_cofunction2function_function_space():
    """Test that the converted Function has the expected function space."""
    V = fd.FunctionSpace(fd.UnitSquareMesh(1, 1), "CG", 1)
    cofunc = fd.Cofunction(V.dual())
    func = cofunction2function(cofunc)
    assert func.function_space() == V
    assert func.function_space().dual() == cofunc.function_space()


def test_cofunction2function_data():
    """Test that the data in the Cofunction is correctly transferred to the Function."""
    V = fd.FunctionSpace(fd.UnitSquareMesh(4, 4), "CG", 1)
    cofunc = fd.Cofunction(V.dual())
    cofunc.dat.data[:] = np.arange(len(cofunc.dat.data))
    func = cofunction2function(cofunc)
    assert np.allclose(func.dat.data, cofunc.dat.data)


@pytest.mark.parallel(nprocs=[2, 3])
def test_cofunction2function_data_with_halos():
    """Test that the Cofunction data with halos is correctly transferred."""
    V = fd.FunctionSpace(fd.UnitSquareMesh(16, 16), "CG", 1)
    cofunc = fd.Cofunction(V.dual())
    cofunc.dat.data_with_halos[:] = np.arange(len(cofunc.dat.data_with_halos))
    func = cofunction2function(cofunc)
    assert np.allclose(func.dat.data_with_halos, cofunc.dat.data_with_halos)


def test_function2cofunction_function_space():
    """Test that the converted Cofunction has the expected function space."""
    V = fd.FunctionSpace(fd.UnitSquareMesh(1, 1), "CG", 1)
    cofunc = function2cofunction(fd.Function(V))
    assert V.dual() == cofunc.function_space()


def test_function2cofunction_data():
    """Test that the data in the Function is correctly transferred to the Cofunction."""
    V = fd.FunctionSpace(fd.UnitSquareMesh(4, 4), "CG", 1)
    func = fd.Function(V)
    func.dat.data[:] = np.arange(len(func.dat.data))
    cofunc = function2cofunction(func)
    assert np.allclose(func.dat.data, cofunc.dat.data)


@pytest.mark.parallel(nprocs=[2, 3])
def test_function2cofunction_data_with_halos():
    """Test that the Function data with halos is correctly transferred."""
    V = fd.FunctionSpace(fd.UnitSquareMesh(16, 16), "CG", 1)
    func = fd.Function(V)
    func.dat.data_with_halos[:] = np.arange(len(func.dat.data_with_halos))
    cofunc = function2cofunction(func)
    assert np.allclose(func.dat.data_with_halos, cofunc.dat.data_with_halos)
