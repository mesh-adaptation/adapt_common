"""Test the VTKFile class from the utility module."""

import os

import firedrake as fd
import pytest

from adapt_common.utility import VTKFile


@pytest.fixture
def vtk_test_setup(tmp_path):
    """Return a function space and a temporary filename for VTKFile tests."""
    fs = fd.FunctionSpace(fd.UnitSquareMesh(1, 1), "CG", 1)
    fname = tmp_path / "tmp.pvd"
    yield fs, fname


def test_adaptive(vtk_test_setup):
    """Test that VTKFile is set to adaptive mode."""
    _, fname = vtk_test_setup
    file = VTKFile(fname)
    assert os.path.exists(fname)
    assert file.adaptive


def test_different_fnames(vtk_test_setup):
    """Test that VTKFile raises an error for different filenames."""
    fs, fname = vtk_test_setup
    f = fd.Function(fs, name="f")
    g = fd.Function(fs, name="g")
    file = VTKFile(fname)
    file.write(f)
    file.write(g)
    assert g.name() == "f"


def test_different_lengths(vtk_test_setup):
    """Test that VTKFile raises an error for different number of functions."""
    fs, fname = vtk_test_setup
    f = fd.Function(fs, name="f")
    g = fd.Function(fs, name="g")
    file = VTKFile(fname)
    file.write(f)
    msg = "Writing different number of functions: expected 1, got 2."
    with pytest.raises(ValueError, match=msg):
        file.write(f, g)
