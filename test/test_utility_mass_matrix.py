"""Test the assemble_mass_matrix function from the utility module."""

import firedrake as fd
import numpy as np
import pytest

from adapt_common.utility import assemble_mass_matrix


@pytest.mark.parametrize("norm_type", ["L2", "H1"])
def test_tiny(norm_type):
    """Test assemble_mass_matrix gives the expected result on a tiny mesh."""
    V = fd.FunctionSpace(fd.UnitSquareMesh(1, 1), "DG", 0)
    matrix = assemble_mass_matrix(V, norm_type=norm_type)
    assert np.allclose(np.eye(2) / 2, matrix.convert("dense").getDenseArray())


def test_norm_type_error():
    """Test that an invalid norm type raises a ValueError."""
    V = fd.FunctionSpace(fd.UnitTriangleMesh(), "DG", 0)
    msg = "Norm type 'HDiv' not recognised."
    with pytest.raises(ValueError, match=msg):
        assemble_mass_matrix(V, norm_type="HDiv")


def test_lumping():
    """Test that lumped mass matrix is diagonal and has expected entries."""
    fs = fd.FunctionSpace(fd.UnitTriangleMesh(), "CG", 1)
    matrix = assemble_mass_matrix(fs, norm_type="L2", lumped=True)
    assert matrix.type == "diagonal"
    assert np.allclose(np.eye(3) / 6, matrix.convert("dense").getDenseArray())
