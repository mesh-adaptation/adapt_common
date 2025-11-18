"""Module for MPI reduction operators related to Firedrake Functions."""

import numpy as np
import ufl
from mpi4py import MPI

__all__ = ["function_data_max", "function_data_min", "function_data_sum"]


def function_data_min(f):
    """Compute node-wise global minimum of Firedrake function.

    :arg f: the function to take the minimum over
    :type f: :class:`firedrake.function.Function`
    :return: the global minimum value
    :rtype: int
    """
    mesh = ufl.domain.extract_unique_domain(f)
    return mesh.comm.allreduce(f.dat.data_ro.min(initial=np.inf), MPI.MIN)


def function_data_max(f):
    """Compute node-wise global maximum of Firedrake function.

    :arg f: the function to take the maximum over
    :type f: :class:`firedrake.function.Function`
    :return: the global maximum value
    :rtype: int
    """
    mesh = ufl.domain.extract_unique_domain(f)
    return mesh.comm.allreduce(f.dat.data_ro.max(initial=-np.inf), MPI.MAX)


def function_data_sum(f):
    """Compute global sum of nodal values of Firedrake function.

    :arg f: the function to take the sum over
    :type f: :class:`firedrake.function.Function`
    :return: the global sum
    :rtype: int
    """
    mesh = ufl.domain.extract_unique_domain(f)
    return mesh.comm.allreduce(f.dat.data_ro.sum(), MPI.SUM)
