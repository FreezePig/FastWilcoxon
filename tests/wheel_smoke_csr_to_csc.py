"""Run against the installed, repaired wheel in each cibuildwheel test env."""

import platform
import subprocess
import sys

import numpy as np
import scipy.sparse as sp

from fastwilcoxon import csr_to_csc
from fastwilcoxon import _fastwilcoxon as native
from fastwilcoxon.wrappers import sumGroups_dense, sumGroups_dense_T


if sys.platform == "linux" and platform.machine().lower() == "x86_64":
    assert native.has_mkl, "Linux x86_64 wheel must include the real MKL backend"
    dependencies = subprocess.run(["ldd", native.__file__], check=True,
                                  capture_output=True, text=True).stdout
    assert "libmkl_" not in dependencies, dependencies
    assert "not found" not in dependencies, dependencies

backends = (["mkl"] if native.has_mkl else [])
if sys.version_info >= (3, 10):
    backends.append("graphblas")

max_threads = native.cpp_resolve_nthreads(0)
for backend in backends:
    for dtype in (np.float32, np.float64):
        for index_dtype in (np.int32, np.int64):
            values = np.array([0.0, -0.0, .125, 2.5, 3.25], dtype=dtype)
            X = sp.csr_matrix((values, [0, 3, 1, 0, 4], [0, 2, 2, 3, 5]), shape=(4, 6))
            X.indices = X.indices.astype(index_dtype)
            X.indptr = X.indptr.astype(index_dtype)
            snapshots = [a.copy() for a in (X.data, X.indices, X.indptr)]
            expected = X.tocsc()
            for threads in (1, 2, 0, -2, max_threads + 1):
                Y = csr_to_csc(X, backend=backend, nthreads=threads)
                assert type(Y) is sp.csc_matrix
                assert Y.shape == X.shape and Y.nnz == X.nnz
                assert Y.data.dtype == dtype
                assert Y.indices.dtype == np.int64 and Y.indptr.dtype == np.int64
                np.testing.assert_array_equal(Y.data.view(np.uint8), expected.data.view(np.uint8))
                np.testing.assert_array_equal(Y.indices, expected.indices)
                np.testing.assert_array_equal(Y.indptr, expected.indptr)
                for result in (Y.data, Y.indices, Y.indptr):
                    assert result.flags.writeable and result.flags.c_contiguous
                    assert not any(np.shares_memory(result, a) for a in (X.data, X.indices, X.indptr))
            for actual, before in zip((X.data, X.indices, X.indptr), snapshots):
                np.testing.assert_array_equal(actual.view(np.uint8), before.view(np.uint8))
        for shape in ((0, 0), (0, 4), (3, 0), (3, 4)):
            Y = csr_to_csc(sp.csr_matrix(shape, dtype=dtype), backend=backend, nthreads=2)
            assert Y.shape == shape and Y.nnz == 0
            assert Y.indices.dtype == np.int64 and Y.indptr.dtype == np.int64

# Switching the BLAS provider must also preserve the existing dense sums.
X = np.array([[.125, 2., 3.], [4., .5, 6.], [7., 8., .25]], dtype=np.float64)
groups = np.array([0, 1, 0], dtype=np.int64)
for order in ("C", "F"):
    matrix = np.array(X, order=order)
    np.testing.assert_array_equal(sumGroups_dense(matrix, groups, 2), np.vstack((X[0] + X[2], X[1])))
    np.testing.assert_array_equal(sumGroups_dense_T(matrix, groups, 2), np.vstack((X[:, 0] + X[:, 2], X[:, 1])))

print("Installed-wheel CSR conversion and existing BLAS sums passed:", backends,
      "has_mkl=", native.has_mkl, "extension=", native.__file__)
