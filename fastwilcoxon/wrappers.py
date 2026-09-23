"""
Python wrappers for matrix operations with comprehensive error handling.

This module provides high-level Python interfaces for C++ matrix operations,
including input validation, type checking, and proper error handling.
"""
import sys
import os
from pathlib import Path
import ctypes
import numpy as np
from typing import List, Optional

try:
    from . import _fastwilcoxon as fw
except ImportError as e:
    raise ImportError(
        "Failed to import C++ extension '_fastwilcoxon'. "
        "Please ensure the package is installed correctly."
    ) from e

integer_dtypes = {
    np.dtype('uint8'), np.dtype('uint16'), np.dtype('uint32'), np.dtype('uint64'),
    np.dtype('int8'), np.dtype('int16'), np.dtype('int32'), np.dtype('int64')
}

def validate_int_scalar(name: str, value, *, positive: bool = False) -> int:
    """
    Validate a parameter is a Python integer and satisfies
    optional range constraints.

    Parameters
    ----------
    name : str
        Name of Parameter (used in error messages).
    value : any
        Value to validate.
    positive : bool, optional
        If True, value must be positive (greater than 0). Default is False.

    Returns
    -------
    int
        The validated integer value.
    Raises
    ------
    TypeError
        If value is not an integer.
    ValueError
        If value does not satisfy range constraints.
    """
    if not isinstance(value, int):
        raise TypeError(f"Parameter '{name}' must be an integer, got {type(value)}")

    if positive and value <= 0:
        raise ValueError(f"Parameter '{name}' must be positive, got {value}")
    
    return value

def standardiz_nthreads(nthreads: int) -> int:
    """
    Validate and Standardize the nthreads parameter.

    Parameters
    ----------
    nthreads : int
        Number of threads to use for computation. Must be positive. If nthreads <= 0,
        will use all available threads (nthreads=-1).

    Returns
    -------
    int
        A standardized thread count:
        - positive integer: use that many threads
        - -1: use all available threads

    """
    nthreads = validate_int_scalar('nthreads', nthreads)
    if nthreads <= 0:
        nthreads = -1
    return nthreads

def validate_ndarray(name: str, arr, ndim: Optional[int] = None) -> np.ndarray:
    """
    Validate that an input is a numpy array with optimal dimention check.

    Parameters
    ----------
    name : str
        Name of the parameter (used in error messages).
    arr : Any
        Object to validate
    ndim : int or None, optional
        Expected number of dimensions. If None, any number of dimensions is accepted.
        Default is None.
    
    Returns
    ----------
    np.ndarray
        The validated numpy array.

    Raises
    ----------
    TypeError
        If the input is not a numpy array.
    ValueError
        If dim of array not match ndim.
    """
    if not isinstance(name, str):
        raise TypeError(f"Parameter 'name' must be a string, got {type(name)}")
    
    if ndim is not None and not isinstance(ndim, int):
        raise TypeError(f"Parameter 'ndim' must be an integer or None, got {type(ndim)}")
    
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Parameter '{name}' must be a numpy array, got {type(arr)}")
    
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"Parameter '{name}' must be {ndim}-dimensional, got {arr.ndim}D")
    
    return arr

def validate_float_input(name: str, arr, ndim: Optional[int] = None, 
                         allow_empty: bool = False, allow_zero: bool = False) -> np.ndarray:
    """
    Validate a positive floating-point input array and coerce to np.float64.

    Parameters
    ----------
    name : str
        Name of the parameter (used in error messages).
    arr : Any
        Input object to validate
    ndim : int or None, optional
        Expected number of dimensions. If None, any number of dimensions is accepted.
        Default is None.
    allow_empty : bool, optional
        If False, the array cannot be empty. Default is False.
    allow_zero : bool, optional
        If False, the array cannot contain zero values. Default is False.
    
    Returns
    ----------
    np.ndarray
        The validated and coerced numpy array with dtype np.float64.

    Raises
    ----------
    TypeError
        If the input is not a numpy array or cannot be coerced to float.
    ValueError
        If the array is empty (when allow_empty=False); does not match expected dimensions;
        contains non-positive values
    """
    arr = validate_ndarray(name, arr, ndim = ndim)

    if not np.issubdtype(arr.dtype, np.floating):
        raise TypeError(f"Parameter '{name}' must be a floating-point array, got dtype {arr.dtype}")
    
    if not allow_empty and arr.size == 0:
        raise ValueError(f"Parameter '{name}' cannot be an empty array")

    min_val = np.min(arr)
    if allow_zero:
        if min_val < 0:
            raise ValueError(
                f"All elements in '{name}' must be non-negative (greater than or equal to 0), "
                f"but found minimum value: {min_val}"
            )
    else:
        if min_val <= 0:
            raise ValueError(
                f"All elements in '{name}' must be positive (greater than 0), "
                f"but found minimum value: {min_val}"
            )

    if arr.dtype != np.float64:
        arr = arr.astype(np.float64)
    
    return arr

def validate_output_buffer(name: str, arr, ndim: Optional[int] = None) -> np.ndarray:
    """
    Validate an output array that must already be a float64 numpy ndarray, 
    and if is contiguous and writeable.

    Parameters
    ----------
    name : str
        Parameter name used in error messages.
    arr : Any
        Output array to validate.
    ndim : int or None, default None
        Expected number of dimensions.

    Returns
    -------
    np.ndarray
        The validated float64 numpy array.

    Raises
    ------
    TypeError
        If arr is not a numpy array or does not have dtype float64.
    ValueError
        If ndim does not match.
    """
    arr = validate_ndarray(name, arr, ndim=ndim)

    if arr.dtype != np.float64:
        raise TypeError(
            f"Parameter '{name}' must have dtype float64, got {arr.dtype}"
        )
    
    #  --- Contiguity and writeability ---
    if not arr.flags['C_CONTIGUOUS']:
        raise ValueError(f"Parameter '{name}' must be a contiguous array")
    if not arr.flags['WRITEABLE']:
        raise ValueError(f"Parameter '{name}' must be writable")
    return arr

def validate_int_input(name: str, arr, ndim: Optional[int] = None) -> np.ndarray:
    """
    Validate an integer array and preserve 64-bit-safe dtypes.

    Parameters
    ----------
    name : str
        Parameter name used in error messages.
    arr : Any
        Input array to validate.
    ndim : int, default 1
        Expected number of dimensions.

    Raises
    ------
    TypeError
        If the input is not a numpy array or does not have an integer dtype.
    """
    arr = validate_ndarray(name, arr, ndim=ndim)

    if arr.dtype not in integer_dtypes:
        raise TypeError(
            f"Parameter '{name}' must have an integer dtype (uint8/int8/uint16/int16/uint32/int32/uint64/int64), "
            f"got {arr.dtype}"
        )
    
    if arr.dtype not in (np.int32, np.int64):
        if np.issubdtype(arr.dtype, np.unsignedinteger) and arr.size > 0:
            max_int64 = np.iinfo(np.int64).max
            if np.max(arr) > max_int64:
                raise ValueError(
                    f"Parameter '{name}' contains values too large for int64 conversion"
                )
        arr = arr.astype(np.int64)
    
    return arr

def validate_spmatrix_input(x: np.ndarray, 
                            p: np.ndarray, 
                            i: np.ndarray,
                            groups: np.ndarray,
                            ngroups: int,
                            nrow: int,
                            ncol: int,
                            cell_num: int,
                            matrix_type: str = 'csc') -> None:
    """
    Validate the input arrays and parameters for sparse matrix operations.
    Inludes length checks, value range checks, and format checks for CSC/CSR matrices.

    Parameters
    ----------
    x : np.ndarray
        1D array of non-zero values in the sparse matrix.
    p : np.ndarray
        indptr array for CSC/CSR format.
    i : np.ndarray
        indices corresponding to each non-zero value in x.
    ngroups: int
        Total number of groups. Must be positive.
    nrow : int
        Number of rows in the sparse matrix.
    ncol : int
        Number of columns in the sparse matrix.
    cell_num : int
        Number of non-zero cells in the sparse matrix.
    matrix_type : str, default 'csc'
        Type of sparse matrix format (either 'csc' or 'csr').

    Raises
    ------
    TypeError
        If any of the inputs are of incorrect type.
    ValueError
        If any of the inputs are of incorrect value.
    """

    # Basic type and dimension checks
    if len(x) != len(i):
        raise ValueError(f"Arrays 'x' and 'i' must have the same length, "
                        f"got x: {len(x)}, i: {len(i)}")
    
    if p[0] != 0:
        raise ValueError(f"First element of 'p' must be 0, got {p[0]}")
    if p[-1] != len(x):
        raise ValueError(f"Last element of 'p' must equal length of 'x' ({len(x)}), "
                        f"got {p[-1]}")
    if not np.all(np.diff(p) >= 0):
        raise ValueError("Array 'p' must be non-decreasing (valid CSC format)")
    
    if len(groups) != cell_num:
        raise ValueError(f"Array 'groups' must have length equal to number of rows ({cell_num}), "
                        f"got {len(groups)}")
    if np.max(groups) >= ngroups:
        raise ValueError(f"Group indices must be in range [0, {ngroups-1}], "
                         f"found maximum group index: {np.max(groups)}")
    
    if matrix_type == 'csc':
        if np.max(i) >= nrow:
            raise ValueError(f"Row indices in 'i' must be in range [0, {nrow-1}], "
                             f"found maximum row index: {np.max(i)}")
        if len(p) != ncol + 1:
            raise ValueError(f"Array 'p' must have length ncol + 1 = {ncol + 1}, "
                            f"got {len(p)}")
    elif matrix_type == 'csr':
        if np.max(i) >= ncol:
            raise ValueError(f"Column indices in 'i' must be in range [0, {ncol-1}], "
                             f"found maximum column index: {np.max(i)}")
        if len(p) != nrow + 1:
            raise ValueError(f"Array 'p' must have length nrow + 1 = {nrow + 1}, "
                            f"got {len(p)}")
    else:
        raise TypeError(f"Invalid matrix_type '{matrix_type}'. Expected 'csc' or 'csr'.")    

def validate_denmatrix_input(
    groups: np.ndarray,
    ngroups: int,
    cell_num: int
) -> None:
    """
    Validate the input arrays and parameters for dense matrix operations.
    Inludes length checks, value range checks, and format checks for dense matrices.

    Parameters
    ----------
    groups : np.ndarray
        1D array indicating group membership for rows or columns.
    ngroups : int
        Total number of groups. Must be positive.
    cell_num : int
        Number of rows or columns in the dense matrix (depending on grouping).

    Raises
    ------
    TypeError
        If any of the inputs are of incorrect type.
    ValueError
        If any of the inputs are of incorrect value.
    """

    if len(groups) != cell_num:
        raise ValueError(f"Array 'groups' must have length equal to number of rows/columns ({cell_num}), "
                        f"got {len(groups)}")
    if np.max(groups) >= ngroups:
        raise ValueError(f"Group indices must be in range [0, {ngroups-1}], "
                         f"found maximum group index: {np.max(groups)}")

def sumGroups_csc(
    x: np.ndarray,
    p: np.ndarray,
    i: np.ndarray,
    nrow: int,
    ncol: int,
    groups: np.ndarray,
    ngroups: int,
    nthreads: int = 1
) -> np.ndarray:
    """
    Sum groups for CSC (Compressed Sparse Column) sparse matrix by column.
    
    This function performs group-wise summation of a sparse matrix stored in CSC format
    Each row belongs to a group, and the function sums all values within each group
    for each column.
    
    Parameters
    ----------
    x : np.ndarray
        1D array of non-zero values in the sparse matrix (dtype: float64).
    p : np.ndarray
        1D array of column pointers indicating start/end indices for each column
        in arrays x and i (dtype: size_t/uint64). Length should be ncol + 1.
    i : np.ndarray
        1D array of row indices corresponding to each non-zero value in x
        (dtype: size_t/uint64).
    ncol : int
        Number of columns in the sparse matrix. Must be positive.
    nrow: int
        Number of rows in the sparse matrix. Must be positive. 
        Just for input tests, not used for calc.
    groups : np.ndarray
        1D array indicating which group each row belongs to (dtype: size_t/uint64).
        Values should be in range [0, ngroups-1].
    ngroups : int
        Total number of groups. Must be positive.
    nthreads: int, optional
        Number of threads to use for computation. Default is 1.
        If nthread <= 0 or >= largest threads, uses all available threads.
        
    Returns
    -------
    np.ndarray
        2D array of shape (ngroups, ncol) containing the sum of values for each
        group and column combination (dtype: float64).
    """

    # Array parameter validation
    x = validate_float_input('x', x, ndim=1, allow_empty=False)
    p = validate_int_input('p', p, ndim=1)
    i = validate_int_input('i', i, ndim=1)
    groups = validate_int_input('groups', groups, ndim=1)
    
    # Scalar parameter validation
    ncol = validate_int_scalar('ncol', ncol, positive=True)
    nrow = validate_int_scalar('nrow', nrow, positive=True)
    ngroups = validate_int_scalar('ngroups', ngroups, positive=True)
    nthreads = standardiz_nthreads(nthreads)

    # csc matrix validation
    validate_spmatrix_input(x, p, i, groups, ngroups,
                            nrow, ncol, cell_num=nrow, matrix_type='csc')
    
    # Call the C++ function
    try:
        result = fw.cpp_sumGroups_csc(x, p, i, ncol, groups, ngroups, nthreads)
        return result
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")
    
def sumGroups_csr(
    x: np.ndarray,
    p: np.ndarray,
    i: np.ndarray,
    nrow: int,
    ncol: int,
    groups: np.ndarray,
    ngroups: int,
    nthreads: int = 1
) -> np.ndarray:
    """
    Sum groups for CSR (Compressed Sparse Row) sparse matrix by column.
    
    This function performs group-wise summation of a sparse matrix stored in CSR format.
    Each row belongs to a group, and the function sums all values within each group
    for each column.
    
    Parameters
    ----------
    x : np.ndarray
        1D array of non-zero values in the sparse matrix (dtype: float64). 
        All values must be positive (greater than 0).
    p : np.ndarray
        1D array of row pointers indicating start/end indices for each row
        in arrays x and i (dtype: size_t/uint64). Length should be nrow + 1.
    i : np.ndarray
        1D array of column indices corresponding to each non-zero value in x
        (dtype: size_t/uint64).
    ncol : int
        Number of columns in the sparse matrix. Must be positive.
    nrow : int
        Number of rows in the sparse matrix. Must be positive.
    groups : np.ndarray
        1D array indicating which group each row belongs to (dtype: size_t/uint64).
        Values should be in range [0, ngroups-1]. Length should be nrow.
    ngroups : int
        Total number of groups. Must be positive.
    nthreads: int, optional
        Number of threads to use for computation. Must be positive. Default is 1.
        
    Returns
    -------
    np.ndarray
        2D array of shape (ngroups, ncol) containing the sum of values for each
        group and column combination (dtype: float64).
    """

    # Array parameter validation
    x = validate_float_input('x', x, ndim=1)
    p = validate_int_input('p', p, ndim=1)
    i = validate_int_input('i', i, ndim=1)
    groups = validate_int_input('groups', groups, ndim=1)

    # Scalar parameter validation
    ncol = validate_int_scalar('ncol', ncol, positive=True)
    nrow = validate_int_scalar('nrow', nrow, positive=True)
    ngroups = validate_int_scalar('ngroups', ngroups, positive=True)
    nthreads = standardiz_nthreads(nthreads)
    
    # csr matrix validation
    validate_spmatrix_input(x, p, i, groups, ngroups,
                            nrow, ncol, cell_num=nrow, matrix_type='csr')
    
    # Call the C++ function
    try:
        result = fw.cpp_sumGroups_csr(x, p, i, ncol, nrow, groups, ngroups, nthreads)
        return result
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")
    
def sumGroups_csc_T(
    x: np.ndarray,
    p: np.ndarray,
    i: np.ndarray,
    nrow: int,
    ncol: int,
    groups: np.ndarray,
    ngroups: int,
    nthreads: int = 1
) -> np.ndarray:
    """
    Sum groups for transposed CSC (Compressed Sparse Column) sparse matrix by row.
    
    This function performs group-wise summation of a sparse matrix stored in CSC format,
    but groups are applied to COLUMNS instead of rows. Each column belongs to a group,
    and the function sums all values within each group for each row.
    
    This is equivalent to summing groups on the transpose of the original matrix.
    
    Parameters
    ----------
    x : np.ndarray
        1D array of non-zero values in the sparse matrix (dtype: float64). 
        All values must be positive (greater than 0).
    p : np.ndarray
        1D array of column pointers indicating start/end indices for each column
        in arrays x and i (dtype: size_t/uint64). Length should be ncol + 1.
    i : np.ndarray
        1D array of row indices corresponding to each non-zero value in x
        (dtype: size_t/uint64).
    ncol : int
        Number of columns in the sparse matrix. Must be positive.
    nrow : int
        Number of rows in the sparse matrix. Must be positive.
    groups : np.ndarray
        1D array indicating which group each COLUMN belongs to (dtype: size_t/uint64).
        Values should be in range [0, ngroups-1]. Length should be ncol.
    ngroups : int
        Total number of groups. Must be positive.
    nthreads: int, optional
        Number of threads to use for computation. Must be positive. Default is 1.
        
    Returns
    -------
    np.ndarray
        2D array of shape (ngroups, nrow) containing the sum of values for each
        group and row combination (dtype: float64).
    """

    # Array parameter validation
    x = validate_float_input('x', x, ndim=1)
    p = validate_int_input('p', p, ndim=1)
    i = validate_int_input('i', i, ndim=1)
    groups = validate_int_input('groups', groups, ndim=1)

    # Scalar parameter validation
    ncol = validate_int_scalar('ncol', ncol, positive=True)
    nrow = validate_int_scalar('nrow', nrow, positive=True)
    ngroups = validate_int_scalar('ngroups', ngroups, positive=True)
    nthreads = standardiz_nthreads(nthreads)

    # csc matrix validation
    validate_spmatrix_input(x, p, i, groups, ngroups,
                            nrow, ncol, cell_num=ncol, matrix_type='csc')
    
    # Call the C++ function
    try:
        result = fw.cpp_sumGroups_csc_T(x, p, i, ncol, nrow, groups, ngroups, nthreads)
        return result
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")
    

def sumGroups_csr_T(
    x: np.ndarray,
    p: np.ndarray,
    i: np.ndarray,
    nrow: int,
    ncol: int,
    groups: np.ndarray,
    ngroups: int,
    nthreads: int = 1
) -> np.ndarray:
    """
    Sum groups for transposed CSR (Compressed Sparse Row) sparse matrix by rows.
    
    This function performs group-wise summation of a sparse matrix stored in CSR format,
    but groups are applied to COLUMNS instead of rows. Each column belongs to a group,
    and the function sums all values within each group for each row.
    
    This is equivalent to summing groups on the transpose of the original matrix.
    
    Parameters
    ----------
    x : np.ndarray
        1D array of non-zero values in the sparse matrix (dtype: float64). 
        All values must be positive (greater than 0).
    p : np.ndarray
        1D array of row pointers indicating start/end indices for each row
        in arrays x and i (dtype: size_t/uint64). Length should be nrow + 1.
    i : np.ndarray
        1D array of column indices corresponding to each non-zero value in x
        (dtype: size_t/uint64).
    ncol: int
        Number of columns in the sparse matrix. Must be positive.
        Just for input tests, not used for calc.
    nrow : int
        Number of rows in the sparse matrix. Must be positive.
    groups : np.ndarray
        1D array indicating which group each COLUMN belongs to (dtype: size_t/uint64).
        Values should be in range [0, ngroups-1]. Length should match the total number of columns.
    ngroups : int
        Total number of groups. Must be positive.
    nthreads: int, optional
        Number of threads to use for computation. Must be positive. Default is 1.
        
    Returns
    -------
    np.ndarray
        2D array of shape (ngroups, nrow) containing the sum of values for each
        group and row combination (dtype: float64).
    """

    # Array parameter validation
    x = validate_float_input('x', x, ndim=1)
    p = validate_int_input('p', p, ndim=1)
    i = validate_int_input('i', i, ndim=1)
    groups = validate_int_input('groups', groups, ndim=1)
    
    # Scalar parameter validation
    ncol = validate_int_scalar('ncol', ncol, positive=True)
    nrow = validate_int_scalar('nrow', nrow, positive=True)
    ngroups = validate_int_scalar('ngroups', ngroups, positive=True)
    nthreads = standardiz_nthreads(nthreads)
    
    # validate csr matrix
    validate_spmatrix_input(x, p, i, groups, ngroups,
                            nrow, ncol, cell_num=ncol, matrix_type='csr')
    
    # Call the C++ function
    try:
        result = fw.cpp_sumGroups_csr_T(x, p, i, nrow, groups, ngroups, nthreads)
        return result
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")
    
def sumGroups_dense(
    x: np.ndarray,
    groups: np.ndarray,
    ngroups: int
) -> np.ndarray:
    """
    Sum groups for dense matrix by columns.
    
    This function performs group-wise summation of a dense matrix.
    Each row belongs to a group, and the function sums all values within each group
    for each column.
    
    Parameters
    ----------
    x : np.ndarray
        2D dense matrix (dtype: float64). All values must be non-negative.
    groups : np.ndarray
        1D array indicating which group each row belongs to (dtype: size_t/uint64).
        Values should be in range [0, ngroups-1]. Length should equal number of rows in x.
    ngroups : int
        Total number of groups. Must be positive.
        
    Returns
    -------
    np.ndarray
        2D array of shape (ngroups, ncols) containing the sum of values for each
        group and column combination (dtype: float64).
    """
    # Array parameter validation
    x = validate_float_input('x', x, ndim=2, allow_zero=True)
    groups = validate_int_input('groups', groups, ndim=1)

    # scalar parameter validation
    ngroups = validate_int_scalar('ngroups', ngroups, positive=True)

    # validate dense matrix
    nrows, _ = x.shape
    validate_denmatrix_input(groups, ngroups, cell_num=nrows)
        
    # Call the C++ function
    try:
        result = fw.cpp_sumGroups_dense(x, groups, ngroups)
        return result
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")
    

def sumGroups_dense_T(
    x: np.ndarray,
    groups: np.ndarray,
    ngroups: int
) -> np.ndarray:
    """
    Sum groups for transposed dense matrix by rows.
    
    This function performs group-wise summation of a dense matrix,
    but groups are applied to COLUMNS instead of rows. Each column belongs to a group,
    and the function sums all values within each group for each row.
    
    This is equivalent to summing groups on the transpose of the original matrix.
    
    Parameters
    ----------
    x : np.ndarray
        2D dense matrix (dtype: float64). All values must be positive (greater than 0).
    groups : np.ndarray
        1D array indicating which group each COLUMN belongs to (dtype: size_t/uint64).
        Values should be in range [0, ngroups-1]. Length should equal number of columns in x.
    ngroups : int
        Total number of groups. Must be positive.
        
    Returns
    -------
    np.ndarray
        2D array of shape (ngroups, nrows) containing the sum of values for each
        group and row combination (dtype: float64).
    """

    # Array parameter validation
    x = validate_float_input('x', x, ndim=2, allow_zero=True)
    groups = validate_int_input('groups', groups, ndim=1)
    
    # Scalar parameter validation
    ngroups = validate_int_scalar('ngroups', ngroups, positive=True)

    # validate dense matrix
    _, ncols = x.shape
    validate_denmatrix_input(groups, ngroups, cell_num=ncols)
        
    # Call the C++ function
    try:
        result = fw.cpp_sumGroups_dense_T(x, groups, ngroups)
        return result
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")
    
def nnzeroGroups_csc(
    p: np.ndarray,
    i: np.ndarray,
    nrow: int,
    ncol: int,
    groups: np.ndarray,
    ngroups: int,
    nthreads: int = 1
) -> np.ndarray:
    """
    Count non-zero elements for CSC (Compressed Sparse Column) sparse matrix by groups.
    
    This function counts the number of non-zero elements in a sparse matrix stored in CSC format.
    Each row belongs to a group, and the function counts non-zero elements within each group.
    for each column.
    
    Parameters
    ----------
    p : np.ndarray
        1D array of column pointers indicating start/end indices for each column
        in array i (dtype: size_t/uint64). Length should be ncol + 1.
    i : np.ndarray
        1D array of row indices corresponding to each non-zero element
        (dtype: size_t/uint64).
    ncol : int
        Number of columns in the sparse matrix. Must be positive.
    nrow : int
        Number of rows in the sparse matrix. Must be positive.
        Just for input tests, not used for calc.
    groups : np.ndarray
        1D array indicating which group each row belongs to (dtype: size_t/uint64).
        Values should be in range [0, ngroups-1].
    ngroups : int
        Total number of groups. Must be positive.
    nthreads: int, optional
        Number of threads to use for computation. Must be positive. Default is 1.
        
    Returns
    -------
    np.ndarray
        2D array of shape (ngroups, ncol) containing the count of non-zero elements for each
        group and column combination (dtype: int32).
    """

    # Array parameter validation
    p = validate_int_input('p', p, ndim=1)
    i = validate_int_input('i', i, ndim=1)
    groups = validate_int_input('groups', groups, ndim=1)
    
    # Scalar parameter validation
    ncol = validate_int_scalar('ncol', ncol, positive=True)
    nrow = validate_int_scalar('nrow', nrow, positive=True)
    ngroups = validate_int_scalar('ngroups', ngroups, positive=True)
    nthreads = standardiz_nthreads(nthreads)
    
    # csc matrix validation
    validate_spmatrix_input(np.zeros(len(i), dtype=np.float64), p, i, groups,
                            ngroups, nrow, ncol, cell_num=nrow, matrix_type='csc')

    # Call the C++ function
    try:
        result = fw.cpp_nnzeroGroups_csc(p, i, ncol, groups, ngroups, nthreads)
        return result
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")
    
def nnzeroGroups_csr(
    p: np.ndarray,
    i: np.ndarray,
    nrow: int,
    ncol: int,
    groups: np.ndarray,
    ngroups: int,
    nthreads: int = 1
) -> np.ndarray:
    """
    Count non-zero elements for CSR (Compressed Sparse Row) sparse matrix by groups.
    
    This function counts the number of non-zero elements in a sparse matrix stored in CSR format.
    Each row belongs to a group, and the function counts non-zero elements within each group
    for each column.
    
    Parameters
    ----------
    p : np.ndarray
        1D array of row pointers indicating start/end indices for each row
        in array i (dtype: size_t/uint64). Length should be nrow + 1.
    i : np.ndarray
        1D array of column indices corresponding to each non-zero element
        (dtype: size_t/uint64).
    ncol : int
        Number of columns in the sparse matrix. Must be positive.
    nrow : int
        Number of rows in the sparse matrix. Must be positive.
    groups : np.ndarray
        1D array indicating which group each row belongs to (dtype: size_t/uint64).
        Values should be in range [0, ngroups-1].
    ngroups : int
        Total number of groups. Must be positive.
    nthreads: int, optional
        Number of threads to use for computation. Must be positive. Default is 1.
    """

    # Array parameter validation
    p = validate_int_input('p', p, ndim=1)
    i = validate_int_input('i', i, ndim=1)
    groups = validate_int_input('groups', groups, ndim=1)
    
    # Scalar parameter validation
    ncol = validate_int_scalar('ncol', ncol, positive=True)
    nrow = validate_int_scalar('nrow', nrow, positive=True)
    ngroups = validate_int_scalar('ngroups', ngroups, positive=True)
    nthreads = standardiz_nthreads(nthreads)
    
    # csr matrix validation
    validate_spmatrix_input(np.zeros(len(i), dtype=np.float64), p, i, groups,
                            ngroups, nrow, ncol, cell_num=nrow, matrix_type='csr')
    
    # Call the C++ function
    try:
        result = fw.cpp_nnzeroGroups_csr(p, i, ncol, nrow, groups, ngroups, nthreads)
        return result
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")
    
def nnzeroGroups_csc_T(
    p: np.ndarray,
    i: np.ndarray,
    nrow: int,
    ncol: int,
    groups: np.ndarray,
    ngroups: int,
    nthreads: int = 1
) -> np.ndarray:
    """
    Count non-zero elements for transposed CSC (Compressed Sparse Column) sparse matrix by groups.
    
    This function counts the number of non-zero elements in a sparse matrix stored in CSC format,
    but groups are applied to COLUMNS instead of rows. Each column belongs to a group,
    and the function counts non-zero elements within each group for each row.
    
    This is equivalent to counting non-zeros on the transpose of the original matrix.
    
    Parameters
    ----------
    p : np.ndarray
        1D array of column pointers indicating start/end indices for each column
        in array i (dtype: size_t/uint64). Length should be ncol + 1.
    i : np.ndarray
        1D array of row indices corresponding to each non-zero element
        (dtype: size_t/uint64).
    ncol : int
        Number of columns in the sparse matrix. Must be positive.
    nrow : int
        Number of rows in the sparse matrix. Must be positive.
    groups : np.ndarray
        1D array indicating which group each COLUMN belongs to (dtype: size_t/uint64).
        Values should be in range [0, ngroups-1]. Length should be ncol.
    ngroups : int
        Total number of groups. Must be positive.
    nthreads: int, optional
        Number of threads to use for computation. Must be positive. Default is 1.
        
    Returns
    -------
    np.ndarray
        2D array of shape (ngroups, nrow) containing the count of non-zero elements for each
        group and row combination (dtype: int32).
    """
    # Array parameter validation
    p = validate_int_input('p', p, ndim=1)
    i = validate_int_input('i', i, ndim=1)
    groups = validate_int_input('groups', groups, ndim=1)

    # Scalar parameter validation
    ncol = validate_int_scalar('ncol', ncol, positive=True)
    nrow = validate_int_scalar('nrow', nrow, positive=True)
    ngroups = validate_int_scalar('ngroups', ngroups, positive=True)
    nthreads = standardiz_nthreads(nthreads)

    # validate csc matrix
    validate_spmatrix_input(np.zeros(len(i), dtype=np.float64), p, i, groups,
                            ngroups, nrow, ncol, cell_num=ncol, matrix_type='csc')
        
    # Call the C++ function
    try:
        result = fw.cpp_nnzeroGroups_csc_T(p, i, ncol, nrow, groups, ngroups, nthreads)
        return result
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")

def nnzeroGroups_csr_T(
    p: np.ndarray,
    i: np.ndarray,
    nrow: int,
    ncol: int,
    groups: np.ndarray,
    ngroups: int,
    nthreads: int = 1
) -> np.ndarray:
    """
    Count non-zero elements for transposed CSR (Compressed Sparse Row) sparse matrix by groups.
    
    This function counts the number of non-zero elements in a sparse matrix stored in CSR format,
    but groups are applied to COLUMNS instead of rows. Each column belongs to a group,
    and the function counts non-zero elements within each group for each row.
    
    This is equivalent to counting non-zeros on the transpose of the original matrix.
    
    Parameters
    ----------
    p : np.ndarray
        1D array of row pointers indicating start/end indices for each row
        in array i (dtype: size_t/uint64). Length should be nrow + 1.
    i : np.ndarray
        1D array of column indices corresponding to each non-zero element
        (dtype: size_t/uint64).
    ncol : int
        Number of columns in the sparse matrix. Must be positive.
        Used for validation only (not passed to C++ function).
    nrow : int
        Number of rows in the sparse matrix. Must be positive.
    groups : np.ndarray
        1D array indicating which group each COLUMN belongs to (dtype: size_t/uint64).
        Values should be in range [0, ngroups-1]. Length should match the actual number of columns.
    ngroups : int
        Total number of groups. Must be positive.
    nthreads: int, optional
        Number of threads to use for computation. Must be positive. Default is 1.
        
    Returns
    -------
    np.ndarray
        2D array of shape (ngroups, nrow) containing the count of non-zero elements for each
        group and row combination (dtype: int32).
    """

    # Array parameter validation
    p = validate_int_input('p', p, ndim=1)
    i = validate_int_input('i', i, ndim=1)
    groups = validate_int_input('groups', groups, ndim=1)

    # Scalar parameter validation
    ncol = validate_int_scalar('ncol', ncol, positive=True)
    nrow = validate_int_scalar('nrow', nrow, positive=True)
    ngroups = validate_int_scalar('ngroups', ngroups, positive=True)
    nthreads = standardiz_nthreads(nthreads)
    
    # validate csr matrix
    validate_spmatrix_input(np.zeros(len(i), dtype=np.float64), p, i, groups,
                            ngroups, nrow, ncol, cell_num=ncol, matrix_type='csr')
        
    # Call the C++ function
    try:
        result = fw.cpp_nnzeroGroups_csr_T(p, i, ncol, nrow, groups, ngroups, nthreads)
        return result
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")
    
def nnzeroGroups_dense(
    x: np.ndarray,
    groups: np.ndarray,
    ngroups: int,
    nthreads: int = 1
) -> np.ndarray:
    
    """
    Count non-zero elements in a dense matrix by row groups, per column.

    This function counts, for each group and each column, how many rows in that group
    have a non-zero entry in the given column.

    Parameters
    ----------
    x : np.ndarray
        2D dense input matrix (dtype: float64). Shape: (nrows, ncols).
    groups : np.ndarray
        1D array assigning each row to a group (dtype: any integer type).
        Length must be equal to number of rows in `x`.
        Group indices must be in range [0, ngroups - 1].
    ngroups : int
        Total number of groups. Must be positive.
    nthreads : int, optional
        Number of OpenMP threads to use. If <= 0, uses all available threads.
        Default is 1.

    Returns
    -------
    np.ndarray
        2D integer array of shape (ngroups, ncols), where result[g, c]
        is the count of non-zero entries in column `c` among rows assigned to group `g`.
    """

    # Array parameter validation
    x = validate_float_input('x', x, ndim=2, allow_zero=True)
    groups = validate_int_input('groups', groups, ndim=1)

    # --- Scalar validation ---
    ngroups = validate_int_scalar('ngroups', ngroups, positive=True)
    nthreads = standardiz_nthreads(nthreads)

    # validate dense matrix
    nrows, _ = x.shape
    validate_denmatrix_input(groups, ngroups, cell_num=nrows)

    # Call the C++ function
    try:
        result = fw.cpp_nnzeroGroups_dense(x, groups, ngroups, nthreads)
        return result
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")
    
def nnzeroGroups_dense_T(
    x: np.ndarray,
    groups: np.ndarray,
    ngroups: int,
    nthreads: int = 1
) -> np.ndarray:
    """
    Count non-zero elements in a dense matrix by column groups, per row.

    This function transposes the logic of `nnzeroGroups_dense`: each *column* belongs to a group,
    and for each group and each *row*, it counts how many columns in that group have a non-zero
    entry in the given row.

    Parameters
    ----------
    x : np.ndarray
        2D dense input matrix (dtype: float64). Shape: (nrows, ncols).
    groups : np.ndarray
        1D array assigning each *column* to a group (dtype: any integer type).
        Length must be equal to number of columns in `x`.
        Group indices must be in range [0, ngroups - 1].
    ngroups : int
        Total number of groups. Must be positive.
    nthreads : int, optional
        Number of OpenMP threads to use. If <= 0, uses all available threads.
        Default is 1.

    Returns
    -------
    np.ndarray
        2D integer array of shape (ngroups, nrows), where result[g, r]
        is the count of non-zero entries in row `r` among columns assigned to group `g`.
    """

    # Type validation
    x = validate_float_input('x', x, ndim=2, allow_zero=True)
    groups = validate_int_input('groups', groups, ndim=1)

    # Scalar parameter validation
    ngroups = validate_int_scalar('ngroups', ngroups, positive=True)
    nthreads = standardiz_nthreads(nthreads)

    # validate dense matrix
    _, ncols = x.shape
    validate_denmatrix_input(groups, ngroups, cell_num=ncols)
        
    # --- Call C++ backend ---
    try:
        result = fw.cpp_nnzeroGroups_dense_T(x, groups, ngroups, nthreads)
        return result
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")
    
def rank_matrix_csc(
    x: np.ndarray,
    p: np.ndarray,
    rank_data_out: np.ndarray,
    nrow: int,
    ncol: int,
    nthreads: int = 1 
) -> List[List[float]]:
    """
    Compute average ranks for non-zero elements in a CSC-formatted sparse matrix, column-wise.

    This function assigns ranks to non-zero values within each column of a sparse matrix
    stored in CSC format. Tied values receive the average of the ranks they would have
    occupied. Zeros are implicitly treated as the smallest values and are not stored,
    but their count is used to shift the ranks of non-zero entries upward.

    The function modifies the input array `x` in-place by replacing non-zero values
    with their computed ranks (adjusted for implicit zeros).

    Parameters
    ----------
    x : np.ndarray
        1D array of non-zero values in CSC format (dtype: float64).
        Length must equal the number of non-zero elements (nnz).
    p : np.ndarray
        1D array of column pointers in CSC format (dtype: int32 or int64).
        Length must be `ncol + 1`, with `p[0] == 0` and non-decreasing.
    rank_data_out : np.ndarray
        1D output array to store tie group sizes and zero counts (dtype: float64).
        Must be writable, C-contiguous, and have same length as `x`.
    nrow : int
        Number of rows in the matrix. Must be positive.
    ncol : int
        Number of columns in the matrix. Must be positive.
    nthreads : int, optional
        Number of OpenMP threads to use. If <= 0, uses all available threads.
        Default is 1.

    Returns
    -------
    list[list[float]]
        A list of length `ncol`. Each element is a list containing:
        - The sizes of tie groups (number of tied elements) for that column (only for ties with size > 1),
        - Followed by the number of implicit zero elements in that column (as a float).

        Example: `[[2.0, 3.0], [0.0], [4.0, 1.0]]` means:
          - Column 0: one tie of size 2, and 3 zeros.
          - Column 1: no ties, and 0 zeros.
          - Column 2: one tie of size 4, and 1 zero.

    Notes
    -----
    - **This function modifies `x` in-place**: non-zero entries are replaced with their ranks.
    - Ranks start at 1 (i.e., smallest non-zero gets rank = (#zeros + 1)).
    - Input arrays must be contiguous and writable.
    """

    # Array parameter validation
    x = validate_float_input('x', x, ndim=1)
    p = validate_int_input('p', p, ndim=1)
    rank_data_out = validate_output_buffer('rank_data_out', rank_data_out, ndim=1)
    
    # Scalar parameter validation
    ncol = validate_int_scalar('ncol', ncol, positive=True)
    nrow = validate_int_scalar('nrow', nrow, positive=True)
    nthreads = standardiz_nthreads(nthreads)
    
    # validate csc matrix
    if len(x) != len(rank_data_out):
        raise ValueError(f"Array 'rank_data_out' must have the same length as 'x' ({len(x)}), "
                         f"got {len(rank_data_out)}")

    if p[0] != 0:
        raise ValueError(f"First element of 'p' must be 0, got {p[0]}")
    if p[-1] != len(x):
        raise ValueError(f"Last element of 'p' must equal length of 'x' ({len(x)}), "
                        f"got {p[-1]}")
    if not np.all(np.diff(p) >= 0):
        raise ValueError("Array 'p' must be non-decreasing (valid CSC format)")
    if len(p) != ncol + 1:
        raise ValueError(f"Array 'p' must have length ncol + 1 = {ncol + 1}, "
                        f"got {len(p)}")
    
    # Call the C++ function
    try:
        result = fw.cpp_rank_matrix_csc(x, p, rank_data_out, nrow, ncol, nthreads)
        return result
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")
    
def rank_matrix_csr(
    x: np.ndarray,
    p: np.ndarray,
    i: np.ndarray,
    rank_data_out: np.ndarray,
    nrow: int,
    ncol: int,
    nthreads: int = 1
) -> dict:
    """
    Compute average ranks for non-zero elements in a CSR-formatted sparse matrix, column-wise.

    This function processes a sparse matrix in CSR format and computes average ranks
    of non-zero values **within each column**. Tied values receive the average of the
    ranks they would have occupied. Implicit zeros are treated as smallest values,
    and their count is used to shift non-zero ranks upward.

    Unlike `rank_matrix_csc`, this function **does not modify the input `data`**.
    Instead, it writes the resulting rank values into a pre-allocated output array
    `rank_data_out`, which must have the same length as `data`.
    Parameters
    ----------
    data : np.ndarray
        1D array of non-zero values in CSR format (dtype: float64).
        Length must equal the number of non-zero elements (nnz).
    p : np.ndarray
        1D array of row pointers (dtype: int32 or int64). Length = nrow + 1.
    i : np.ndarray
        1D array of column indices corresponding to each non-zero (dtype: int32 or int64).
        Length = nnz.
    rank_data_out : np.ndarray
        1D output array to store computed rank values (dtype: float64).
        Must be writable, C-contiguous, and have same length as `data`.
    nrow : int
        Number of rows in the matrix. Must be positive.
    ncol : int
        Number of columns in the matrix. Must be positive.
    nthreads : int, optional
        Number of OpenMP threads to use. If <= 0, uses all available threads.
        Default is 1.

    Returns
    -------
    dict
        A dictionary with three keys:
        - 'ties': list of list of float.
                  Length = number of columns. Each sublist contains the sizes
                  (as float) of tie groups for that column (only groups with size > 1) and zero numbers.
        - 'indptr': indptr for csc format, list of int.
                    Length = number of columns + 1.
        - 'indices': indices for csc format, list of int.
                     Length = number of non-zeros.

    Notes
    -----
    - Input arrays `data`, `p`, `i` are **not modified**.
    - `rank_data_out` is **overwritten** with rank values in an internal CSC-like order
      (grouped by column, but original per-column entry order is preserved via indexing).
    - The order of values in `rank_data_out` **does not match** the original `data` order.
      It is ordered column by column (like CSC `data`), not row by row (like CSR `data`).
    """

    # Array parameter validation
    x = validate_float_input('x', x, ndim=1)
    p = validate_int_input('p', p, ndim=1)
    i = validate_int_input('i', i, ndim=1)
    rank_data_out = validate_output_buffer('rank_data_out', rank_data_out, ndim=1)

    # Scalar parameter validation
    nrow = validate_int_scalar('nrow', nrow, positive=True)
    ncol = validate_int_scalar('ncol', ncol, positive=True)
    nthreads = standardiz_nthreads(nthreads)

    # validate csr matrix
    if len(x) != len(i):
        raise ValueError(f"Arrays 'x' and 'i' must have the same length, "
                        f"got x: {len(x)}, i: {len(i)}")
    if len(x) != len(rank_data_out):
        raise ValueError(f"Array 'rank_data_out' must have the same length as 'x' ({len(x)}), "
                         f"got {len(rank_data_out)}")
    
    if p[0] != 0:
        raise ValueError(f"First element of 'p' must be 0, got {p[0]}")
    if p[-1] != len(x):
        raise ValueError(f"Last element of 'p' must equal length of 'x' ({len(x)}), "
                        f"got {p[-1]}")
    if not np.all(np.diff(p) >= 0):
        raise ValueError("Array 'p' must be non-decreasing (valid CSC format)")
    if len(p) != nrow + 1:
        raise ValueError(f"Array 'p' must have length nrow + 1 = {nrow + 1}, "
                        f"got {len(p)}")

    if np.max(i) >= ncol:
        raise ValueError(f"Column indices in 'i' must be in range [0, {ncol-1}], "
                            f"found maximum column index: {np.max(i)}")
    
    # Call the C++ function
    try:
        result_dict = fw.cpp_rank_matrix_csr(x, p, i, rank_data_out, nrow, ncol, nthreads)
        return result_dict
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")

def rank_matrix_csr_(
    x: np.ndarray,
    p: np.ndarray,
    i: np.ndarray,
    rank_data_out: np.ndarray,
    nrow: int,
    ncol: int,
    nthreads: int = 1
) -> List[List[float]]:
    """
    Compute average ranks for non-zero elements in a CSR-formatted sparse matrix, column-wise.

    This function processes a sparse matrix in CSR format and computes average ranks
    of non-zero values **within each column**. Tied values receive the average of the
    ranks they would have occupied. Implicit zeros are treated as smallest values,
    and their count is used to shift non-zero ranks upward.

    Unlike `rank_matrix_csc`, this function **does not modify the input `data`**.
    Instead, it writes the resulting rank values into a pre-allocated output array
    `rank_data_out`, which must have the same length as `data`.
    Parameters
    ----------
    data : np.ndarray
        1D array of non-zero values in CSR format (dtype: float64).
        Length must equal the number of non-zero elements (nnz).
    p : np.ndarray
        1D array of row pointers (dtype: int32 or int64). Length = nrow + 1.
    i : np.ndarray
        1D array of column indices corresponding to each non-zero (dtype: int32 or int64).
        Length = nnz.
    rank_data_out : np.ndarray
        1D output array to store computed rank values (dtype: float64).
        Must be writable, C-contiguous, and have same length as `data`.
    nrow : int
        Number of rows in the matrix. Must be positive.
    ncol : int
        Number of columns in the matrix. Must be positive.
    nthreads : int, optional
        Number of OpenMP threads to use. If <= 0, uses all available threads.
        Default is 1.

    Returns
    -------
    dict
        A dictionary with three keys:
        - 'ties': list of list of float.
                  Length = number of columns. Each sublist contains the sizes
                  (as float) of tie groups for that column (only groups with size > 1) and zero numbers.
        - 'indptr': indptr for csc format, list of int.
                    Length = number of columns + 1.
        - 'indices': indices for csc format, list of int.
                     Length = number of non-zeros.

    Notes
    -----
    - Input arrays `data`, `p`, `i` are **not modified**.
    - `rank_data_out` is **overwritten** with rank values in an internal CSC-like order
      (grouped by column, but original per-column entry order is preserved via indexing).
    - The order of values in `rank_data_out` **does not match** the original `data` order.
      It is ordered column by column (like CSC `data`), not row by row (like CSR `data`).
    """

    # Array parameter validation
    x = validate_float_input('x', x, ndim=1)
    p = validate_int_input('p', p, ndim=1)
    i = validate_int_input('i', i, ndim=1)
    rank_data_out = validate_output_buffer('rank_data_out', rank_data_out, ndim=1)

    # Scalar parameter validation
    ncol = validate_int_scalar('ncol', ncol, positive=True)
    nrow = validate_int_scalar('nrow', nrow, positive=True)
    nthreads = standardiz_nthreads(nthreads)

    # validate csr matrix
    if len(x) != len(i):
        raise ValueError(f"Arrays 'x' and 'i' must have the same length, "
                        f"got x: {len(x)}, i: {len(i)}")
    if len(x) != len(rank_data_out):
        raise ValueError(f"Array 'rank_data_out' must have the same length as 'x' ({len(x)}), "
                         f"got {len(rank_data_out)}")
    
    if p[0] != 0:
        raise ValueError(f"First element of 'p' must be 0, got {p[0]}")
    if p[-1] != len(x):
        raise ValueError(f"Last element of 'p' must equal length of 'x' ({len(x)}), "
                        f"got {p[-1]}")
    if not np.all(np.diff(p) >= 0):
        raise ValueError("Array 'p' must be non-decreasing (valid CSC format)")
    if len(p) != nrow + 1:
        raise ValueError(f"Array 'p' must have length nrow + 1 = {nrow + 1}, "
                        f"got {len(p)}")

    if np.max(i) >= ncol:
        raise ValueError(f"Column indices in 'i' must be in range [0, {ncol-1}], "
                            f"found maximum column index: {np.max(i)}")
    
    # Call the C++ function
    try:
        result = fw.cpp_rank_matrix_csr_(x, p, i, rank_data_out, nrow, ncol, nthreads)
        return result
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")
    
def rank_matrix_dense(
    x: np.ndarray,
    nthreads: int = 1
) -> dict:
    """
    Compute average ranks for each column of a dense matrix.

    This function replaces each element in the input matrix with its average rank
    within its column. Tied values receive the average of the ranks they would have
    occupied. Ranks start at 1 (smallest value gets rank 1).

    Parameters
    ----------
    x : np.ndarray
        2D dense input matrix (dtype: float64). Shape: (nrows, ncols).
        Must be non-empty (at least one row and one column).
    nthreads : int, optional
        Number of OpenMP threads to use. If <= 0, uses all available threads.
        Default is 1.

    Returns
    -------
    dict
        A dictionary with two keys:
        - 'X_ranked': np.ndarray of shape (nrows, ncols), dtype float64.
                      Same shape as input, with values replaced by column-wise ranks.
        - 'ties': list of list of float.
                  Length = number of columns. Each sublist contains the sizes
                  (as float) of tie groups for that column (only groups with size > 1).

    Notes
    -----
    - Input matrix `x` is not modified; a new array is returned.
    - Ranking is column-wise and independent.
    """

    # Array parameter validation
    x = validate_float_input('x', x, ndim=2, allow_zero=True)
    # Scaler value validation
    nthreads = standardiz_nthreads(nthreads)

    # Call C++ function
    try:
        result_dict = fw.cpp_rank_matrix_dense(x, nthreads)
        return result_dict
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")

def group_rank_csc(
    x: np.ndarray,
    p: np.ndarray,
    i: np.ndarray,
    rank_data_out: np.ndarray,
    groups: np.ndarray,
    nrow: int,
    ncol: int,
    nthreads: int = 1
) -> None:
    """
    Compute average ranks for non-zero elements in a CSC-formatted sparse matrix,
    grouped by specified column groups.

    This function assigns ranks to non-zero values within each group of columns
    in a sparse matrix stored in CSC format. Zeros are implicitly treated as the smallest
    values and are not stored, the rank starts from 1.

    The function modifies the input array `rank_data_out` in-place by replacing non-zero values
    with their computed ranks (adjusted for implicit zeros).

    Parameters
    ----------
    x : np.ndarray
        1D array of non-zero values in CSC format (dtype: float64).
        Length must equal the number of non-zero elements (nnz).
    p : np.ndarray
        1D array of column pointers in CSC format (dtype: int32 or int64).
        Length must be `ncol + 1`, with `p[0] == 0` and non-decreasing.
    i : np.ndarray
        1D array of row indices corresponding to each non-zero (dtype: int32 or int64).
        Length must equal the number of non-zero elements (nnz).
    rank_data_out : np.ndarray
        1D output array to store computed rank values (dtype: float64).
        Must be writable, contiguous, and have same length as `x`.
    groups : np.ndarray
        1D array assigning each row to a group (dtype: any integer type).
        Length must be equal to number of rows in the matrix.
        Group indices must be in range [0, ngroups - 1].
    ncol : int
        Number of columns in the matrix. Must be positive.
    nrow : int
        Number of rows in the matrix. Must be positive.
        Just for input tests, not used for calc.
    nthreads : int, optional
        Number of OpenMP threads to use. If <= 0, uses all available threads.
        Default is 1.
    Returns
    -------
    None
        The function modifies `rank_data_out` in-place; no return value.
    """
    # Type validation
    x = validate_float_input('x', x, ndim=1)
    p = validate_int_input('p', p, ndim=1)
    i = validate_int_input('i', i, ndim=1)
    groups = validate_int_input('groups', groups, ndim=1)
    rank_data_out = validate_output_buffer('rank_data_out', rank_data_out, ndim=1)
    
    # Scalar value validation
    ncol = validate_int_scalar('ncol', ncol, positive=True)
    nrow = validate_int_scalar('nrow', nrow, positive=True)
    nthreads = standardiz_nthreads(nthreads)

    # validate csc matrix
    if len(x) != len(rank_data_out):
        raise ValueError(f"Array 'rank_data_out' must have the same length as 'x' ({len(x)}), "
                         f"got {len(rank_data_out)}")
    validate_spmatrix_input(x, p, i, groups, np.max(groups) + 1,
                            nrow, ncol, cell_num=nrow, matrix_type='csc')
    
    # Call the C++ function
    try:
        fw.cpp_group_rank_csc(x, rank_data_out, p, i, groups, ncol, nthreads)
        return rank_data_out
    except Exception as e:
        raise RuntimeError(f"C++ function failed: {e}")

def group_rank_dense(
    x: np.ndarray,
    rank_data_out: np.ndarray,
    groups: np.ndarray,
    nthreads: int = 1
) -> None:
    """
    Compute average ranks for non-zero elements in a dense matrix.

    This function assigns ranks to non-zero values within each group of columns
    in a dense matrix. Zeros are implicitly treated as the smallest values and 
    are not stored, the rank started from 1.

    The function modifies the input array `rank_data_out` in-place by replacing non-zero values
    with their computed ranks (adjusted for implicit zeros).

    Parameters
    ----------
    x : np.ndarray
        2D dense input matrix (dtype: float64). Shape: (nrows, ncols).
    rank_data_out : np.ndarray
        2D output array to store computed rank values (dtype: float64).
        Must be writable, contiguous, and have same shape as `x`.
    groups : np.ndarray
        1D array assigning each row to a group (dtype: any integer type).
        Length must be equal to number of rows in the matrix.
        Group indices must be in range [0, ngroups - 1].
    nthreads : int, optional
        Number of OpenMP threads to use. If <= 0, uses all available threads.
        Default is 1.
    Returns
    -------
    None
        The function modifies `rank_data_out` in-place; no return value.
    """
    # Type validation
    x = validate_float_input('x', x, ndim=2, allow_zero=True)
    rank_data_out = validate_output_buffer('rank_data_out', rank_data_out, ndim=2)
    groups = validate_int_input('groups', groups, ndim=1)
    nthreads = standardiz_nthreads(nthreads)

    # validate dense matrix
    nrow, _ = x.shape
    validate_denmatrix_input(groups, np.max(groups) + 1, cell_num=nrow)
    # call c++ function
    try:
        fw.cpp_group_rank_dense(x, rank_data_out, groups, nthreads)
        return rank_data_out
    except Exception as e:
        print(f"C++ function failed: {e}")
