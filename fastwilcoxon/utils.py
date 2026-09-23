import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import norm

from typing import Union, Tuple, List, Dict, Any
from .wrappers import *

import time
from functools import singledispatch

# ====== 1. ranking functions ======

@singledispatch
def rank_matrix(X, nthreads: int = 1):
    """General interface for matrix sorting"""
    raise NotImplementedError(f"rank_matrix not implemented for type {type(X)}")

@rank_matrix.register
def _(X: np.ndarray, nthreads: int = 1) -> Dict:
    """Rank the rows of a dense numpy array"""
    return rank_matrix_dense(X, nthreads)

@rank_matrix.register
def _(X: sp.csr_matrix, nthreads: int = 1) -> Dict:
    """Rank the rows of a sparse CSR matrix"""
    X_csc = X.tocsc()

    rank_data_out = np.zeros_like(X_csc.data, dtype=np.float64)
    ties = rank_matrix_csc(X_csc.data, X_csc.indptr, rank_data_out, 
                           X_csc.shape[0], X_csc.shape[1], nthreads)
    X_ranked = sp.csc_matrix((rank_data_out, X_csc.indices, X_csc.indptr), shape=X_csc.shape)
    return {'X_ranked': X_ranked, 'ties': ties}
    
@rank_matrix.register  
def _(X: sp.csc_matrix, nthreads: int = 1) -> Dict:
    """CSC sparse matrix ranking"""
    # Modify X.data in place
    X_copy = X.copy()  # Avoid modifying the original matrix
    rank_data_out = np.zeros_like(X_copy.data, dtype=np.float64)
    ties = rank_matrix_csc(X_copy.data, X_copy.indptr, rank_data_out, 
                           X.shape[0], X.shape[1], nthreads)
    X_ranked = sp.csc_matrix((rank_data_out, X_copy.indices, X_copy.indptr), shape=X_copy.shape)
    return {'X_ranked': X_ranked, 'ties': ties}

# ====== 2. sum groups function ======

@singledispatch
def sum_groups(X, groups: np.ndarray, trans: bool = False, nthreads: int = 1):
    """General interface for summing groups in a matrix"""
    raise NotImplementedError(f"sum_groups not implemented for type {type(X)}")

@sum_groups.register
def _(X: np.ndarray, groups: np.ndarray, trans: bool = False, nthreads: int = 1) -> np.ndarray:
    """Sum groups in a dense numpy array"""
    ngroups = len(np.unique(groups))
    if not trans:
        return sumGroups_dense(X, groups, ngroups)
    else: 
        return sumGroups_dense_T(X, groups, ngroups)
    
@sum_groups.register
def _(X: sp.csc_matrix, groups: np.ndarray, trans: bool = False, nthreads: int = 1) -> np.ndarray:
    """Sum groups in a sparse CSC matrix"""
    ngroups = len(np.unique(groups))
    if not trans:
        return sumGroups_csc(X.data, X.indptr, X.indices, 
                             X.shape[0], X.shape[1], groups, ngroups, nthreads)
    else:
        return sumGroups_csc_T(X.data, X.indptr, X.indices,
                               X.shape[0], X.shape[1], groups, ngroups, nthreads)

@sum_groups.register
def _(X: sp.csr_matrix, groups: np.ndarray, trans: bool = False, nthreads: int = 1) -> np.ndarray:
    """Sum groups in a sparse CSR matrix"""
    ngroups = len(np.unique(groups))
    if not trans:
        return sumGroups_csr(X.data, X.indptr, X.indices,
                             X.shape[0], X.shape[1], groups, ngroups, nthreads)
    else:
        return sumGroups_csr_T(X.data, X.indptr, X.indices,
                               X.shape[0], X.shape[1], groups, ngroups, nthreads)

# ====== 3. nnzero groups function ======

@singledispatch
def nnz_groups(X, groups: np.ndarray, trans: bool = False, nthreads: int = 1) -> np.ndarray:
    """General interface for counting non-zero elements in groups of a matrix"""
    raise NotImplementedError(f"nnz_groups not implemented for type {type(X)}")

@nnz_groups.register
def _(X: np.ndarray, groups: np.ndarray, trans: bool = False, nthreads: int = 1) -> np.ndarray:
    """Count non-zero elements in groups of a dense numpy array"""
    ngroups = len(np.unique(groups))
    if not trans:
        return nnzeroGroups_dense(X, groups, ngroups, nthreads)
    else:
        return nnzeroGroups_dense_T(X, groups, ngroups, nthreads)

@nnz_groups.register
def _(X: sp.csc_matrix, groups: np.ndarray, trans: bool = False, nthreads: int = 1) -> np.ndarray:
    """Count non-zero elements in groups of a sparse CSC matrix"""
    ngroups = len(np.unique(groups))
    if not trans:
        return nnzeroGroups_csc(X.indptr, X.indices, X.shape[0], 
                                X.shape[1], groups, ngroups, nthreads)
    else:
        return nnzeroGroups_csc_T(X.indptr, X.indices, X.shape[0], 
                                  X.shape[1], groups, ngroups, nthreads)

@nnz_groups.register
def _(X: sp.csr_matrix, groups: np.ndarray, trans: bool = False, nthreads: int = 1) -> np.ndarray:
    """Count non-zero elements in groups of a sparse CSR matrix"""
    ngroups = len(np.unique(groups))
    if not trans:
        return nnzeroGroups_csr(X.indptr, X.indices, X.shape[0], 
                                X.shape[1], groups, ngroups, nthreads)
    else:
        return nnzeroGroups_csr_T(X.indptr, X.indices, X.shape[0], 
                                  X.shape[1], groups, ngroups, nthreads)
    
# ====== 4. group ranking function ======

@singledispatch
def group_rank(X, groups:np.ndarray, nthreads: int = 1):
    """General interface for grouply ranking matrix generating(for gini coefficient)"""
    raise NotImplementedError(f"rank_matrix not implemented for type {type(X)}")

@group_rank.register
def _(X: np.ndarray, groups: np.ndarray, nthreads: int = 1) -> dict:
    """Grouply rank the cols of a dense numpy array"""
    rank_data_out = np.zeros_like(X, dtype=np.float64)
    group_rank_dense(X, rank_data_out, groups, nthreads)
    return {'ranked': rank_data_out, 'X': X}

@group_rank.register
def _(X: sp.csr_matrix, groups: np.ndarray, nthreads: int = 1) -> dict:
    """Grouply rank the cols of a sparse CSR matrix"""
    X_csc = X.tocsc()
    rank_data_out = np.zeros_like(X_csc.data, dtype=np.float64)
    group_rank_csc(X_csc.data, X_csc.indptr, X_csc.indices, rank_data_out,
                   groups, X_csc.shape[0], X_csc.shape[1], nthreads)
    X_group_ranked = sp.csc_matrix((rank_data_out, X_csc.indices, X_csc.indptr), shape=X_csc.shape)
    return {'ranked': X_group_ranked, 'X': X_csc}

@group_rank.register
def _(X: sp.csc_matrix, groups: np.ndarray, nthreads: int = 1) -> dict:
    """Grouply rank the cols of a sparse CSC matrix"""
    X_copy = X.copy()  # Avoid modifying the original matrix
    rank_data_out = np.zeros_like(X_copy.data, dtype=np.float64)
    group_rank_csc(X_copy.data, X_copy.indptr, X_copy.indices, rank_data_out,
                   groups, X_copy.shape[0], X_copy.shape[1], nthreads)
    X_group_ranked = sp.csc_matrix((rank_data_out, X_copy.indices, X_copy.indptr), shape=X_copy.shape)
    return {'ranked': X_group_ranked, 'X': X_copy}
    
# ====== 5. Additional function for statistics ======
def compute_ustats(X_rank: Union[np.ndarray, sp.csr_matrix, sp.csc_matrix], 
                   groups: np.ndarray, group_size: np.ndarray, nthreads: int = 1) -> np.ndarray:
    """Compute U statistics for groups in a matrix"""
    # group rank sum with zero uncalculated(for sparse matrix)
    grs = sum_groups(X_rank, groups, nthreads=nthreads)

    # Dealing with sparse matrix
    if sp.issparse(X_rank):
        # calc rank sum for zero values
        # size: [n_groups, n_genes]
        gnz = group_size[:, np.newaxis] - nnz_groups(X_rank, groups, nthreads=nthreads)
        # average rank for zero values, using the equation for arithmetic sequence sum
        # size: [n_genes]
        zero_ranks = (1 + np.sum(gnz, axis=0)) / 2
        ustat = gnz * zero_ranks[np.newaxis, :] + grs - group_size[:, np.newaxis] * (group_size[:, np.newaxis] + 1) / 2
    else:
        # dense mamtrix
        ustat = grs - group_size[:, np.newaxis] * (group_size[:, np.newaxis] + 1) / 2
    
    return ustat

def compute_pval(ustat: np.ndarray, ties: List, N: int, n1n2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute p-values from U statistics"""
    
    # Ensure n1n2 is 1D for proper broadcasting
    n1n2 = n1n2.ravel()

    # continuous correction
    z = ustat - 0.5 * n1n2[:, np.newaxis]
    z = z - np.sign(z) * 0.5

    # tie correction
    # sigma_cor = sqrt(n1n2 * x2 * (x1 - sum(t^3 - t)))
    x1 = N ** 3 - N
    x2 = 1.0 / (12 * (N ** 2 - N))

    # calc tie correction factor for each col(gene)
    sigma = np.zeros(len(ties))
    for j, tie_values in enumerate(ties):
        if len(tie_values) > 0:
            # if zero number = 1, t^3 - t still zero
            tie_correction = sum(t ** 3 - t for t in tie_values)
            sigma[j] = (x1 - tie_correction) * x2
        else:
            sigma[j] = x1 * x2

    invalid_cols = np.isclose(sigma, 0.0) | (sigma < 0)
    valid_cols = ~invalid_cols

    pvals = np.full_like(ustat, fill_value=np.nan, dtype=float)
    z_norm = np.full_like(ustat, fill_value=np.nan, dtype=float)

    if np.any(valid_cols):
        # calc z and p values
        u_sigma = np.sqrt(n1n2[:, np.newaxis] * sigma[valid_cols][np.newaxis, :])
        z_valid = z[:, valid_cols] / u_sigma
        p_valid = 2 * norm.cdf(-np.abs(z_valid))
        # copy values
        z_norm[:, valid_cols] = z_valid
        pvals[:, valid_cols] = p_valid
    return pvals, z_norm

def compute_nnz_gini(X: Union[np.ndarray, sp.csc_matrix, sp.csr_matrix],
                 groups: np.ndarray, nthreads: int = 1) -> np.ndarray:
    """Compute Gini coefficients for groups in a matrix based on non-zero counts"""
    group_nnz = nnz_groups(X, groups, nthreads=nthreads)
    group_sum = sum_groups(X, groups, nthreads=nthreads)
    group_rank_result = group_rank(X, groups, nthreads)
    ranks = group_rank_result['ranked']
    new_X = group_rank_result['X']
    if sp.isspmatrix_csc(ranks):
        mut_X = sp.csc_matrix((ranks.data*new_X.data, new_X.indices, new_X.indptr), shape=new_X.shape)
    elif isinstance(ranks, np.ndarray):
        mut_X = ranks * new_X
    else:
        raise TypeError("Unsupported type for group_rank")
    
    x1 = sum_groups(mut_X, groups, nthreads=nthreads)
    x2 = group_nnz * group_sum
    gini = 2 * x1 / x2 - (group_nnz + 1) / group_nnz
    return gini


def wide2long(wide_res_dict: dict, features: list, groups: list):
    """transform wide format results to long format"""
    n_features = len(features)
    n_groups = len(groups)

    feature_col = features * n_groups
    group_col = []
    for group in groups:
        group_col.extend([group] * n_features)
    result_dict = {
        'gene': feature_col,
        'cluster': group_col
    }

    for stat_name, stat_matrix in wide_res_dict.items():
        stat_vector = stat_matrix.flatten(order='C')
        result_dict[f'{stat_name}'] = stat_vector
    
    df = pd.DataFrame(result_dict)
    desired_order = ['gene', 'cluster', 'avgExpr', 'logfoldchanges', 'score', 
                    'auc', 'pval', 'padj', 'pct_1', 'pct_2']
    available_cols = [col for col in desired_order if col in df.columns]
    other_cols = [col for col in df.columns if col not in available_cols]

    return df[available_cols + other_cols]