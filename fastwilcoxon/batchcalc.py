from dataclasses import dataclass
from typing import Optional, Union, List, Tuple, Dict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from .utils import rank_matrix, sum_groups, nnz_groups, compute_pval
from .wilcoxauc import prefilter_matrix


@dataclass
class MarkerTestCache:
    X: Union[sp.csc_matrix, sp.csr_matrix, np.ndarray]
    ranks: Union[sp.csc_matrix, np.ndarray]
    ties: List[List[float]]
    gene_names: np.ndarray
    obs_names: np.ndarray
    
    n_obs: int
    n_vars: int

    source: str
    is_sparse: bool
    nthreads: int

    @classmethod
    def from_adata(
        cls, 
        adata: AnnData, 
        *, 
        layer: Optional[str] = None,
        use_raw: bool = False,
        copy: bool = True,
        nthreads: int = -1,
    ) -> "MarkerTestCache":
        """Create a MarkerTestCache from an AnnData object."""

        adata = prefilter_matrix(adata, layer=layer, use_raw=use_raw, copy=copy)

        X = adata.X
        gene_names = adata.var_names.values
        source = "adata.raw.X" if use_raw else "adata.X"

        if layer is not None:
            X = adata.layers[layer]
            source = f"adata.layers[{layer!r}]"

        obs_names = adata.obs_names.values
        rank_result = rank_matrix(X, nthreads=nthreads)
        ranks = rank_result["X_ranked"]
        ties = rank_result["ties"]

        return cls(
            X=X,
            ranks=ranks,
            ties=ties,
            gene_names=gene_names,
            obs_names=obs_names,
            n_obs=X.shape[0],
            n_vars=X.shape[1],
            source=source,
            is_sparse=sp.issparse(X),
            nthreads=nthreads,
        )

    @classmethod
    def from_matrix(
        cls,
        matrix: Union[sp.csc_matrix, sp.csr_matrix, np.ndarray],
        gene_names: np.ndarray,
        obs_names: np.ndarray,
        nthreads: int = -1,
    ) -> "MarkerTestCache":
        """Create a MarkerTestCache from a matrix."""

        matrix, gene_names = prefilter_matrix(matrix, var_names=gene_names)
        gene_names = np.asarray(gene_names)
        rank_result = rank_matrix(matrix, nthreads=nthreads)
        ranks = rank_result["X_ranked"]
        ties = rank_result["ties"]

        return cls(
            X=matrix,
            ranks=ranks,
            ties=ties,
            gene_names=gene_names,
            obs_names=obs_names,
            n_obs=matrix.shape[0],
            n_vars=matrix.shape[1],
            source="matrix",
            is_sparse=sp.issparse(matrix),
            nthreads=nthreads,
        )

    def wilcox(self, labels, **kwargs):
        return marker_test(self, labels, **kwargs)
    
    def wilcox_batch(self, labels_array, **kwargs):
        return marker_test_batch(self, labels_array, **kwargs)
    

def _normalize_single_labels(
    cache: MarkerTestCache,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate, align and encode one label vector.

    Returns
    -------
    encoded_labels
        Contiguous int64 array with values 0, ..., n_groups - 1.
    group_names
        Original group names. Row order of returned statistics follows
        this array.
    """

    # align labels with cache.obs_names if labels is a pd.Series
    if isinstance(labels, pd.Series):
        expected_index = pd.Index(cache.obs_names)
        missing = expected_index.difference(labels.index)
        if len(missing) > 0:
            raise ValueError(
                f"labels are missing {len(missing)} cached observations"
            )
        labels_array = labels.reindex(expected_index).to_numpy()

        if labels_array.isna().any():
            raise ValueError("Some adata observations do not have labels.")
    else:
        labels_array = np.asarray(labels)

    # Validate shape
    if labels_array.ndim != 1:
        raise ValueError(
            f"labels must be one-dimensional, got shape {labels_array.shape}"
        )
    if labels_array.shape[0] != cache.n_obs:
        raise ValueError(
            "Length of labels must match cache.n_obs: "
            f"expected {cache.n_obs}, got {labels_array.shape[0]}"
        )

    group_names, encoded_labels = np.unique(
        labels_array,
        return_inverse=True,
    )

    if len(group_names) < 2:
        raise ValueError(
            f"At least two groups are required for marker testing, "
            f"but only {len(group_names)} group(s) were found."
        )
    
    encoded_labels = np.ascontiguousarray(
        encoded_labels,
        dtype=np.int64,
    )
    return encoded_labels, group_names


def _encode_binary_label_array(
    label_array,
    *,
    target_group
) -> Dict[str, object]:

    """
    Encode binary label array into contiguous integers 0 and 1.

    Returns
    -------
    encoded
        int8 array with values 0 and 1.
    group_names
        group_names[0] corresponds to code 0 and group_names[1]
        corresponds to code 1.
    label_to_code
        Mapping from original label to code {0, 1}
    code_to_label
        Mapping from code {0, 1} to original label
    """

    values = np.asarray(label_array)
    if values.ndim != 2:
        raise ValueError("label_array must be a 2D array")
    if np.any(pd.isna(values)):
        raise ValueError("label_array contains NaN values")

    group_names = np.unique(values)
    if group_names.size != 2:
        raise ValueError(
            "Binary marker testing requires exactly two groups, "
            f"got {group_names.size}: {group_names.tolist()}"
        )
    target_matches = np.asarray(
        [group == target_group for group in group_names], dtype=bool
    )
    if np.count_nonzero(target_matches) != 1:
        raise ValueError(
            f"target_group '{target_group}' was not unqiue in label array"
        )
    # set target group as 1 and other group as 0
    membership = np.ascontiguousarray(
        values == target_group,
        dtype = np.int8
    )
    ref_group = group_names[~target_matches][0]
    return {
        "membership": membership,
        "group_names": np.asarray([ref_group, target_group]),
        "label_to_code": {ref_group: 0, target_group: 1},
        "code_to_label": {0: ref_group, 1: target_group}
        }


def _to_dense_array(value) -> np.ndarray:
    """Convert a matrix-product result to a dense ndarray."""
    if sp.issparse(value):
        return value.toarray()
    else:
        return np.asarray(value)
    

def marker_test(
    cache: MarkerTestCache,
    labels,
    *,
    corr_method: str = "benjamini-hochberg",
    nthreads: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute Wilcoxon AUC statistics using a MarkerTestCache.

    Returns
    ----------
    Dictionary containing the same statistics as
        wilcoxauc.py::_wilcoxauc_core(). Every array has shape
        (n_groups, n_vars).
    """

    if corr_method not in {
        "benjamini-hochberg",
        "bonferroni",
    }:
        raise ValueError(
            "corr_method must be 'benjamini-hochberg' "
            "or 'bonferroni'"
        )

    if nthreads is None:
        nthreads = cache.nthreads

    encoded_labels, group_names = _normalize_single_labels(cache, labels)
    group_size = np.bincount(encoded_labels).astype(np.int64, copy=False)
    n_groups = len(group_size)
    n_cells = cache.n_obs

    rank_sum = sum_groups(
        cache.ranks,
        encoded_labels,
        trans=False,
        nthreads=nthreads,
    )
    group_sum = sum_groups(
        cache.X,
        encoded_labels,
        trans=False,
        nthreads=nthreads
    )
    group_nnz = nnz_groups(
        cache.X,
        encoded_labels,
        trans=False,
        nthreads=nthreads
    )
    rank_sum = np.asarray(rank_sum, dtype=np.float64)
    group_sum = np.asarray(group_sum, dtype=np.float64)
    group_nnz = np.asarray(group_nnz, dtype=np.int64)

    #! Delete after debugging
    expected_shape = (n_groups, cache.n_vars)

    if rank_sum.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected rank_sum shape: {rank_sum.shape}; "
            f"expected {expected_shape}"
        )

    if group_sum.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected group_sum shape: {group_sum.shape}; "
            f"expected {expected_shape}"
        )

    if group_nnz.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected group_nnz shape: {group_nnz.shape}; "
            f"expected {expected_shape}"
        )

    # U Statistics
    group_size_2d = group_size.reshape(-1, 1)
    if sp.issparse(cache.X):
        gnz = group_size_2d - group_nnz
        zero_ranks = (1 + np.sum(gnz, axis=0)) / 2
        ustat = gnz * zero_ranks + rank_sum - group_size_2d * (group_size_2d + 1) / 2
    else:
        ustat = rank_sum - group_size_2d * (group_size_2d + 1) / 2
    
    # AUC, z-score, and wilcoxon p-value
    n1n2 = group_size_2d * (n_cells - group_size_2d)
    pval, z_norm = compute_pval(ustat, cache.ties, n_cells, n1n2)
    auc = ustat / n1n2

    # Multiple testing correction
    fdr = np.full_like(pval, fill_value=np.nan, dtype=float)
    for g in range(n_groups):
        valid = ~np.isnan(pval[g, :])
        if np.any(valid):
            _, fdr[g, valid], _, _ = multipletests(
                pval[g, valid],
                alpha=0.05,
                method='fdr_bh' if corr_method == 'benjamini-hochberg' else 'bonferroni'
            )
    
    # pct1 and pct2
    group_mean = group_sum / group_size_2d
    pct_1 = group_nnz / group_size_2d * 100.0
    total_nnz = np.sum(group_nnz, axis=0)
    pct_2 = (total_nnz - group_nnz) / (n_cells - group_size_2d) * 100.0

    # log fold change
    rest_mean = ((np.sum(group_sum, axis=0, keepdims=True) - group_sum) / 
                 (n_cells - group_size_2d))
    lfc = np.log2((group_mean + 1e-9) / (rest_mean + 1e-9))

    group_index = pd.Index(group_names, name="group")
    gene_columns = pd.Index(cache.gene_names, name="gene")

    results = {
        "avgExpr": pd.DataFrame(group_mean, index=group_index, columns=gene_columns),
        "logfoldchanges": pd.DataFrame(lfc, index=group_index, columns=gene_columns),
        "score": pd.DataFrame(z_norm, index=group_index, columns=gene_columns),
        "auc": pd.DataFrame(auc, index=group_index, columns=gene_columns),
        "pvals": pd.DataFrame(pval, index=group_index, columns=gene_columns),
        "padj": pd.DataFrame(fdr, index=group_index, columns=gene_columns),
        "pct1": pd.DataFrame(pct_1, index=group_index, columns=gene_columns),
        "pct2": pd.DataFrame(pct_2, index=group_index, columns=gene_columns),
    }
    return results


def marker_test_batch(
    cache: MarkerTestCache,
    label_array: np.ndarray,
    *,
    corr_method: str = "benjamini-hochberg",
    target_group: str = "left",
    auc_only: bool = False,
):
    # "membership", "group_names", "label_to_code", "code_to_label"
    encoded_result = _encode_binary_label_array(label_array, target_group=target_group)
    membership_1 = encoded_result["membership"]
    n1 = np.sum(
        membership_1,
        axis=1,
        dtype=np.int64,
        keepdims=True
    ) # shape: (n_permutation, 1)
    n0 = cache.n_obs - n1 # shape: (n_permutation, 1)

    # rank sum, total_nnz, target_nnz and total_sum
    target_rank_nnz = _to_dense_array(membership_1 @ cache.ranks).astype(np.float64, copy=False)
    if sp.issparse(cache.ranks):
        X_binary = cache.X.copy()
        X_binary.data = np.ones(
            X_binary.nnz,
            dtype=np.float64,
        )
        total_nnz = np.asarray(
            X_binary.sum(axis=0),
            dtype=np.float64
        ).reshape(1, -1) # shape: (1, n_gene)
        total_nz = cache.n_obs - total_nnz
        target_nnz = _to_dense_array(membership_1 @ X_binary).astype(np.float64, copy=False)
        target_nz = n1 - target_nnz
        rank_sum = target_rank_nnz + target_nz * (total_nz + 1) / 2.0
        total_sum = np.asarray(
            cache.X.sum(axis=0),
            dtype=np.float64,
        ).reshape(1, -1) # shape: (1, n_gene)
    else:
        X_binary = (cache.X > 0).astype(np.float64)
        total_nnz = np.sum(X_binary, axis=0, keepdims=True)
        target_nnz = _to_dense_array(membership_1 @ X_binary).astype(np.float64, copy=False)
        rank_sum = target_rank_nnz
        total_sum = np.sum(cache.X, axis=0, keepdims=True) # shape: (1, n_gene)

    # ustat and auc
    rank_offset = n1 * (n1 + 1) / 2.0 # shape: (n_permutation, 1)
    ustat = (rank_sum - rank_offset)
    n1n2 = n1 * n0 # shape: (n_permutation, 1)
    auc = ustat / n1n2
    if auc_only:
        return {
            "auc": auc,
            "gene_names": cache.gene_names,
            "group_names": encoded_result["code_to_label"]
        }

    # pvalue and adjusted-pvalue
    z = ustat - 0.5 * n1n2
    z = z - np.sign(z) * 0.5
    # tie correction
    # sigma = sqrt(n1n2*(N^3-N)*(sum(t^3-t))/12/(N^2-N))
    x1 = cache.n_obs ** 3 - cache.n_obs
    x2 = 1.0 / (12 * (cache.n_obs ** 2 - cache.n_obs))
    sigma = np.zeros(len(cache.ties), dtype=np.float64)
    for j, tie_values in enumerate(cache.ties):
        if len(tie_values) > 0:
            tie_correction = sum(t ** 3 - t for t in tie_values)
            sigma[j] = (x1 - tie_correction) * x2
        else:
            sigma[j] = x1 * x2
    invalid_cols = np.isclose(sigma, 0.0) | (sigma < 0)
    valid_cols = ~invalid_cols
    pvals = np.full_like(ustat, fill_value=np.nan, dtype=float)
    adj_pvals = np.full_like(ustat, fill_value=np.nan, dtype=float)
    z_norm = np.full_like(ustat, fill_value=np.nan, dtype=float)
    if np.any(valid_cols):
        # calc z and p values
        u_sigma = np.sqrt(n1n2 * sigma[valid_cols][np.newaxis, :])
        z_valid = z[:, valid_cols] / u_sigma
        p_valid = 2 * norm.cdf(-np.abs(z_valid))
        # copy to result matrix
        z_norm[:, valid_cols] = z_valid
        pvals[:, valid_cols] = p_valid
        # calc adjusted p-values
        for i in range(p_valid.shape[0]):
            _, adj_pvals[i, valid_cols], _, _ = multipletests(
                p_valid[i, :],
                alpha=0.05,
                method='fdr_bh' if corr_method == 'benjamini-hochberg' else 'bonferroni'
            )
    
    # pct1, pct2, lfc and avgExpr with shape of (n_permutation, n_gene)
    pct1 = target_nnz / n1 * 100.0
    pct2 = (total_nnz - target_nnz) / n0 * 100.0
    # log fold change
    target_sum = _to_dense_array(
        membership_1 @ cache.X
    ).astype(np.float64, copy=False)
    other_sum = total_sum - target_sum
    target_mean = target_sum / n1
    other_mean = other_sum / n0
    lfc = np.log2((target_mean + 1e-9) / (other_mean + 1e-9))

    return {
        "avgExpr": target_mean,
        "logfoldchanges": lfc,
        "score": z_norm,
        "auc": auc,
        "pval": pvals,
        "padj": adj_pvals,
        "pct1": pct1,
        "pct2": pct2,
        "gene_names": cache.gene_names,
        "group_names": encoded_result["code_to_label"],
    }