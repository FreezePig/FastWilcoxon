from .wilcoxauc import calc_gini, prefilter_matrix, wilcoxauc
from .batchcalc import MarkerTestCache, marker_test, marker_test_batch

__all__ = ["wilcoxauc",
           "prefilter_matrix",
           "calc_gini",
           "MarkerTestCache",
           "marker_test",
           "marker_test_batch"]

# show version
__version__ = "0.1.0"