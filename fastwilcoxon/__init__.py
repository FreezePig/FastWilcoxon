from .wilcoxauc import calc_gini, prefilter_matrix, wilcoxauc, find_all_markers, find_markers
from .batchcalc import MarkerTestCache, marker_test, marker_test_batch
from .plot import volcano_plot, detection_contrast_plot, marker_exclusivity_plot

__all__ = ["wilcoxauc",
           "prefilter_matrix",
           "calc_gini",
           "find_all_markers",
           "find_markers",
           "MarkerTestCache",
           "marker_test",
           "marker_test_batch",
           "volcano_plot",
           "detection_contrast_plot",
           "marker_exclusivity_plot"]

# show version
__version__ = "0.1.0"