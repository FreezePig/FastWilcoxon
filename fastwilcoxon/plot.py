"""Marker gene visualizations for FastWilcoxon results."""

from typing import Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm, SymLogNorm


_POINT_SIZES = (1, 60)
_PADJ_THRESHOLD = 0.05
_TITLE_FONT_SIZE = 12
_AXIS_FONT_SIZE = 12
_TICK_FONT_SIZE = 10
_LEGEND_FONT_SIZE = 10
_ANNOTATION_FONT_SIZE = 6
_LOGFC_CMAP = LinearSegmentedColormap.from_list(
    "fastwilcoxon_logfc",
    ["#2166AC", "#67A9CF", "#F4E3B2", "#EF8A62", "#B2182B"],
)
_AUC_CMAP = LinearSegmentedColormap.from_list(
    "fastwilcoxon_auc",
    ["#FFFBF0", "#FEE8C8", "#FDBB84", "#E34A33", "#99000D"],
)
_AUC_NORM = PowerNorm(gamma=0.55, vmin=0.0, vmax=1.0)
_ANNOTATION_KEYS = ("AUC", "adjp", "logfc", "delta_pct", "delta_logfc", "delta_pct2")
_Annotation = Optional[Union[int, str, Sequence[str], Mapping[str, int]]]

__all__ = ["volcano_plot", 
           "detection_contrast_plot", 
           "marker_exclusivity_plot"]


def _prepare_marker_df(
    marker_df: pd.DataFrame,
    cluster,
    columns: Tuple[str, ...],
) -> Tuple[pd.DataFrame, object]:
    required = {"cluster", *columns}
    missing = sorted(required.difference(marker_df.columns))
    if missing:
        raise ValueError(f"marker_df is missing required columns: {missing}")

    clusters = marker_df["cluster"].dropna().unique()
    if cluster is None:
        if len(clusters) != 1:
            raise ValueError("cluster must be specified when marker_df contains multiple clusters")
        cluster = clusters[0]
    elif cluster not in clusters:
        raise ValueError(f"cluster {cluster!r} was not found in marker_df")

    plot_df = marker_df.loc[marker_df["cluster"] == cluster].copy()
    plot_df = plot_df.dropna(subset=list(columns))
    if plot_df.empty:
        raise ValueError(f"marker_df contains no plottable rows for cluster {cluster!r}")
    return plot_df, cluster


def _logfc_norm(values: pd.Series) -> Normalize:
    limit = float(values.abs().max())
    if limit == 0:
        return Normalize(vmin=-1.0, vmax=1.0)
    if limit <= 0.25:
        return Normalize(vmin=-limit, vmax=limit)
    return SymLogNorm(linthresh=0.25, linscale=0.35, vmin=-limit, vmax=limit, base=10)


def _new_axes(ax: Optional[Axes], figsize: Tuple[float, float] = (7, 6)) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _set_cluster_title(ax: Axes, cluster) -> None:
    ax.set_title(f"Cluster: {cluster}")


def _style_axes(ax: Axes) -> None:
    ax.figure.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(False)
    ax.title.set_fontsize(_TITLE_FONT_SIZE)
    ax.xaxis.label.set_fontsize(_AXIS_FONT_SIZE)
    ax.yaxis.label.set_fontsize(_AXIS_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=_TICK_FONT_SIZE)
    text_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
    text_items.extend(ax.get_xticklabels())
    text_items.extend(ax.get_yticklabels())
    text_items.extend(ax.texts)
    if ax.legend_ is not None:
        ax.legend_.set_alignment("left")
        ax.legend_._legend_box.align = "left"
        ax.legend_.get_title().set_horizontalalignment("left")
        ax.legend_.get_title().set_fontsize(_LEGEND_FONT_SIZE)
        text_items.append(ax.legend_.get_title())
        text_items.extend(ax.legend_.get_texts())
        for legend_text in ax.legend_.get_texts():
            legend_text.set_horizontalalignment("left")
            legend_text.set_multialignment("left")
            legend_text.set_fontsize(_LEGEND_FONT_SIZE)
    for text_item in text_items:
        text_item.set_fontfamily("Arial")


def _move_legend_outside(ax: Axes) -> None:
    if ax.legend_ is None:
        return
    sns.move_legend(
        ax,
        "upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        frameon=False,
    )
    ax.figure.subplots_adjust(right=0.72)


def _annotate_genes(
    ax: Axes,
    plot_df: pd.DataFrame,
    x: str,
    y: str,
    annotate: _Annotation,
) -> None:
    if annotate is None:
        return
    if "gene" not in plot_df.columns:
        raise ValueError("marker_df must contain a gene column when annotate is used")
    if isinstance(annotate, int):
        annotation_df = plot_df.loc[plot_df["padj"] < _PADJ_THRESHOLD]
        annotation_df = annotation_df.nlargest(annotate, "|2AUC-1|")
    elif isinstance(annotate, Mapping):
        if not annotate:
            return
        ranking_key, gene_count = next(iter(annotate.items()))
        normalized_key = ranking_key.lower()
        annotation_df = plot_df.loc[plot_df["padj"] < _PADJ_THRESHOLD].copy()
        if normalized_key == "auc":
            annotation_df["_annotation_rank"] = annotation_df["auc"]
        elif normalized_key == "adjp":
            annotation_df["_annotation_rank"] = annotation_df["padj"]
        elif normalized_key == "logfc":
            annotation_df["_annotation_rank"] = annotation_df["logfoldchanges"]
        elif normalized_key == "delta_pct":
            annotation_df["_annotation_rank"] = annotation_df["pct_1"] - annotation_df["pct_2"]
        elif normalized_key == "delta_logfc":
            annotation_df["_annotation_rank"] = (
                annotation_df["logfoldchanges"] - annotation_df["lfc_sec"]
            )
        elif normalized_key == "delta_pct2":
            annotation_df["_annotation_rank"] = (
                annotation_df["pct_1"] - annotation_df["pct_sec"]
            )
        else:
            raise ValueError(f"annotation ranking must be one of {_ANNOTATION_KEYS}")
        if normalized_key == "adjp":
            annotation_df = annotation_df.nsmallest(gene_count, "_annotation_rank")
        else:
            annotation_df = annotation_df.nlargest(gene_count, "_annotation_rank")
    else:
        genes = [annotate] if isinstance(annotate, str) else annotate
        annotation_df = plot_df.loc[plot_df["gene"].isin(genes)]
    target_x = annotation_df[x].to_numpy()
    target_y = annotation_df[y].to_numpy()
    texts = [
        ax.text(
            row[x],
            row[y],
            str(row["gene"]),
            fontsize=_ANNOTATION_FONT_SIZE,
            fontfamily="Arial",
        )
        for _, row in annotation_df.iterrows()
    ]
    if not texts:
        return
    adjust_text(
        texts,
        x=plot_df[x].to_numpy(),
        y=plot_df[y].to_numpy(),
        target_x=target_x,
        target_y=target_y,
        ax=ax,
        ensure_inside_axes=True,
        prevent_crossings=True,
        expand=(1.1, 1.2),
        time_lim=1.0,
    )
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    inverse_transform = ax.transData.inverted()
    for text, target_x_value, target_y_value in zip(texts, target_x, target_y):
        text_box = text.get_window_extent(renderer=renderer)
        target_display = ax.transData.transform((target_x_value, target_y_value))
        if text_box.x0 + text_box.width / 2 >= target_display[0]:
            text_anchor = (text_box.x0, text_box.y0 + text_box.height / 2)
            elbow = (text_anchor[0] - 10, text_anchor[1])
        else:
            text_anchor = (text_box.x1, text_box.y0 + text_box.height / 2)
            elbow = (text_anchor[0] + 10, text_anchor[1])
        leader_points = inverse_transform.transform([target_display, elbow, text_anchor])
        ax.plot(
            leader_points[:, 0],
            leader_points[:, 1],
            color="#4D4D4D",
            linewidth=0.8,
            solid_capstyle="round",
            zorder=text.get_zorder() - 1,
        )


def volcano_plot(
    marker_df: pd.DataFrame,
    cluster=None,
    *,
    ax: Optional[Axes] = None,
    annotate: _Annotation = None,
) -> Axes:
    """Plot log fold change against absolute AUC discrimination for one cluster.

    Point size represents ``abs(pct_1 - pct_2)`` and color represents log fold change.
    Points with adjusted p-values greater than or equal to 0.05 are grey.
    ``marker_df`` must be returned by ``wilcoxauc``, ``find_all_markers`` or
    ``find_markers``.
    """
    columns = ("logfoldchanges", "padj", "pct_1", "pct_2", "auc")
    plot_df, cluster = _prepare_marker_df(marker_df, cluster, columns)
    plot_df["|pct_1 - pct_2|"] = (plot_df["pct_1"] - plot_df["pct_2"]).abs()
    plot_df["|2AUC-1|"] = (plot_df["auc"] * 2 - 1).abs()
    ax = _new_axes(ax, figsize=(9, 4.5))
    significant = plot_df["padj"] < _PADJ_THRESHOLD

    if (~significant).any():
        sns.scatterplot(
            data=plot_df.loc[~significant],
            x="logfoldchanges",
            y="|2AUC-1|",
            size="|pct_1 - pct_2|",
            size_norm=(0.0, 100.0),
            sizes=_POINT_SIZES,
            color="#B8B8B8",
            alpha=0.35,
            edgecolor="none",
            legend=False,
            ax=ax,
        )
    if significant.any():
        sns.scatterplot(
            data=plot_df.loc[significant],
            x="logfoldchanges",
            y="|2AUC-1|",
            hue="logfoldchanges",
            size="|pct_1 - pct_2|",
            palette=_LOGFC_CMAP,
            hue_norm=_logfc_norm(plot_df["logfoldchanges"]),
            size_norm=(0.0, 100.0),
            sizes=_POINT_SIZES,
            alpha=0.75,
            edgecolor="none",
            ax=ax,
        )
    ax.set_xlabel("log2 fold change")
    ax.set_ylabel("|2AUC-1|")
    _set_cluster_title(ax, cluster)
    _move_legend_outside(ax)
    _style_axes(ax)
    _annotate_genes(ax, plot_df, "logfoldchanges", "|2AUC-1|", annotate)
    return ax


def detection_contrast_plot(
    marker_df: pd.DataFrame,
    cluster=None,
    *,
    ax: Optional[Axes] = None,
    annotate: _Annotation = None,
) -> Axes:
    """Plot target detection rate against reference detection rate for one cluster.

    Point size represents ``abs(2 * AUC - 1)`` and color represents log fold change.
    Points with adjusted p-values greater than or equal to 0.05 are grey.
    """
    columns = ("logfoldchanges", "padj", "pct_1", "pct_2", "auc")
    plot_df, cluster = _prepare_marker_df(marker_df, cluster, columns)
    plot_df["|2AUC-1|"] = (plot_df["auc"] * 2 - 1).abs()
    ax = _new_axes(ax)
    significant = plot_df["padj"] < _PADJ_THRESHOLD

    if (~significant).any():
        sns.scatterplot(
            data=plot_df.loc[~significant],
            x="pct_2",
            y="pct_1",
            size="|2AUC-1|",
            size_norm=(0.0, 1.0),
            sizes=_POINT_SIZES,
            color="#B8B8B8",
            alpha=0.35,
            edgecolor="none",
            legend=False,
            ax=ax,
        )
    if significant.any():
        sns.scatterplot(
            data=plot_df.loc[significant],
            x="pct_2",
            y="pct_1",
            hue="logfoldchanges",
            size="|2AUC-1|",
            palette=_LOGFC_CMAP,
            hue_norm=_logfc_norm(plot_df["logfoldchanges"]),
            size_norm=(0.0, 1.0),
            sizes=_POINT_SIZES,
            alpha=0.75,
            edgecolor="none",
            ax=ax,
        )
    axis_max = max(float(plot_df["pct_1"].max()), float(plot_df["pct_2"].max()))
    axis_max = 1.0 if axis_max == 0 else axis_max * 1.05
    ax.plot([0, axis_max], [0, axis_max], linestyle="--", color="grey", linewidth=1)
    ax.set_xlim(0, axis_max)
    ax.set_ylim(0, axis_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("pct_2")
    ax.set_ylabel("pct_1")
    _set_cluster_title(ax, cluster)
    _move_legend_outside(ax)
    _style_axes(ax)
    _annotate_genes(ax, plot_df, "pct_2", "pct_1", annotate)
    return ax


def marker_exclusivity_plot(
    marker_df: pd.DataFrame,
    cluster=None,
    *,
    ax: Optional[Axes] = None,
    annotate: _Annotation = None,
) -> Axes:
    """Plot detection and fold-change contrasts for one cluster.

    Color represents ``abs(2 * AUC - 1)`` and all points use a fixed size of 10.
    Points with adjusted p-values greater than or equal to 0.05 are grey.
    """
    columns = ("logfoldchanges", "lfc_sec", "pct_1", "pct_sec", "auc", "padj")
    plot_df, cluster = _prepare_marker_df(marker_df, cluster, columns)
    plot_df["delta_pct2"] = plot_df["pct_1"] - plot_df["pct_sec"]
    plot_df["delta_logfc"] = plot_df["logfoldchanges"] - plot_df["lfc_sec"]
    plot_df["|2AUC-1|"] = (plot_df["auc"] * 2 - 1).abs()
    ax = _new_axes(ax, figsize=(9, 4.5))
    significant = plot_df["padj"] < _PADJ_THRESHOLD

    if (~significant).any():
        sns.scatterplot(
            data=plot_df.loc[~significant],
            x="delta_pct2",
            y="delta_logfc",
            s=10,
            color="#B8B8B8",
            alpha=0.35,
            edgecolor="none",
            legend=False,
            ax=ax,
        )
    if significant.any():
        sns.scatterplot(
            data=plot_df.loc[significant],
            x="delta_pct2",
            y="delta_logfc",
            hue="|2AUC-1|",
            palette=_AUC_CMAP,
            hue_norm=_AUC_NORM,
            s=10,
            alpha=0.75,
            edgecolor="none",
            ax=ax,
        )
    ax.axhline(0, linestyle="--", color="grey", linewidth=1)
    ax.axvline(0, linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("delta_pct2")
    ax.set_ylabel("delta_logfc")
    _set_cluster_title(ax, cluster)
    _move_legend_outside(ax)
    _style_axes(ax)
    _annotate_genes(ax, plot_df, "delta_pct2", "delta_logfc", annotate)
    return ax
