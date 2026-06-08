"""Reusable plotting primitives that live above matplotlib but below figures.

`pareto_panel` is the single pattern that underlies every Pareto plot in the
paper: scatter the points, draw the monotone-hull frontier over them, and
clip to a (size, dataset)-dependent window.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.patheffects as pe
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MultipleLocator
from matplotlib.transforms import blended_transform_factory, offset_copy

from . import hull, style


# --- Per-panel point patches ----------------------------------------------
# Targeted per-panel edits are expressed as "clip to one side of a threshold"
# on the x or y axis of the point tuples (tuples are (lr, x, y)). Each figure
# script declares a ``PANEL_PATCHES`` dict of the form
#
#     {(panel_key, series_key): ("clip_right_of", 3.54)}
#
# and calls ``apply_patch(pts, panel_key, series_key, patches)`` before
# plotting. Adding a new kind of transform is a one-line entry in
# ``PATCH_FNS`` below.

PATCH_FNS = {
    # Keep points at or below the threshold on the given axis.
    "clip_right_of":  lambda pts, thr: [p for p in pts if p[1] <= thr],
    "clip_left_of":   lambda pts, thr: [p for p in pts if p[1] >= thr],
    "clip_above":     lambda pts, thr: [p for p in pts if p[2] <= thr],
    "clip_below":     lambda pts, thr: [p for p in pts if p[2] >= thr],
}


def apply_patch(pts, panel_key, series_key, patches):
    """Return ``pts`` transformed by the patch specced in ``patches`` for
    this (panel, series), or unchanged if no match.

    A patch spec is a tuple ``(name, *args)``; ``name`` must be a key of
    ``PATCH_FNS``.
    """
    if not patches:
        return pts
    spec = patches.get((panel_key, series_key))
    if spec is None:
        return pts
    name, *args = spec
    return PATCH_FNS[name](pts, *args)


def minimal_digit_ticks(
    ax, *, axis: str = "y", min_ticks: int = 2, max_ticks: int = 4,
) -> None:
    """Set major-tick step to the fewest-decimal value whose multiples fit
    in the current axis range between ``min_ticks`` and ``max_ticks`` times.

    Within a digit count we prefer the smallest step (i.e. the finest
    division) so all in-between values get their own major tick — that way
    the gridlines align with the displayed labels. Minor ticks stay at
    matplotlib's default subdivision for the finer grid.

    Log-scale axes are skipped (matplotlib's LogLocator already picks good
    decade-based ticks there).
    """
    scale = ax.get_yscale() if axis == "y" else ax.get_xscale()
    if scale != "linear":
        return
    lo, hi = (ax.get_ylim() if axis == "y" else ax.get_xlim())
    if hi < lo:
        lo, hi = hi, lo

    def _apply(step, decs):
        axis_obj = ax.yaxis if axis == "y" else ax.xaxis
        axis_obj.set_major_locator(MultipleLocator(step))
        axis_obj.set_minor_locator(AutoMinorLocator(2))
        axis_obj.set_major_formatter(
            FuncFormatter(lambda v, _p, d=decs: f"{v:.{d}f}"))

    def _count(step):
        low = math.ceil(lo / step - 1e-9)
        high = math.floor(hi / step + 1e-9)
        return high - low + 1

    # Pass 1: fewest-digits with ticks in [min_ticks, max_ticks], preferring
    # smallest step (most intermediate values). We include 3 and 4 as
    # candidate multipliers so we can hit the cap for ranges where only
    # (1, 2, 5) would either over- or under-shoot it.
    muls_small_first = (1, 2, 3, 4, 5)
    for decs in range(0, 8):
        for mul in muls_small_first:
            step = mul * 10 ** (-decs)
            n = _count(step)
            if min_ticks <= n <= max_ticks:
                _apply(step, decs)
                return
    # Pass 2: fallback — accept anything that hits min_ticks (tiny panel
    # ranges where even step=10^-8 would overflow the label budget).
    for decs in range(0, 8):
        for mul in (5, 4, 3, 2, 1):
            step = mul * 10 ** (-decs)
            if _count(step) >= min_ticks:
                _apply(step, decs)
                return


TARGET_LEGEND_GAP_IN = 0.05  # inches of whitespace between axes bottom and legend top
TARGET_LEGEND_ROW_SPACING_IN = 0.14  # inches between legend rows


LOSS_ARROW_LW = 0.55
LOSS_ARROW_HEAD = 6.0
LOSS_ARROW_SPAN = 0.95
LOSS_ARROW_GAP_BEFORE_PT = 3.5  # tick labels -> arrow centerline
LOSS_ARROW_GAP_AFTER_PT = 3.5   # arrow centerline -> axis label


PANEL_ANNOT_KW = dict(
    fontsize=style.FONTS["annot"],
    color="black",
    zorder=6,
    path_effects=[pe.withStroke(linewidth=2, foreground="white")],
)


def panel_annotation(ax, position: str, text: str, *, pad: float = 0.03):
    """Place a small stroked text label inside one of the four panel corners.

    ``position`` is one of ``"top-left"``, ``"top-right"``, ``"bottom-left"``,
    ``"bottom-right"``.
    """
    corners = {
        "top-left":     (pad,        1.0 - pad, "left",  "top"),
        "top-right":    (1.0 - pad,  1.0 - pad, "right", "top"),
        "bottom-left":  (pad,        pad,       "left",  "bottom"),
        "bottom-right": (1.0 - pad,  pad,       "right", "bottom"),
        "center-left":  (pad,        0.5,       "left",  "center"),
        "center-right": (1.0 - pad,  0.5,       "right", "center"),
    }
    if position not in corners:
        raise ValueError(f"position must be one of {sorted(corners)}, got {position!r}")
    x, y, ha, va = corners[position]
    ax.text(x, y, text, transform=ax.transAxes,
            ha=ha, va=va, ma="left", **PANEL_ANNOT_KW)


def _ticklabel_extent_pt(ax, axis: str) -> float:
    """Measure how far visible tick labels extend past the axis edge, in points.

    For ``axis="x"`` this is the vertical extent below the bottom spine; for
    ``axis="y"`` it is the horizontal extent left of the left spine.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bbox = ax.get_window_extent(renderer)
    ticklabels = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    extent_px = 0.0
    for tl in ticklabels:
        if not tl.get_visible() or not tl.get_text():
            continue
        bb = tl.get_window_extent(renderer)
        if axis == "x":
            extent_px = max(extent_px, ax_bbox.y0 - bb.y0)
        else:
            extent_px = max(extent_px, ax_bbox.x0 - bb.x0)
    return max(0.0, extent_px) * 72.0 / fig.dpi


def loss_arrow(ax, axis: str, *, reverse: bool = False,
               span: float = LOSS_ARROW_SPAN,
               gap_before_pt: float = LOSS_ARROW_GAP_BEFORE_PT,
               gap_after_pt: float = LOSS_ARROW_GAP_AFTER_PT,
               lw: float = LOSS_ARROW_LW,
               head: float = LOSS_ARROW_HEAD):
    """Queue a thin arrow next to the tick labels indicating decreasing loss.

    The arrow's centerline sits ``gap_before_pt`` points outside the visible
    tick labels, and the axis label is positioned with ``gap_after_pt`` points
    of whitespace past the arrow. Set ``reverse=True`` if the data axis is
    inverted so the arrow's head stays on the lower-loss side.

    Arrows are queued on the figure and drawn by ``finalize_loss_arrows`` (which
    ``io.save`` calls automatically) so the tick-label measurement always uses
    the final tick labels, regardless of the order in which the figure script
    formats its axes.
    """
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    fig = ax.figure
    queue = getattr(fig, "_loss_arrows", None)
    if queue is None:
        queue = []
        fig._loss_arrows = queue
    queue.append(dict(ax=ax, axis=axis, reverse=reverse, span=span,
                      gap_before_pt=gap_before_pt, gap_after_pt=gap_after_pt,
                      lw=lw, head=head))
    labelpad_pt = gap_before_pt + gap_after_pt
    if axis == "x":
        ax.xaxis.labelpad = labelpad_pt
    else:
        ax.yaxis.labelpad = labelpad_pt


def finalize_loss_arrows(fig):
    """Measure tick extents and draw all arrows queued by ``loss_arrow``.

    Called automatically by ``io.save``; safe to call manually if you need to
    render the figure before saving.
    """
    queue = getattr(fig, "_loss_arrows", None)
    if not queue:
        return
    fig.canvas.draw()
    for spec in queue:
        ax = spec["ax"]
        axis = spec["axis"]
        tick_ext_pt = _ticklabel_extent_pt(ax, axis)
        offset_pt = tick_ext_pt + spec["gap_before_pt"]
        base = blended_transform_factory(ax.transAxes, ax.transAxes)
        half = spec["span"] / 2.0
        if axis == "x":
            trans = offset_copy(base, fig=fig, y=-offset_pt, units="points")
            if spec["reverse"]:
                tail, head_pos = (0.5 - half, 0.0), (0.5 + half, 0.0)
            else:
                tail, head_pos = (0.5 + half, 0.0), (0.5 - half, 0.0)
        else:
            trans = offset_copy(base, fig=fig, x=-offset_pt, units="points")
            if spec["reverse"]:
                tail, head_pos = (0.0, 0.5 - half), (0.0, 0.5 + half)
            else:
                tail, head_pos = (0.0, 0.5 + half), (0.0, 0.5 - half)
        ax.annotate(
            "", xy=head_pos, xycoords=trans,
            xytext=tail, textcoords=trans,
            arrowprops=dict(arrowstyle="->", color="black", lw=spec["lw"],
                            mutation_scale=spec["head"], shrinkA=0, shrinkB=0),
            annotation_clip=False, zorder=10,
        )
    fig._loss_arrows = []


SUBPLOT_LABEL_X_PT = -21.0  # left of axes' top-left corner, in points
SUBPLOT_LABEL_Y_PT = 4.0    # above axes' top edge, in points


def subplot_label(ax, letter: str, *,
                  x_pt: float = SUBPLOT_LABEL_X_PT,
                  y_pt: float = SUBPLOT_LABEL_Y_PT):
    """Draw a bold ``(letter)`` above the top-left corner of ``ax``.

    Offsets are in points so the placement is identical across panels of any
    width.
    """
    trans = offset_copy(ax.transAxes, fig=ax.figure,
                        x=x_pt, y=y_pt, units="points")
    ax.text(0.0, 1.0, f"({letter})", transform=trans,
            ha="left", va="bottom",
            fontweight="bold", fontsize=style.FONTS["title"])


def _axes_bottom_fraction(fig):
    """Return the lowest y (in figure-fraction) across all axes' tight bboxes.

    The tight bbox includes tick labels, xlabel, and title, so this gives the
    true bottom of the plotted content after layout.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    min_y_px = min(ax.get_tightbbox(renderer).y0 for ax in fig.axes)
    fig_h_px = fig.get_figheight() * fig.dpi
    return min_y_px / fig_h_px


def place_legend_below(fig, row_handles, row_labels, *,
                       gap_in: float = TARGET_LEGEND_GAP_IN,
                       row_spacing_in: float = TARGET_LEGEND_ROW_SPACING_IN,
                       legend_kwargs_per_row=None):
    """Place one or more legend rows below all axes with a uniform whitespace gap.

    ``row_handles`` and ``row_labels`` are sequences of per-row handles/labels,
    e.g. ``[[h1, h2], [h3]]`` makes a 2-row legend. The top row sits
    ``gap_in`` inches below the axes' tight-bbox bottom; subsequent rows are
    ``row_spacing_in`` inches below the previous row. Must be called AFTER
    any ``fig.subplots_adjust`` that sets the axes' final vertical extent.

    Returns the list of Legend artists (one per row), in order top→bottom.
    """
    assert len(row_handles) == len(row_labels)
    axes_bottom = _axes_bottom_fraction(fig)
    fig_h_in = fig.get_figheight()
    gap_frac = gap_in / fig_h_in
    row_frac = row_spacing_in / fig_h_in
    legends = []
    per_row_kw = legend_kwargs_per_row or [{}] * len(row_handles)
    for i, (handles, labels) in enumerate(zip(row_handles, row_labels)):
        y = axes_bottom - gap_frac - i * row_frac
        kw = dict(per_row_kw[i])
        kw.setdefault("ncol", len(handles))
        kw.setdefault("frameon", False)
        leg = fig.legend(handles, labels, loc="upper center",
                         bbox_to_anchor=(0.5, y), **kw)
        if i < len(row_handles) - 1:
            fig.add_artist(leg)
        legends.append(leg)
    return legends


def _clip(points: Sequence[Tuple[float, float, float]],
          xlim: Optional[Tuple[float, float]] = None,
          ylim: Optional[Tuple[float, float]] = None):
    out = []
    for lr, x, y in points:
        if xlim is not None and not (xlim[0] <= x <= xlim[1]):
            continue
        if ylim is not None and not (ylim[0] <= y <= ylim[1]):
            continue
        out.append((lr, x, y))
    return out


def pareto_panel(
    ax, series: Iterable[tuple],
    *,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    alpha: float = style.LINE_ALPHA,
) -> None:
    """Draw one Pareto panel.

    `series` is an iterable of tuples ``(label, color, marker, points)`` where
    each element of ``points`` is ``(cpt_lr, x, y)``. Points outside
    (xlim, ylim) are filtered out.
    """
    for label, color, marker, points in series:
        pts = _clip(points, xlim=xlim, ylim=ylim)
        if not pts:
            continue
        xs = [x for _lr, x, _y in pts]
        ys = [y for _lr, _x, y in pts]
        ax.scatter(xs, ys, color=color, marker=marker, alpha=alpha,
                   s=style.SCATTER_S)
        if len(pts) >= 3:
            hx, hy = hull.monotone_hull([(x, y) for _lr, x, y in pts])
            ax.plot(hx, hy, color=color, marker=marker, alpha=alpha,
                    label=label)
        else:
            # Tiny series: draw a plain line so the legend still gets an entry.
            ax.plot(xs, ys, color=color, marker=marker, alpha=alpha,
                    label=label)
    # Pretrain-loss-vs-finetune-loss panels should render as a square box so
    # the Pareto frontier is read symmetrically on both axes.
    ax.set_box_aspect(1)
    minimal_digit_ticks(ax, axis="y")
    minimal_digit_ticks(ax, axis="x")
