"""Web/talk figure: per-dataset learning–forgetting frontiers (OLMo-60M / 192B),
AdamW vs SAM, one panel per CPT dataset.

Same data + outlier trims as the paper's Figure 2 at 60M/192B, rendered with
tight auto-scaled windows for the web page (one row, five panels, shared
bottom legend).

Output: ``assets/fig_datasets.png``.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from lib import data as _data, plot as _plot, style as _style

_style.apply()

LABEL_FS = 19
TICK_FS = 14
TITLE_FS = 21
LEGEND_FS = 17
LINE_LW = 2.8       # frontier line width
MARKER_MS = 7.0     # marker size on the frontier line
SCATTER_S = 30      # scatter-point size

mpl.rcParams.update({
    "axes.labelsize":   LABEL_FS,
    "axes.titlesize":   TITLE_FS,
    "xtick.labelsize":  TICK_FS,
    "ytick.labelsize":  TICK_FS,
    "legend.fontsize":  LEGEND_FS,
    "lines.linewidth":  LINE_LW,
    "lines.markersize": MARKER_MS,
})
_style.SCATTER_S = SCATTER_S   # pareto_panel reads this at call time


# --- config ----------------------------------------------------------------

SIZE = 60
TOKEN = 192
CPT_TOKENS = 10
SAM_RHO = 5e-2
CPT_DATASETS = ("starcoder", "musicpile", "tulu", "gsm8k", "stackmathqa")
OPTIMIZERS = ("adamw", "sam")

# Per-panel outlier trims (same as the paper's Figure 2 at 60M/192B).
PANEL_PATCHES = {
    ("tulu",  "adamw"): ("clip_right_of", 3.55),
    ("tulu",  "sam"):   ("clip_right_of", 3.55),
    ("gsm8k", "sam"):   ("clip_right_of", 3.75),
}

OUT = Path(__file__).resolve().parent.parent / "assets" / "fig_datasets.png"


# --- processing ------------------------------------------------------------

def _points(records, ds, optim):
    pt_lr = _data.PT_LR[SIZE][TOKEN]
    return _data.cpt_points(
        records, size=SIZE, token=TOKEN, optimizer=optim,
        pretrain_lrs="cosine", pretrain_lr=pt_lr,
        rho=(SAM_RHO if optim == "sam" else None),
        cpt_dataset=ds, cpt_tokens=CPT_TOKENS,
    )


def _panel_data(records, ds):
    """Return (series, xs, ys) for one dataset after x-clip + outlier trims."""
    xmax = _data.pareto_window(SIZE, ds)[0][1]
    series, xs, ys = [], [], []
    for optim in OPTIMIZERS:
        pts = [p for p in _points(records, ds, optim) if p[1] <= xmax]
        pts = _plot.apply_patch(pts, ds, optim, PANEL_PATCHES)
        series.append((_style.LABELS[optim], _style.COLORS[optim],
                       _style.MARKERS[optim], pts))
        xs += [p[1] for p in pts]
        ys += [p[2] for p in pts]
    return series, xs, ys


# --- plotting --------------------------------------------------------------

def render(records):
    n = len(CPT_DATASETS)
    fig, axs = plt.subplots(1, n, figsize=(2.6 * n, 3.1))
    for ax, ds in zip(axs, CPT_DATASETS):
        series, xs, ys = _panel_data(records, ds)
        xr = (max(xs) - min(xs)) or 1.0
        yr = (max(ys) - min(ys)) or 0.1
        xlo, xhi = min(xs) - 0.05 * xr, max(xs) + 0.06 * xr
        ylo, yhi = min(ys) - 0.12 * yr, max(ys) + 0.12 * yr

        _plot.pareto_panel(ax, series, xlim=(xlo, xhi), ylim=(ylo, yhi))
        ax.set_box_aspect(None)            # release pareto_panel's 1:1 box
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_title(_style.DATASET_DISPLAY[ds])
        ax.set_xlabel("Pretraining loss")
        _plot.loss_arrow(ax, "x")
        _plot.minimal_digit_ticks(ax, axis="x")
        _plot.minimal_digit_ticks(ax, axis="y")

    axs[0].set_ylabel("Fine-tuning loss")
    _plot.loss_arrow(axs[0], "y", gap_after_pt=6.0)

    fig.subplots_adjust(left=0.055, right=0.99, top=0.9, bottom=0.30, wspace=0.42)

    handles = [Line2D([0], [0], color=_style.COLORS[o], marker=_style.MARKERS[o],
                      lw=LINE_LW, label=_style.LABELS[o])
               for o in OPTIMIZERS]
    _plot.place_legend_below(fig, [handles], [[_style.LABELS[o] for o in OPTIMIZERS]])
    return fig


def main():
    records = _data.load_small_scale()
    fig = render(records)
    _plot.finalize_loss_arrows(fig)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
