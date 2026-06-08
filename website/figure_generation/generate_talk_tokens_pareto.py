"""Web/talk figure: SAM vs AdamW learning–forgetting frontiers on StarCoder
(OLMo-60M, 10M CPT tokens), one panel per *pretraining*-token budget.

Sweeping the pretraining-token budget (12B → 192B) with fine-tuning fixed to
StarCoder at 10M CPT tokens shows the frontier improving as the base model sees
more pretraining tokens, with SAM staying ahead of AdamW at every budget. A
shared window across panels keeps the shift directly comparable and trims the
divergent high-LR AdamW tails.

Output: ``assets/fig_tokens.png``.
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
CPT_DATASET = "starcoder"
CPT_TOKENS = 10
SAM_RHO = 5e-2
PT_TOKENS = (12, 24, 48, 96, 192)          # billions of pretraining tokens
OPTIMIZERS = ("adamw", "sam")

# Shared window across all panels, matched to the StarCoder panel of the
# datasets figure (its auto-fit window at 60M / 192B / 10M CPT) so the two
# figures share an identical frame.
XLIM = (3.4349, 4.5121)
YLIM = (2.5107, 2.8642)

OUT = Path(__file__).resolve().parent.parent / "assets" / "fig_tokens.png"


# --- processing ------------------------------------------------------------

def _points(records, token, optim):
    pt_lr = _data.PT_LR[SIZE][token]
    return _data.cpt_points(
        records, size=SIZE, token=token, optimizer=optim,
        pretrain_lrs="cosine", pretrain_lr=pt_lr,
        rho=(SAM_RHO if optim == "sam" else None),
        cpt_dataset=CPT_DATASET, cpt_tokens=CPT_TOKENS,
    )


# --- plotting --------------------------------------------------------------

def render(records):
    n = len(PT_TOKENS)
    # Same per-panel footprint as fig_datasets (2.6"/panel, 1.86" tall, the same
    # left/right/wspace), with extra canvas height for the "Pretraining tokens"
    # header above the per-panel titles.
    fig, axs = plt.subplots(1, n, figsize=(2.6 * n, 3.8))
    for ax, token in zip(axs, PT_TOKENS):
        series = [(_style.LABELS[o], _style.COLORS[o], _style.MARKERS[o],
                   _points(records, token, o)) for o in OPTIMIZERS]
        _plot.pareto_panel(ax, series, xlim=XLIM, ylim=YLIM)
        ax.set_box_aspect(None)            # release pareto_panel's 1:1 box
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_title(f"{token}B")
        ax.set_xlabel("Pretraining loss")
        _plot.loss_arrow(ax, "x")
        _plot.minimal_digit_ticks(ax, axis="x")
        _plot.minimal_digit_ticks(ax, axis="y")

    axs[0].set_ylabel("Fine-tuning loss")
    _plot.loss_arrow(axs[0], "y", gap_after_pt=6.0)

    fig.suptitle("Pretraining tokens", y=0.94, fontsize=TITLE_FS)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.74, bottom=0.244, wspace=0.42)

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
