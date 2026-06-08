"""Talk slide: 1×1 Pareto for OLMo-60M / 192B / cosine, best vs worst LR.

Companion to ``generate_talk_peak_lr_loss.py``: plots the StarCoder
learning-forgetting Pareto for ONLY the best (lowest pretraining loss) and
worst (highest pretraining loss) peak LRs from the same magma-coloured
sweep. Each curve uses the same color and marker slot it had in the full
peak-LR plot, so the two slides line up visually. No top breakout box —
the base model is not shown here.

Output: ``figures/talk_peak_lr_pareto.png``.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from lib import data as _data, plot as _plot, style as _style

_style.apply()

LABEL_FS = 19
TICK_FS = 14
ANNOT_FS = 17
LEGEND_FS = 16
LINE_LW = 2.8       # frontier line width
MARKER_MS = 7.0     # marker size on the frontier line
SCATTER_S = 30      # scatter-point size

mpl.rcParams.update({
    "axes.labelsize":   LABEL_FS,
    "axes.titlesize":   LABEL_FS,
    "xtick.labelsize":  TICK_FS,
    "ytick.labelsize":  TICK_FS,
    "lines.linewidth":  LINE_LW,
    "lines.markersize": MARKER_MS,
})
_plot.PANEL_ANNOT_KW["fontsize"] = ANNOT_FS
_style.SCATTER_S = SCATTER_S   # pareto_panel reads this at call time


# --- config ----------------------------------------------------------------

SIZE = 60
TOKEN = 192
PEAK_LRS = (1e-4, 3e-4, 6e-4, 1e-3, 3e-3)
MARKERS = _style.RAMP_MARKERS

XLIM = (3.4, 5.25)
YLIM = (2.45, 2.95)


def _color(i):
    return _style.magma(i, len(PEAK_LRS))


def _lr_label(lr):
    return {1e-4: "1e-4", 3e-4: "3e-4", 6e-4: "6e-4",
            1e-3: "1e-3", 3e-3: "3e-3"}[lr]


# --- processing ------------------------------------------------------------

def pretrain_loss_per_lr(records):
    return {lr: _data.pretrain_loss(
        records, size=SIZE, token=TOKEN, optimizer="adamw",
        pretrain_lrs="cosine", pretrain_lr=lr,
    ) for lr in PEAK_LRS}


def pareto_points(records, lr):
    return _data.cpt_points(
        records, size=SIZE, token=TOKEN, optimizer="adamw",
        pretrain_lrs="cosine", pretrain_lr=lr,
        cpt_dataset="starcoder", cpt_tokens=10,
    )


# --- plotting --------------------------------------------------------------

def render(records):
    fig, ax = plt.subplots(figsize=(4.6, 4.9))

    losses = pretrain_loss_per_lr(records)
    valid_lrs = [lr for lr in PEAK_LRS if losses.get(lr) is not None]
    best_lr  = min(valid_lrs, key=lambda lr: losses[lr])
    worst_lr = max(valid_lrs, key=lambda lr: losses[lr])

    series = []
    for tag, lr in (("best", best_lr), ("worst", worst_lr)):
        i = PEAK_LRS.index(lr)
        series.append((
            f"{tag} LR ({_lr_label(lr)})",
            _color(i),
            MARKERS[i],
            pareto_points(records, lr),
        ))

    _plot.pareto_panel(ax, series, xlim=XLIM, ylim=YLIM)
    # pareto_panel forces a square box; release it so the figsize controls
    # the actual proportions.
    ax.set_box_aspect(None)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xlabel("Pretraining loss")
    ax.set_ylabel("Fine-tuning loss")
    _plot.loss_arrow(ax, "x")
    _plot.loss_arrow(ax, "y", gap_after_pt=6.0)
    _plot.panel_annotation(ax, "top-right", "  base model\n+ fine-tuning")
    _plot.minimal_digit_ticks(ax, axis="x")
    _plot.minimal_digit_ticks(ax, axis="y")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.58, 0.01),
        ncol=2, frameon=False, fontsize=LEGEND_FS,
        handlelength=1.6, columnspacing=1.6,
    )

    # Larger bottom margin (with a taller canvas above) leaves a gap between the
    # x-axis label and the legend while keeping the plot the same size.
    fig.subplots_adjust(left=0.18, right=0.96, top=0.97, bottom=0.265)
    return fig


def main():
    records = _data.load_small_scale()
    fig = render(records)
    _plot.finalize_loss_arrows(fig)
    out = Path(__file__).resolve().parent / "figures" / "talk_peak_lr_pareto.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
