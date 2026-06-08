"""Talk slide: 1×1 pretraining loss vs peak LR for OLMo-60M / 192B / cosine.

Same data as panel (a) of ``generate_figure_peak_lr.py``: the magma-coloured
dot-line plot of base-model pretraining loss across a peak-LR sweep.
Annotates the best (lowest loss) and worst (highest loss) LR points with
arrows so the viewer can read off the range at a glance.

Output: ``figures/talk_peak_lr_loss.png``.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from lib import data as _data, plot as _plot, style as _style

_style.apply()

LABEL_FS = 19
TICK_FS = 14
ANNOT_FS = 17

mpl.rcParams.update({
    "axes.labelsize":  LABEL_FS,
    "axes.titlesize":  LABEL_FS,
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS,
})
_plot.PANEL_ANNOT_KW["fontsize"] = ANNOT_FS


# --- config ----------------------------------------------------------------

SIZE = 60
TOKEN = 192
PEAK_LRS = (1e-4, 3e-4, 6e-4, 1e-3, 3e-3)


def _color(i):
    return _style.magma(i, len(PEAK_LRS))


def _lr_label(lr):
    return {1e-4: "1e-4", 3e-4: "3e-4", 6e-4: "6e-4",
            1e-3: "1e-3", 3e-3: "3e-3"}[lr]


# --- processing ------------------------------------------------------------

def pretrain_series(records):
    return [
        (lr, _data.pretrain_loss(
            records, size=SIZE, token=TOKEN, optimizer="adamw",
            pretrain_lrs="cosine", pretrain_lr=lr,
        ))
        for lr in PEAK_LRS
    ]


# --- plotting --------------------------------------------------------------

def render(records):
    fig, ax = plt.subplots(figsize=(4.6, 4.0))

    pairs = pretrain_series(records)
    valid = [(lr, v) for lr, v in pairs if v is not None]

    xs = [lr for lr, _ in valid]
    ys = [v for _, v in valid]
    ax.plot(xs, ys, color=_style.DOT_LINE_COLOR,
            lw=_style.DOT_LINE_LW, zorder=1)
    for i, (lr, v) in enumerate(pairs):
        if v is None:
            continue
        ax.scatter(lr, v, s=_style.DOT_S, color=_color(i),
                   edgecolors=_style.DOT_EDGECOLOR,
                   linewidths=_style.DOT_EDGEWIDTH, zorder=2)

    ax.set_xscale("log")
    ax.set_xticks(PEAK_LRS)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _p: _lr_label(x) if x in PEAK_LRS else ""))
    ax.tick_params(axis="x", labelrotation=45)
    ax.minorticks_off()
    ax.set_xlabel("Pretraining LR")
    ax.set_ylabel("Pretraining loss")
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    _plot.loss_arrow(ax, "y", gap_after_pt=6.0)
    # Right-pointing arrow centred on the x-axis, indicating direction of
    # increasing LR.
    _plot.loss_arrow(ax, "x", reverse=True)
    _plot.panel_annotation(ax, "top-left", "base model")
    _plot.minimal_digit_ticks(ax, axis="y")

    # Best (lowest loss) and worst (highest loss) callouts.
    best_lr, best_loss = min(valid, key=lambda p: p[1])
    worst_lr, worst_loss = max(valid, key=lambda p: p[1])

    ax.annotate(
        "best lr", xy=(best_lr, best_loss),
        xytext=(18, 28), textcoords="offset points",
        ha="left", va="bottom",
        fontsize=ANNOT_FS,
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
        arrowprops=dict(arrowstyle="->", lw=0.7, color="black",
                        shrinkA=3, shrinkB=4),
    )
    ax.annotate(
        "worst lr", xy=(worst_lr, worst_loss),
        xytext=(-22, -22), textcoords="offset points",
        ha="right", va="top",
        fontsize=ANNOT_FS,
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
        arrowprops=dict(arrowstyle="->", lw=0.7, color="black",
                        shrinkA=3, shrinkB=4),
    )

    fig.subplots_adjust(left=0.20, right=0.96, top=0.95, bottom=0.20)
    return fig


def main():
    records = _data.load_small_scale()
    fig = render(records)
    _plot.finalize_loss_arrows(fig)
    out = Path(__file__).resolve().parent / "figures" / "talk_peak_lr_loss.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
