"""Talk slide: 1×1 Pareto for OLMo-60M / 192B PT tokens, SAM only.

Two-box variant: the SAM frontier (StarCoder, 10M CPT tokens) lives in the
main bottom panel, and the base-model point (well above the frontier) gets
its own small fully-bordered box at the top so the main panel isn't squished.
The two panels share x-scale and grid and are separated by whitespace.

Output: ``figures/talk_pareto_60m_sam.png``. No legend.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

from lib import data as _data, plot as _plot, style as _style

_style.apply()

# Talk-slide font sizes — bigger than the paper defaults so labels read at
# projector distance.
LABEL_FS = 14
TICK_FS = 10
ANNOT_FS = 12

mpl.rcParams.update({
    "axes.labelsize":  LABEL_FS,
    "axes.titlesize":  LABEL_FS,
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS,
})
# panel_annotation captures FONTS["annot"] at lib import time, so override
# the kwargs dict directly.
_plot.PANEL_ANNOT_KW["fontsize"] = ANNOT_FS


# --- config ----------------------------------------------------------------

SIZE = 60
TOKEN = 192
CPT_DATASET = "starcoder"
CPT_TOKENS = 10
SAM_RHO = 5e-2

# User-provided base-model (DCLM PT loss, StarCoder loss before fine-tuning).
BASE_MODEL_POINT = (3.4133, 3.7121)

# Shared x window. The SAM curve plus the base-model x both fit inside this.
XLIM = (3.4, 5.25)
# Bottom panel: tight around the SAM curve.
YLIM_MAIN = (2.5, 2.88)
# Top breakout: just enough headroom around the base-model point.
YLIM_BREAK = (3.66, 3.76)
# Top panel is ~1/6 the height of the bottom panel so the main Pareto stays
# roughly square.
HEIGHT_RATIOS = (1, 6)


# --- processing ------------------------------------------------------------

def sam_points(records):
    pt_lr = _data.PT_LR[SIZE][TOKEN]
    return _data.cpt_points(
        records, size=SIZE, token=TOKEN, optimizer="sam",
        pretrain_lrs="cosine", pretrain_lr=pt_lr, rho=SAM_RHO,
        cpt_dataset=CPT_DATASET, cpt_tokens=CPT_TOKENS,
    )


# --- plotting --------------------------------------------------------------

def render(records):
    fig = plt.figure(figsize=(4.0, 4.6))
    gs = fig.add_gridspec(2, 1, height_ratios=HEIGHT_RATIOS, hspace=0.08)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)

    # --- main panel: SAM frontier ----------------------------------------
    pts = sam_points(records)
    series = [(_style.LABELS["sam"], _style.COLORS["sam"],
               _style.MARKERS["sam"], pts)]
    _plot.pareto_panel(ax_bot, series, xlim=XLIM, ylim=YLIM_MAIN)
    # pareto_panel forces a 1:1 box; release it so gridspec can size both
    # panels naturally.
    ax_bot.set_box_aspect(None)
    ax_bot.set_xlim(*XLIM)
    ax_bot.set_ylim(*YLIM_MAIN)
    ax_bot.set_xlabel("Pretraining loss")
    ax_bot.set_ylabel("Fine-tuning loss")
    _plot.loss_arrow(ax_bot, "x")
    _plot.loss_arrow(ax_bot, "y", gap_after_pt=6.0)

    # Best fine-tuned point (lowest y in window).
    visible = [(lr, x, y) for lr, x, y in pts
               if XLIM[0] <= x <= XLIM[1] and YLIM_MAIN[0] <= y <= YLIM_MAIN[1]]
    _, best_x, best_y = min(visible, key=lambda p: p[2])
    ax_bot.scatter([best_x], [best_y], color=_style.COLORS["sam"],
                   marker=_style.MARKERS["sam"],
                   s=_style.SCATTER_S * 2.0, zorder=5,
                   edgecolors="white", linewidths=0.6)
    ax_bot.annotate(
        "best fine-tuned\nmodel", xy=(best_x, best_y),
        xytext=(12, 22), textcoords="offset points",
        ha="left", va="bottom",
        fontsize=ANNOT_FS,
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
        arrowprops=dict(arrowstyle="->", lw=0.7, color="black",
                        shrinkA=2, shrinkB=3),
    )
    _plot.panel_annotation(ax_bot, "top-right", "  base model\n+ fine-tuning")

    # --- breakout panel: base model only ---------------------------------
    ax_top.set_xlim(*XLIM)
    ax_top.set_ylim(*YLIM_BREAK)
    bx, by = BASE_MODEL_POINT
    ax_top.scatter([bx], [by], color=_style.COLORS["baseline"], marker="o",
                   s=_style.SCATTER_S * 1.6, zorder=5,
                   edgecolors="white", linewidths=0.6)
    ax_top.annotate(
        "base model", xy=(bx, by),
        xytext=(16, 0), textcoords="offset points",
        ha="left", va="center",
        fontsize=ANNOT_FS,
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
        arrowprops=dict(arrowstyle="->", lw=0.7, color="black",
                        shrinkA=2, shrinkB=3),
    )
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_top.set_yticks([round(by, 1)])
    ax_top.tick_params(axis="y", which="both", labelsize=TICK_FS)
    # Same grid style as the main panel.
    ax_top.grid(True, alpha=_style.GRID_ALPHA)
    ax_top.set_axisbelow(True)

    # Lock down ticks now that xlim/ylim are final.
    _plot.minimal_digit_ticks(ax_bot, axis="x")
    _plot.minimal_digit_ticks(ax_bot, axis="y")

    fig.subplots_adjust(left=0.20, right=0.96, top=0.97, bottom=0.15,
                        hspace=0.08)
    return fig


def main():
    records = _data.load_small_scale()
    fig = render(records)
    _plot.finalize_loss_arrows(fig)
    out = Path(__file__).resolve().parent / "figures" / "talk_pareto_60m_sam.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
