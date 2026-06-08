"""Very-wide variant of ``generate_talk_pareto_60m_sam`` for the web page.

Reuses the frozen talk-figure ``render()`` unchanged, then stretches the
canvas to a wide banner aspect (the motivation-section frontier on the project
page) and writes straight to ``assets/fig_frontier.png``.

Run from this directory:  python generate_talk_pareto_60m_sam_wide.py
"""
from pathlib import Path

import matplotlib as mpl

import generate_talk_pareto_60m_sam as base
from lib import data as _data, plot as _plot

# Wide aspect (~2/3 of the original very-wide banner).
WIDTH_IN = 8.0
HEIGHT_IN = 5.0

# Larger fonts than the talk default so labels read on the web page.
LABEL_FS = 20
TICK_FS = 15
ANNOT_FS = 17

OUT = Path(__file__).resolve().parent.parent / "assets" / "fig_frontier.png"


def main():
    # Bump every font knob the frozen render() reads (module globals + shared
    # rcParams + the panel-annotation kwargs) before building the figure.
    base.LABEL_FS = LABEL_FS
    base.TICK_FS = TICK_FS
    base.ANNOT_FS = ANNOT_FS
    mpl.rcParams.update({
        "axes.labelsize":  LABEL_FS,
        "axes.titlesize":  LABEL_FS,
        "xtick.labelsize": TICK_FS,
        "ytick.labelsize": TICK_FS,
    })
    _plot.PANEL_ANNOT_KW["fontsize"] = ANNOT_FS

    records = _data.load_small_scale()
    fig = base.render(records)

    # Stretch the canvas to a very wide aspect. The axes are positioned with
    # relative (figure-fraction) margins, so everything re-flows; the queued
    # loss arrows are drawn afterwards against the final layout.
    fig.set_size_inches(WIDTH_IN, HEIGHT_IN, forward=True)
    # Slightly wider margins than before so the larger tick/axis labels don't
    # clip on the left and bottom.
    fig.subplots_adjust(left=0.135, right=0.985, top=0.95, bottom=0.185,
                        hspace=0.08)

    _plot.finalize_loss_arrows(fig)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
