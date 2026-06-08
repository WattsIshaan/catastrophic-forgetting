"""Web figure: the peak-LR story as a single side-by-side panel (``fig_lr.png``).

Composites the two talk panels into one image:

  * ``generate_talk_peak_lr_loss.py``   — base-model pretraining loss vs peak LR
  * ``generate_talk_peak_lr_pareto.py`` — learning–forgetting frontier for the
    best vs worst peak LR

with a "finetuning" arrow pointing from the loss panel to the frontier panel.
Each sub-panel is produced by that script's own ``render()``, so this file is a
pure compositor — there is no duplicated plotting logic and the panels stay in
sync with their standalone slide versions.

Output: ``assets/fig_lr.png``.
"""
import io as _bytesio
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from lib import data as _data, plot as _plot
import generate_talk_peak_lr_loss as _loss
import generate_talk_peak_lr_pareto as _pareto

OUT = Path(__file__).resolve().parent.parent / "assets" / "fig_lr.png"

PANEL_DPI = 300       # render each sub-panel crisply before compositing
TARGET_H = 1000       # common sub-panel height (px) after scaling
GAP_FRAC = 0.40       # arrow-gap width as a fraction of TARGET_H
ARROW_Y = 0.52        # arrow height (fraction of panel height), ~plot centre
OUT_DPI = 200


def _panel_png(module, records):
    """Render a panel script's figure to an in-memory RGBA image."""
    fig = module.render(records)
    _plot.finalize_loss_arrows(fig)
    buf = _bytesio.BytesIO()
    fig.savefig(buf, dpi=PANEL_DPI, format="png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


def _scaled(im, h):
    return im.resize((round(im.width * h / im.height), h), Image.LANCZOS)


def main():
    records = _data.load_small_scale()
    left = _scaled(_panel_png(_loss, records), TARGET_H)
    right = _scaled(_panel_png(_pareto, records), TARGET_H)

    gap = round(TARGET_H * GAP_FRAC)
    W = left.width + gap + right.width
    H = TARGET_H

    fig = plt.figure(figsize=(W / OUT_DPI, H / OUT_DPI), dpi=OUT_DPI)
    fig.patch.set_facecolor("white")

    axL = fig.add_axes([0, 0, left.width / W, 1])
    axL.imshow(np.asarray(left)); axL.axis("off")
    axR = fig.add_axes([(left.width + gap) / W, 0, right.width / W, 1])
    axR.imshow(np.asarray(right)); axR.axis("off")

    # Arrow gap between the panels: "finetuning" turns the base-model losses
    # (left) into the post-finetuning frontier (right).
    axM = fig.add_axes([left.width / W, 0, gap / W, 1])
    axM.axis("off")
    axM.set_xlim(0, 1)
    axM.set_ylim(0, 1)
    axM.annotate(
        "", xy=(0.94, ARROW_Y), xytext=(0.06, ARROW_Y),
        arrowprops=dict(arrowstyle="-|>", lw=2.6, color="#374151",
                        mutation_scale=30),
    )
    axM.text(0.5, ARROW_Y + 0.055, "fine-tuning", ha="center", va="bottom",
             fontsize=22, family="serif", color="#374151")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=OUT_DPI, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
