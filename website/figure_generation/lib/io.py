"""Tiny save helper: every figure lands in `figures/` as a PDF."""
from pathlib import Path

import matplotlib.pyplot as plt

from . import plot as _plot

HERE = Path(__file__).resolve().parent.parent
FIGURES = HERE / "figures"


def save(fig, name: str, *, also_png: bool = False) -> Path:
    """Save `fig` to `figures/{name}.pdf`. `name` must not contain an extension."""
    assert "." not in Path(name).name or name.endswith((".pdf",)), (
        f"pass `name` without extension; got {name!r}"
    )
    if name.endswith(".pdf"):
        name = name[:-4]
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"{name}.pdf"
    _plot.finalize_loss_arrows(fig)
    fig.savefig(out)
    if also_png:
        fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"wrote {out}")
    return out
