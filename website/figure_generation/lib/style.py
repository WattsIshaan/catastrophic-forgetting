"""Global plotting style. Edit these constants to restyle every figure.

Per-figure overrides live inside the individual `generate_*.py` scripts as
plain module-level constants. Only truly global defaults belong here.

.. warning::
    ╔══════════════════════════════════════════════════════════════════════╗
    ║  ``generate_figure_1.py`` is INTENTIONALLY NOT a consumer of this    ║
    ║  module. Figure 1 is independent and hand-tuned; do NOT touch it     ║
    ║  unless the user specifically references "Figure 1" by name, and if  ║
    ║  you're unsure, use ``AskUserQuestion`` to confirm before editing    ║
    ║  either this file or ``generate_figure_1.py``. "Make all figures X"  ║
    ║  instructions from the user do NOT include Figure 1.                 ║
    ╚══════════════════════════════════════════════════════════════════════╝
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# --- palette ---------------------------------------------------------------
# Shared palette is the Figure-1 palette: AdamW = light navy (OLMo baseline),
# SAM = pink. Every "baseline vs SAM" comparison should use these so the
# paper reads consistently.

COLORS = {
    "adamw":     "#2F5FB5",  # AdamW (medium blue)
    "sam":       "#9A1D50",  # SAM (medium burgundy, slightly brighter/saturated)
    "wsd_adamw": "#2F5FB5",
    "wsd_sam":   "#9A1D50",
    "baseline":  "black",
}

# Darker variants for text / lines that need to read on white.
TEXT_COLORS = {
    "adamw": "#163A7A",
    "sam":   "#5C0E31",
}

LABELS = {
    "adamw":     "AdamW",
    "sam":       "SAM",
    "wsd_adamw": "WSD",
    "wsd_sam":   "SAM annealing",
}

MARKERS = {
    "adamw":     "o",
    "sam":       "^",
    "wsd_adamw": "s",
    "wsd_sam":   "D",
}

# --- typography ------------------------------------------------------------

# Matplotlib defaults are axis=10, tick=10, legend=10, title=12. We stay
# close to those so the panels render readably at their compact figsizes.
FONTS = {"axis": 10, "tick": 6, "legend": 9, "title": 10, "annot": 8}

# --- sizes -----------------------------------------------------------------
# These mirror Figure 1's 6.5×4.5 footprint so that half-width figures
# render at identical effective font sizes after LaTeX scaling.

SIZES = {
    "wide":   (3.5, 2.5),    # half-page width (one or two panels)
    "xwide":  (7.0, 2.5),    # full-page width (both columns)
    "square": (3.5, 3.5),
}

# --- misc constants --------------------------------------------------------

GRID_ALPHA = 0.2
LINE_ALPHA = 0.9
HULL_LW    = 1.5
SCATTER_S  = 14

# Muted gray for reference / approximation lines (e.g., unquantized baseline,
# quadratic approximation). Keep in one place so every "reference" overlay
# reads as the same auxiliary signal.
GRAY_REF = "#888888"

# --- sequential ramp helpers ----------------------------------------------
# Used by every figure that colours a discrete-but-ordered series (peak LRs,
# anneal percents, perturbation γ). The marker tuple gives one shape per
# index up to 5 series; the dot-plot constants standardise the scatter
# overlay used on top of ramp-coloured lines.

RAMP_MARKERS = ("o", "v", "s", "p", "D")

# Scatter overlay style for "dot-line" plots (a black line through viridis
# scatter dots): identical params anywhere this idiom appears.
DOT_S          = 55
DOT_EDGECOLOR  = "black"
DOT_EDGEWIDTH  = 0.5
DOT_LINE_LW    = 1.6
DOT_LINE_COLOR = "black"


MAGMA_MIN = 0.2
MAGMA_MAX = 0.8
CIVIDIS_MIN = 0.0
CIVIDIS_MAX = 0.9


def _ramp(cmap, vmin: float, vmax: float, i: int, n: int):
    if n <= 1:
        return cmap((vmin + vmax) / 2.0)
    return cmap(vmin + (vmax - vmin) * i / (n - 1))


def magma(i: int, n: int):
    """Truncated magma: ``n>=2`` spans [``MAGMA_MIN``, ``MAGMA_MAX``]
    linearly, darkest = smallest index. Skipping the very bottom of the
    colormap avoids the near-black end and the bright yellow at the top."""
    return _ramp(plt.cm.magma, MAGMA_MIN, MAGMA_MAX, i, n)


def cividis(i: int, n: int):
    """Truncated cividis: ``n>=2`` spans [``CIVIDIS_MIN``, ``CIVIDIS_MAX``]
    linearly, darkest = smallest index. Used for annealing-time series so
    the ramp reads distinctly from the magma-coloured LR/γ series."""
    return _ramp(plt.cm.cividis, CIVIDIS_MIN, CIVIDIS_MAX, i, n)


def apply() -> None:
    """Apply shared rcParams. Call once at module load of each generate_*.py."""
    mpl.rcParams.update({
        "font.family":       "serif",
        "axes.labelsize":    FONTS["axis"],
        "axes.titlesize":    FONTS["title"],
        "xtick.labelsize":   FONTS["tick"],
        "ytick.labelsize":   FONTS["tick"],
        "legend.fontsize":   FONTS["legend"],
        "legend.frameon":    False,
        "axes.grid":         True,
        "axes.grid.which":   "major",  # gridlines only at labeled major ticks
        "grid.alpha":        GRID_ALPHA,
        "axes.axisbelow":    True,
        "savefig.bbox":      "tight",
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
        "lines.linewidth":   HULL_LW,
        "lines.markersize":  4.0,
    })


# --- display-name mappings (paper-facing strings) --------------------------

DATASET_DISPLAY = {
    "starcoder":   "StarCoder",
    "musicpile":   "MusicPile",
    "tulu":        "T\u00fclu-3",
    "gsm8k":       "GSM8K",
    "siqa":        "SIQA",
    "stackmathqa": "StackMathQA",
    "alpaca":      "Alpaca",
    "codealpaca":  "CodeAlpaca",
    "meta-math":   "Meta-Math",
    "magicoder":   "Magicoder",
}
