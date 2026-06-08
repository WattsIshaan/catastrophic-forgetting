"""Talk-slide variant of Figure 1.

Same bespoke layout as ``generate_figure_1.py``, but only includes the
Meta-Math, T\u00fclu-3, and Quantization bars so the slide stays uncluttered.

This file deliberately duplicates ``generate_figure_1.py``'s rendering code
rather than importing from it, since that file is documented as a frozen,
hand-tuned artifact whose imports must stay independent of ``lib/`` and whose
implementation should not be refactored. Treat this file the same way: don't
restyle it via ``lib/style.py`` or any global "make all figures X" change.

Output: ``figures/talk_figure_1.png``.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory, offset_copy
from scipy.spatial import ConvexHull


# --- config ----------------------------------------------------------------

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "midtrain_olmo_results.json"
OUT_PATH = HERE / "figures" / "talk_figure_1.png"

PRETRAIN_T, MIDTRAIN_B, SAM_RHO = 4, 50, 5e-2

OLMES_TASKS = (
    "arc_challenge", "hellaswag", "mmlu", "winogrande",
    "drop", "naturalqs_open",
    "agi_eval_english", "gsm8k", "mmlu_pro", "triviaqa",
)

COLOR = {"adamw": "#ADBBD5", "sam": "#F1499E"}
TEXT_COLOR = {"adamw": "#253B5C", "sam": "#79083F"}
LABEL = {"adamw": "OLMo Baseline", "sam": "Sharpness-Aware Minimization"}


@dataclass(frozen=True)
class Group:
    name: str
    dataset: str
    tokens: int
    lr_range: tuple
    loss_margin: float


# Talk slide: Meta-Math + T\u00fclu-3 only (StackMathQA and MusicPile dropped
# from the paper's set). Quantization is appended separately below.
CPT_GROUPS = (
    Group("Meta-Math",   "meta-math",   80, (4e-5, 1e-4), 0.0005),
    Group("T\u00fclu-3", "tulu",        50, (2e-6, 3e-5), 0.005),
)


# --- processing ------------------------------------------------------------

@dataclass(frozen=True)
class Bar:
    label: str
    adamw: float
    sam: float
    adamw_drop_pct: float
    sam_drop_pct: float
    improvement_pct: float


@dataclass(frozen=True)
class FigureData:
    bars: List[Bar]
    baseline_adamw: float
    baseline_sam: float


def _olmes_mean(run):
    oe = run.get("olmes_eval") or {}
    vals = [oe.get(t) for t in OLMES_TASKS]
    return None if any(v is None for v in vals) else float(np.mean(vals))


def _matches_run(r, optimizer):
    return (r.get("pretrain_token") == PRETRAIN_T
            and r.get("midtrain_tokens") == MIDTRAIN_B
            and r.get("optimizer") == optimizer
            and (optimizer != "sam" or r.get("rho") == SAM_RHO))


def _downstream_score(records, optimizer, eval_types=("hf", "olmo")):
    for et in eval_types:
        for r in records:
            if (r.get("run_type") == "downstream"
                    and _matches_run(r, optimizer)
                    and r.get("eval_type") == et):
                score = _olmes_mean(r)
                if score is not None:
                    return score
    return None


def _cpt_points(records, group, optimizer):
    lo, hi = group.lr_range
    pts = []
    for r in records:
        if not (r.get("run_type") == "cpt"
                and _matches_run(r, optimizer)
                and r.get("cpt_dataset") == group.dataset
                and r.get("cpt_tokens") == group.tokens
                and r.get("step", -1) == -1):
            continue
        lr, y = r.get("cpt_lr"), r.get("finetuning_val_loss")
        x = _olmes_mean(r)
        if None not in (lr, x, y) and lo <= lr <= hi:
            pts.append((x, y, lr))
    return sorted(pts, key=lambda p: p[2])


def _open_hull(points):
    if len(points) < 3:
        return [p[0] for p in points], [p[1] for p in points]
    xy = np.array([(x, y) for x, y, _ in points])
    lrs = np.array([lr for _, _, lr in points])
    verts = ConvexHull(xy).vertices
    i0, i1 = int(np.argmin(lrs[verts])), int(np.argmax(lrs[verts]))
    n = len(verts)
    step = 1 if (i0 + 1) % n != i1 else -1
    order = [verts[(i0 + k * step) % n] for k in range(n)]
    path = xy[order]
    return path[:, 0].tolist(), path[:, 1].tolist()


def _highest_x_at(xs, ys, y_line):
    if len(xs) < 2:
        return None
    y_line = min(max(y_line, min(ys)), max(ys))
    best = None
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        lo, hi = sorted((y0, y1))
        if not (lo - 1e-12 <= y_line <= hi + 1e-12):
            continue
        t = 0.0 if abs(y1 - y0) < 1e-15 else (y_line - y0) / (y1 - y0)
        x = x0 + t * (x1 - x0)
        best = x if best is None else max(best, x)
    return best


def _cpt_bar(records, group, optimizer, adamw_min_loss):
    xs, ys = _open_hull(_cpt_points(records, group, optimizer))
    return _highest_x_at(xs, ys, adamw_min_loss * (1.0 + group.loss_margin))


def _make_bar(label, adamw, sam, baseline):
    da = 100.0 * (baseline - adamw) / baseline
    ds = 100.0 * (baseline - sam) / baseline
    imp = 100.0 * (da - ds) / da if da else 0.0
    return Bar(label, adamw, sam, da, ds, imp)


def build_figure_data(records) -> FigureData:
    base_adamw = _downstream_score(records, "adamw")
    base_sam = _downstream_score(records, "sam")

    bars = []
    for g in CPT_GROUPS:
        min_loss = min(y for _, y, _ in _cpt_points(records, g, "adamw"))
        bars.append(_make_bar(
            g.name,
            _cpt_bar(records, g, "adamw", min_loss),
            _cpt_bar(records, g, "sam", min_loss),
            base_adamw,
        ))
    bars.append(_make_bar(
        "Quantization (4bit)",
        _downstream_score(records, "adamw", ("hf-4bit",)),
        _downstream_score(records, "sam", ("hf-4bit",)),
        base_adamw,
    ))

    return FigureData(bars=bars, baseline_adamw=base_adamw, baseline_sam=base_sam)


# --- plotting --------------------------------------------------------------

QUANT_LABEL = "Quantization (4bit)"
QUANT_DISPLAY = "PTQ\n(4-bit)"
BAR_WIDTH = 0.42
ARROW_COLOR = "#000000"
ARROW_LW = 1.75
ARROW_BOTTOM_OFFSET = 0.035
ARROW_TIP_OFFSET = 0.10
ARROW_LEFT_OFFSET = 0.06

FONTSIZE = {"axis": 21, "tick": 16, "legend": 19, "improvement": 16, "brace": 7}

TICK_LABEL_DISPLAY = {
    "Meta-Math":   "Meta\nMath",
    "StackMathQA": "StackMath\nQA",
    QUANT_LABEL:   QUANT_DISPLAY,
}


def _draw_forgetting_brace(ax, y_bottom, y_top, x_vertical, label,
                           tick=0.08, lw=0.9, label_gap=0.30):
    ax.plot([x_vertical, x_vertical + tick], [y_top, y_top],
            color="black", lw=lw, clip_on=False, zorder=6)
    ax.plot([x_vertical, x_vertical], [y_top, y_bottom],
            color="black", lw=lw, clip_on=False, zorder=6)
    ax.plot([x_vertical, x_vertical + tick], [y_bottom, y_bottom],
            color="black", lw=lw, clip_on=False, zorder=6)
    ax.text(x_vertical - label_gap, (y_top + y_bottom) / 2, label,
            ha="center", va="center", rotation=90,
            fontsize=FONTSIZE["brace"], clip_on=False, zorder=6)


BASELINE_LW = 2.7
BASELINE_ALPHA = 0.95
SAM_BASELINE_ALPHA = 0.5
SAM_BASELINE_COLOR = "#B8396F"
SAM_STUB_FRAC = 0.08


def _top_aligned_hline(ax, y, color, alpha, lw=BASELINE_LW):
    base_trans = blended_transform_factory(ax.transAxes, ax.transData)
    trans = offset_copy(base_trans, fig=ax.figure, y=-lw / 2.0, units="points")
    ax.plot([0, 1], [y, y], transform=trans,
            ls="--", color=color, lw=lw, alpha=alpha, zorder=4,
            solid_capstyle="butt", dash_capstyle="butt")


def _draw_panel(ax, bars, base_adamw, base_sam):
    x = np.arange(len(bars))
    for offset, scores, base, color, label in (
        (-BAR_WIDTH / 2, [b.adamw for b in bars], base_adamw,
         COLOR["adamw"], LABEL["adamw"]),
        (+BAR_WIDTH / 2, [b.sam for b in bars], base_sam,
         COLOR["sam"], LABEL["sam"]),
    ):
        ax.bar(x + offset, [s - base for s in scores], BAR_WIDTH,
               bottom=base, color=color, edgecolor="none",
               linewidth=0, label=label)

    border_lw = 0.875
    for i in range(len(bars)):
        for offset, score in (
            (-BAR_WIDTH / 2, bars[i].adamw),
            (+BAR_WIDTH / 2, bars[i].sam),
        ):
            cx = i + offset
            x_l = cx - BAR_WIDTH / 2
            x_r = cx + BAR_WIDTH / 2
            ax.plot([x_l, x_l], [score, base_adamw],
                    color="black", lw=border_lw,
                    solid_capstyle="butt", zorder=3)
            ax.plot([x_r, x_r], [score, base_adamw],
                    color="black", lw=border_lw,
                    solid_capstyle="butt", zorder=3)
            ax.plot([x_l, x_r], [score, score],
                    color="black", lw=border_lw,
                    solid_capstyle="butt", zorder=3)

    _top_aligned_hline(ax, base_adamw, color=TEXT_COLOR["adamw"],
                       alpha=BASELINE_ALPHA)
    _top_aligned_hline(ax, base_sam, color=SAM_BASELINE_COLOR,
                       alpha=SAM_BASELINE_ALPHA)

    inline_labels = {"T\u00fclu-3", QUANT_LABEL}

    for i, b in enumerate(bars):
        if abs(b.improvement_pct) < 1.5:
            pct = "0%"
            imp_color = "#000000"
        else:
            pct = f"\u2212{b.improvement_pct:.0f}%"
            imp_color = "#15803D" if b.improvement_pct >= 0 else "#B91C1C"
        imp_label = f"{pct}\nforgetting" if b.label == "Meta-Math" else pct
        inline = b.label in inline_labels

        if b.improvement_pct >= 5:
            x_start = x[i] + ARROW_LEFT_OFFSET
            y_start = b.adamw + ARROW_BOTTOM_OFFSET
            x_knee  = x[i] + BAR_WIDTH / 2
            y_end   = b.sam - ARROW_TIP_OFFSET
            y_mid   = (y_start + y_end) / 2

            ax.plot([x_start, x_knee], [y_start, y_start],
                    ls=":", color=ARROW_COLOR, lw=ARROW_LW, zorder=5)

            if inline:
                gap = 0.22 if "\n" in imp_label else 0.12
                if y_mid - gap > y_start + 0.02:
                    ax.plot([x_knee, x_knee], [y_start, y_mid - gap],
                            ls=":", color=ARROW_COLOR, lw=ARROW_LW, zorder=5)
                if y_end > y_mid + gap + 0.02:
                    ax.plot([x_knee, x_knee], [y_mid + gap, y_end],
                            ls=":", color=ARROW_COLOR, lw=ARROW_LW, zorder=5)
                ax.plot(x_knee, y_end, marker="^", ms=4.375,
                        color=ARROW_COLOR, zorder=5, clip_on=False)
                ax.text(x_knee, y_mid, imp_label,
                        ha="center", va="center_baseline",
                        fontsize=FONTSIZE["improvement"],
                        fontweight="bold", color=imp_color, zorder=5,
                        path_effects=[pe.withStroke(linewidth=2,
                                                    foreground="white")])
            else:
                ax.plot([x_knee, x_knee], [y_start, y_end],
                        ls=":", color=ARROW_COLOR, lw=ARROW_LW, zorder=5)
                ax.plot(x_knee, y_end, marker="^", ms=4.375,
                        color=ARROW_COLOR, zorder=5, clip_on=False)
                ax.text(x[i] + BAR_WIDTH / 2, min(b.adamw, b.sam) - 0.08,
                        imp_label,
                        ha="center", va="top",
                        fontsize=FONTSIZE["improvement"],
                        fontweight="bold", color=imp_color, zorder=5,
                        path_effects=[pe.withStroke(linewidth=2,
                                                    foreground="white")])
        else:
            ax.text(x[i] + BAR_WIDTH / 2, min(b.adamw, b.sam) - 0.08,
                    imp_label,
                    ha="center", va="top", fontsize=FONTSIZE["improvement"],
                    fontweight="bold", color=imp_color, zorder=5,
                    path_effects=[pe.withStroke(linewidth=2,
                                                foreground="white")])

    display_labels = [TICK_LABEL_DISPLAY.get(b.label, b.label) for b in bars]
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=FONTSIZE["tick"], ha="center")
    ax.tick_params(axis="y", labelsize=FONTSIZE["tick"])
    ax.tick_params(axis="x", pad=1)
    ax.set_xlim(-0.65, len(bars) - 0.35)
    ax.grid(axis="y", alpha=0.22, zorder=0, linewidth=0.5)
    ax.set_axisbelow(True)


def render(data: FigureData) -> plt.Figure:
    ft_bars    = [b for b in data.bars if b.label != QUANT_LABEL]
    quant_bars = [b for b in data.bars if b.label == QUANT_LABEL]

    ymin = 38.5
    ymax = 43.5

    fig = plt.figure(figsize=(9.0, 4.3))
    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[len(ft_bars), len(quant_bars)], wspace=0.05,
        left=0.10, right=0.71, top=0.96, bottom=0.16,
    )
    ax_ft = fig.add_subplot(gs[0, 0])
    ax_q  = fig.add_subplot(gs[0, 1], sharey=ax_ft)

    _draw_panel(ax_ft, ft_bars,    data.baseline_adamw, data.baseline_sam)
    _draw_panel(ax_q,  quant_bars, data.baseline_adamw, data.baseline_sam)

    ax_ft.set_xlim(-0.65, len(ft_bars) - 0.35)

    ax_ft.set_ylim(ymin, ymax)
    ax_ft.set_yticks([39, 40, 41, 42, 43])
    ax_ft.set_ylabel("Pretraining benchmark\naccuracy",
                     fontsize=FONTSIZE["axis"])
    ax_ft.set_xlabel("Post-training dataset", fontsize=FONTSIZE["axis"])
    ax_q.tick_params(labelleft=False)
    for spine in list(ax_ft.spines.values()) + list(ax_q.spines.values()):
        spine.set_linewidth(0.5)

    bar_handles, bar_labels = ax_ft.get_legend_handles_labels()
    bar_by_label = dict(zip(bar_labels, bar_handles))
    baseline_adamw_line = Line2D([0], [0], ls="--", color=TEXT_COLOR["adamw"],
                                 lw=BASELINE_LW, alpha=BASELINE_ALPHA)
    baseline_sam_line = Line2D([0], [0], ls="--", color=SAM_BASELINE_COLOR,
                               lw=BASELINE_LW, alpha=SAM_BASELINE_ALPHA)
    base_legend = fig.legend(
        [baseline_adamw_line, baseline_sam_line],
        ["OLMo baseline", "SAM"],
        title="Base model",
        bbox_to_anchor=(0.74, 0.95), loc="upper left",
        ncol=1, frameon=False, fontsize=FONTSIZE["legend"],
        title_fontsize=FONTSIZE["legend"],
    )
    fig.add_artist(base_legend)
    fig.legend(
        [bar_by_label[LABEL["adamw"]], bar_by_label[LABEL["sam"]]],
        ["OLMo baseline", "SAM"],
        title="Post-trained",
        bbox_to_anchor=(0.74, 0.50), loc="upper left",
        ncol=1, frameon=False, fontsize=FONTSIZE["legend"],
        title_fontsize=FONTSIZE["legend"],
    )
    return fig


# --- entry point -----------------------------------------------------------

def main():
    records = json.loads(DATA_PATH.read_text())
    data = build_figure_data(records)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig = render(data)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close("all")
    print(f"Saved {OUT_PATH}")
    print(f"Baselines: AdamW={data.baseline_adamw:.2f}, SAM={data.baseline_sam:.2f}")
    for b in data.bars:
        print(f"  {b.label}: AdamW={b.adamw:.2f} (-{b.adamw_drop_pct:.1f}%), "
              f"SAM={b.sam:.2f} (-{b.sam_drop_pct:.1f}%), "
              f"{b.improvement_pct:+.1f}% reduction")


if __name__ == "__main__":
    main()
