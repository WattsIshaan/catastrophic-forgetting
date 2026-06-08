"""Animated, talk-slide variant of Figure 1.

Builds an 8s animation (then loops):

  1. 0-1s   empty chart with both legends already shown (no lines, no bars).
  2. 1-3s   the two horizontal *base-model* lines fade in (OLMo slightly above
            SAM)  ->  caption: "SAM has worse performance after pretraining".
  3. 3-4s   the post-training / quantization bars drop down.
  4. 4-8s   green "XX%" callouts appear (caption: "...but less forgetting after
            SFT & quantization") and the final result is held.

This file does NOT modify ``generate_talk_figure_1.py`` (a frozen, hand-tuned
artifact). It only *imports* that module's data pipeline + styling constants and
re-implements the rendering so it can be drawn incrementally for animation.

Outputs (into ``assets/``):
  - ``figure1_animation.gif``  (always; universal, autoplays via <img>)
  - ``figure1_animation.mp4``  (only if an ffmpeg encoder is discoverable)
Also writes a few keyframe PNGs into ``figures/_anim_keyframes/`` for review.
"""

import math
import os
from pathlib import Path

# MPLCONFIGDIR must be writable + set before matplotlib is imported anywhere.
# This script lives at <website>/figure_generation/, so the website root
# (which holds assets/ and .mplcache) is one level up.
WS = Path(__file__).resolve().parents[1]
_mpldir = WS / ".mplcache"
_mpldir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpldir))

import matplotlib
matplotlib.use("Agg")
import json
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import generate_talk_figure_1 as base  # frozen artifact: data pipeline + constants


# --- config ----------------------------------------------------------------

GREEN = "#15803D"
RED = "#B91C1C"
BORDER_LW = 0.875

FPS_MP4 = 25      # smooth primary video
FPS_GIF = 12.5    # fewer frames -> smaller fallback gif

# Timeline in SECONDS (total 8s, then loops):
#   0.0 - 1.0    empty chart + both legends (no lines, no bars)
#   1.0 - 3.0    horizontal base-model lines + "SAM has worse performance
#                after pretraining"  (visible 2s)
#   3.0 - 4.0    bars drop down (1s)
#   4.0 - 8.0    hold the final result (4s)
T_TOTAL = 8.0
S_LINES = (1.0, 1.4)    # base lines fade in
S_MSG1 = (1.2, 2.0)     # caption 1 fade in
S_BARS = (3.0, 4.0)     # bars drop (eased over 1s)
S_MSG2 = (3.3, 4.0)     # caption 2 + green callouts fade in


def _seg(t, a, b):
    if t <= a:
        return 0.0
    if t >= b:
        return 1.0
    return (t - a) / (b - a)


def _ease_in_out(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)


# --- incremental drawing ---------------------------------------------------

def _draw_panel(ax, bars, base_adamw, base_sam, bar_t, base_alpha, label_alpha):
    x = np.arange(len(bars))

    if base_alpha > 0.001:
        base._top_aligned_hline(ax, base_adamw, color=base.TEXT_COLOR["adamw"],
                                alpha=base.BASELINE_ALPHA * base_alpha)
        base._top_aligned_hline(ax, base_sam, color=base.SAM_BASELINE_COLOR,
                                alpha=base.SAM_BASELINE_ALPHA * base_alpha)

    bw = base.BAR_WIDTH
    if bar_t > 0.001:
        for offset, scores, b0, color in (
            (-bw / 2, [b.adamw for b in bars], base_adamw, base.COLOR["adamw"]),
            (+bw / 2, [b.sam for b in bars], base_sam, base.COLOR["sam"]),
        ):
            cur = [b0 + (s - b0) * bar_t for s in scores]
            ax.bar(x + offset, [c - b0 for c in cur], bw, bottom=b0,
                   color=color, edgecolor="none", linewidth=0, zorder=2)
            for i, c in enumerate(cur):
                cx = i + offset
                xl, xr = cx - bw / 2, cx + bw / 2
                ax.plot([xl, xl], [c, base_adamw], color="black", lw=BORDER_LW,
                        solid_capstyle="butt", zorder=3)
                ax.plot([xr, xr], [c, base_adamw], color="black", lw=BORDER_LW,
                        solid_capstyle="butt", zorder=3)
                ax.plot([xl, xr], [c, c], color="black", lw=BORDER_LW,
                        solid_capstyle="butt", zorder=3)

    if label_alpha > 0.01:
        a = label_alpha
        for i, b in enumerate(bars):
            y_lo, y_hi = b.adamw, b.sam  # SAM (y_hi) forgets less -> sits higher
            x_knee = x[i] + bw / 2
            ax.plot([x[i] + base.ARROW_LEFT_OFFSET, x_knee], [y_lo, y_lo],
                    ls=":", color="black", lw=base.ARROW_LW, alpha=a, zorder=5)
            ax.plot([x_knee, x_knee], [y_lo, y_hi],
                    ls=":", color="black", lw=base.ARROW_LW, alpha=a, zorder=5)
            ax.plot(x_knee, y_hi, marker="^", ms=5, color="black", alpha=a,
                    zorder=5, clip_on=False)
            pct = f"{b.improvement_pct:.0f}%"
            if b.label == "Meta-Math":
                # explanatory label below the bars (room for the extra words)
                ax.text(x[i], min(y_lo, y_hi) - 0.18, f"{pct}\nless forgetting",
                        ha="center", va="top",
                        fontsize=base.FONTSIZE["improvement"],
                        fontweight="bold", color=GREEN, alpha=a, zorder=6,
                        path_effects=[pe.withStroke(linewidth=2,
                                                    foreground="white")])
            else:
                ax.text(x_knee - 0.16, (y_lo + y_hi) / 2, pct, ha="right",
                        va="center", fontsize=base.FONTSIZE["improvement"],
                        fontweight="bold", color=GREEN, alpha=a, zorder=6,
                        path_effects=[pe.withStroke(linewidth=2,
                                                    foreground="white")])


def _style(ax_ft, ax_q, ft_bars, quant_bars):
    for ax in (ax_ft, ax_q):
        ax.set_ylim(38.5, 43.5)
        ax.grid(axis="y", alpha=0.22, zorder=0, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=base.FONTSIZE["tick"])
        ax.tick_params(axis="x", pad=1)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
    ax_ft.set_yticks([39, 40, 41, 42, 43])
    ax_ft.set_xlim(-0.65, len(ft_bars) - 0.35)
    ax_q.set_xlim(-0.65, len(quant_bars) - 0.35)
    ax_ft.set_xticks(np.arange(len(ft_bars)))
    ax_ft.set_xticklabels(
        [base.TICK_LABEL_DISPLAY.get(b.label, b.label) for b in ft_bars],
        fontsize=base.FONTSIZE["tick"])
    ax_q.set_xticks(np.arange(len(quant_bars)))
    ax_q.set_xticklabels(
        [base.TICK_LABEL_DISPLAY.get(b.label, b.label) for b in quant_bars],
        fontsize=base.FONTSIZE["tick"])
    ax_ft.set_ylabel("Pretraining benchmark\naccuracy",
                     fontsize=base.FONTSIZE["axis"])
    ax_ft.set_xlabel("Post-training dataset", fontsize=base.FONTSIZE["axis"])
    ax_q.tick_params(labelleft=False)


# --- figure + animation ----------------------------------------------------

def build(data, fps):
    ft_bars = [b for b in data.bars if b.label != base.QUANT_LABEL]
    quant_bars = [b for b in data.bars if b.label == base.QUANT_LABEL]
    n_frames = int(round(T_TOTAL * fps)) + 1

    fig = plt.figure(figsize=(11.0, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[len(ft_bars), len(quant_bars)],
                          wspace=0.05, left=0.14, right=0.62,
                          top=0.80, bottom=0.20)
    ax_ft = fig.add_subplot(gs[0, 0])
    ax_q = fig.add_subplot(gs[0, 1], sharey=ax_ft)

    title1 = fig.text(0.38, 0.95, "SAM has worse performance after pretraining",
                      ha="center", va="center", fontsize=18,
                      fontweight="bold", color=RED, alpha=0.0)
    title2 = fig.text(0.38, 0.875,
                      "\u2026but less forgetting after SFT \u0026 quantization",
                      ha="center", va="center", fontsize=18,
                      fontweight="bold", color=GREEN, alpha=0.0)

    base_proxies = [
        Line2D([0], [0], ls="--", color=base.TEXT_COLOR["adamw"],
               lw=base.BASELINE_LW, alpha=base.BASELINE_ALPHA),
        Line2D([0], [0], ls="--", color=base.SAM_BASELINE_COLOR,
               lw=base.BASELINE_LW, alpha=base.SAM_BASELINE_ALPHA),
    ]
    post_proxies = [
        Patch(facecolor=base.COLOR["adamw"], edgecolor="black", lw=0.8),
        Patch(facecolor=base.COLOR["sam"], edgecolor="black", lw=0.8),
    ]
    # both legends are shown from the very start
    base_legend = fig.legend(base_proxies, ["OLMo baseline", "SAM"],
                             title="Base model", bbox_to_anchor=(0.645, 0.92),
                             loc="upper left", frameon=False,
                             fontsize=base.FONTSIZE["legend"],
                             title_fontsize=base.FONTSIZE["legend"])
    fig.add_artist(base_legend)
    fig.legend(post_proxies, ["OLMo baseline", "SAM"],
               title="Post-trained", bbox_to_anchor=(0.645, 0.52),
               loc="upper left", frameon=False,
               fontsize=base.FONTSIZE["legend"],
               title_fontsize=base.FONTSIZE["legend"])

    def update(f):
        t = f / fps
        line_a = _seg(t, *S_LINES)
        msg1 = _seg(t, *S_MSG1)
        bar_t = _ease_in_out(_seg(t, *S_BARS))
        msg2 = _seg(t, *S_MSG2)

        ax_ft.cla()
        ax_q.cla()
        _draw_panel(ax_ft, ft_bars, data.baseline_adamw, data.baseline_sam,
                    bar_t, line_a, msg2)
        _draw_panel(ax_q, quant_bars, data.baseline_adamw, data.baseline_sam,
                    bar_t, line_a, msg2)
        _style(ax_ft, ax_q, ft_bars, quant_bars)

        title1.set_alpha(msg1)
        title2.set_alpha(msg2)
        return []

    return fig, update, n_frames


def main():
    records = json.loads(base.DATA_PATH.read_text())
    data = base.build_figure_data(records)

    out_dir = WS / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    kf_dir = base.HERE / "figures" / "_anim_keyframes"
    kf_dir.mkdir(parents=True, exist_ok=True)

    # poster + review keyframes (rendered at representative wall-clock times)
    fig, update, n_frames = build(data, FPS_GIF)
    update(n_frames - 1)
    fig.savefig(out_dir / "figure1_poster.png", dpi=150)
    print(f"Saved poster to {out_dir / 'figure1_poster.png'}")
    for t in (0.5, 2.0, 3.5, T_TOTAL - 0.1):
        update(min(int(round(t * FPS_GIF)), n_frames - 1))
        fig.savefig(kf_dir / f"frame_t{t:04.1f}.png", dpi=110)
    print(f"Saved keyframes to {kf_dir}")

    if os.environ.get("ANIM_KEYFRAMES_ONLY"):
        plt.close("all")
        return

    # GIF fallback (lower fps + dpi to keep the 14s clip lightweight)
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / FPS_GIF)
    gif_path = out_dir / "figure1_animation.gif"
    anim.save(gif_path, writer=PillowWriter(fps=FPS_GIF), dpi=80)
    print(f"Saved {gif_path}")
    plt.close(fig)

    # MP4 (primary): higher fps + dpi, only if an ffmpeg binary is discoverable
    try:
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    try:
        from matplotlib.animation import FFMpegWriter
        if FFMpegWriter.isAvailable():
            fig_m, update_m, n_m = build(data, FPS_MP4)
            anim_m = FuncAnimation(fig_m, update_m, frames=n_m,
                                   interval=1000 / FPS_MP4)
            mp4_path = out_dir / "figure1_animation.mp4"
            anim_m.save(mp4_path, writer=FFMpegWriter(
                fps=FPS_MP4, bitrate=2400,
                extra_args=["-pix_fmt", "yuv420p"]), dpi=140)
            print(f"Saved {mp4_path}")
            plt.close(fig_m)
        else:
            print("ffmpeg not available - skipped mp4 (gif written)")
    except Exception as e:  # pragma: no cover
        print(f"mp4 export skipped: {e}")

    plt.close("all")


if __name__ == "__main__":
    main()
