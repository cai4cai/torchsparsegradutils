"""Generate the dummy sweep charts embedded in ../benchmarks.md §6.

Run:  uv run --with matplotlib python make_dummy_charts.py
All values are illustrative placeholders — regenerate with real data by
swapping the DATA blocks once the suite produces CSVs.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent

# Palette: fixed hue per kernel family (validated, light surface)
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e7e6e2"
BLUE_SPMM = "#2a78d6"  # spmm family
AQUA_SEGLSE = "#1baf7a"  # seglse family
YELLOW_BASE = "#eda100"  # baseline paths in multi-series charts


def sweep_chart(fname, title, subtitle, xlabel, x, y, color, log2_x, ymax):
    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # recessive grid + axes
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)

    # parity reference — neutral, dashed, labelled directly (no legend needed)
    ax.axhline(1.0, color=TEXT_2, linewidth=1, linestyle=(0, (4, 4)), zorder=1)
    ax.annotate("parity 1.0×", xy=(0.99, 1.0), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=8, color=TEXT_2)

    # the one series — 2px line, 8px markers; title carries its identity
    ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=7,
            markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.5,
            zorder=3)

    # selective direct label: endpoint only, in text ink (not series color)
    ax.annotate(f"{y[-1]:.2f}×", xy=(x[-1], y[-1]), xytext=(0, 9),
                textcoords="offset points", ha="center", fontsize=9,
                color=TEXT, fontweight="bold")

    if log2_x:
        ax.set_xscale("log", base=2)
        ax.set_xticks(x)
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.minorticks_off()
    else:
        ax.set_xticks(x)

    ax.set_ylim(0, ymax)
    ax.set_xlabel(xlabel, fontsize=9, color=TEXT_2)
    ax.set_ylabel("speedup ×", fontsize=9, color=TEXT_2)
    ax.set_title(subtitle, fontsize=9, color=TEXT_2, loc="left", pad=4)
    fig.suptitle(title, fontsize=11, color=TEXT, x=0.125, ha="left", y=0.97)

    ax.annotate("DUMMY DATA", xy=(0.01, 0.02), xycoords="axes fraction",
                fontsize=8, color=TEXT_2, alpha=0.8, style="italic")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / fname, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / fname)


sweep_chart(
    "p_sweep_spmm.png",
    "spmm p-sweep — cfd2, CSR, B=1, f32",
    "speedup vs cuSPARSE SpMM · parity crossover near p=8 (launch overhead below)",
    "n_rhs p (log₂)",
    [1, 8, 32, 128, 512],
    [0.97, 1.02, 1.05, 1.06, 1.11],
    BLUE_SPMM,
    log2_x=True,
    ymax=2.0,
)

sweep_chart(
    "b_sweep_spmm.png",
    "spmm B-sweep — synth-ragged 10k², f32/i32",
    "speedup vs block-diag + cuSPARSE · the win grows with B (index bloat is theirs)",
    "batch size B (log₂)",
    [1, 8, 64, 256],
    [1.1, 3.4, 6.7, 8.9],
    BLUE_SPMM,
    log2_x=True,
    ymax=10.0,
)

def memory_chart(fname):
    """Memory B-sweep: 3 series → legend + direct endpoint labels (contrast relief)."""
    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)

    B = [1, 8, 64, 256]
    series = [
        ("ours (tsgu::spmm bwd)", [25, 200, 1_600, 6_400], BLUE_SPMM, "-"),
        ("block-diag path (git)", [48, 420, 4_800, 26_000], YELLOW_BASE, "-"),
        ("dense-grad counterfactual", [800, 6_400, 51_200, 204_800], TEXT_2, (0, (4, 4))),
    ]
    for name, y, color, ls in series:
        ax.plot(B, y, color=color, linewidth=2, linestyle=ls, marker="o",
                markersize=7, markerfacecolor=color, markeredgecolor=SURFACE,
                markeredgewidth=1.5, zorder=3, label=name)
        ax.annotate(f"{y[-1]/1024:.0f} GB", xy=(B[-1], y[-1]), xytext=(6, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=9, color=TEXT, fontweight="bold")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(B)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.minorticks_off()
    ax.set_xlim(right=B[-1] * 2.6)  # room for endpoint labels
    ax.set_xlabel("batch size B (log₂)", fontsize=9, color=TEXT_2)
    ax.set_ylabel("peak backward memory, MB (log)", fontsize=9, color=TEXT_2)
    ax.set_title("peak_bwd · synth-ragged 10k², ~500k nse/item, f32/i32",
                 fontsize=9, color=TEXT_2, loc="left", pad=4)
    fig.suptitle("memory B-sweep — sparse_mm backward", fontsize=11,
                 color=TEXT, x=0.125, ha="left", y=0.97)
    leg = ax.legend(loc="upper left", fontsize=8, frameon=False)
    for t in leg.get_texts():
        t.set_color(TEXT_2)
    ax.annotate("DUMMY DATA", xy=(0.99, 0.02), xycoords="axes fraction",
                ha="right", fontsize=8, color=TEXT_2, alpha=0.8, style="italic")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / fname, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / fname)


sweep_chart(
    "density_sweep_seglse.png",
    "seglse density-sweep — DLMC, CSR, B=1, f32",
    "speedup vs pytorch_scatter · dip at 98%: segments too short to fill warps",
    "sparsity %",
    [70, 80, 90, 95, 98],
    [1.6, 2.0, 2.4, 2.8, 2.6],
    AQUA_SEGLSE,
    log2_x=False,
    ymax=4.0,
)

memory_chart("memory_b_sweep_spmm.png")
