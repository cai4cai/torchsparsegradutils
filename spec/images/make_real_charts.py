"""Generate the real-data sweep charts embedded in ../benchmarks.md §6.

Run:  uv run --with matplotlib python spec/images/make_real_charts.py

Successor of the retired make_dummy_charts.py (git history; same palette,
spine/grid treatment, parity line, endpoint direct-labels — no redesign).
Every point is read
from a persisted result JSON under benchmarks/results/; nothing is
invented. Slots whose §6 sweep (p-sweep over {1..512}, B-sweep, seglse
density-sweep) has no sweep JSONs yet are filled with the closest sweep
the real data supports — see the chart docstrings and benchmarks.md §6.
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

OUT = Path(__file__).parent
RESULTS = OUT.parent.parent / "benchmarks" / "results"

# Palette: fixed hue per kernel family (validated, light surface) — same
# assignments as benchmarks/viz.py FAMILY_COLORS; e2e hue is new, never
# recycled from another family.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e7e6e2"
BLUE_SPMM = "#2a78d6"  # spmm + sddmm families
AQUA_SEGLSE = "#1baf7a"  # seglse family
RED_SPSM = "#c0392b"  # spsm family
PURPLE_GG = "#8e5fd6"  # grouped_gemm family
YELLOW_BASE = "#eda100"  # coo2csr + baseline paths in multi-series charts
MAGENTA_E2E = "#c9528e"  # e2e composites (rsample, CG)


def load(name):
    return json.loads((RESULTS / name).read_text())


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def styled_axes(figsize=(7, 4)):
    fig, ax = plt.subplots(figsize=figsize, dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    return fig, ax


def source_note(ax, text, loc="left", y=0.02):
    x, ha = (0.01, "left") if loc == "left" else (0.99, "right")
    ax.annotate(text, xy=(x, y), xycoords="axes fraction", ha=ha,
                fontsize=7, color=TEXT_2, alpha=0.8, style="italic")


def sweep_chart(fname, title, subtitle, xlabel, x, y, color, log2_x, ymax, source,
                source_y=0.02):
    fig, ax = styled_axes()

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
        ax.get_xaxis().set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M" if v >= 1e5 else f"{int(v)}"))
        ax.minorticks_off()
    else:
        ax.set_xticks(x)

    ax.set_ylim(0, ymax)
    ax.set_xlabel(xlabel, fontsize=9, color=TEXT_2)
    ax.set_ylabel("speedup ×", fontsize=9, color=TEXT_2)
    ax.set_title(subtitle, fontsize=9, color=TEXT_2, loc="left", pad=4)
    fig.suptitle(title, fontsize=11, color=TEXT, x=0.125, ha="left", y=0.97)
    source_note(ax, source, y=source_y)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / fname, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / fname)


# ---------------------------------------------------------------------------
# Slot 1 (§6 p-sweep) — the op-level p-sweep over {1..512} has no sweep JSONs
# yet; the real p-sweep in results/ is the e2e CG composite at p ∈ {1, 8}.
cg = [load("e2e_cg_n4096_p1.json"), load("e2e_cg_n4096_p8.json")]
sweep_chart(
    "p_sweep_cg.png",
    "e2e CG p-sweep — synth-spd-n4096, CSR, f32/i64",
    "speedup vs dense torch.linalg.solve · flat in p: the solve is SpMV-bound, not RHS-bound",
    "n_rhs p (log₂)",
    [r["p"] for r in cg],
    [r["speedup"] for r in cg],
    MAGENTA_E2E,
    log2_x=True,
    ymax=20.0,
    source="benchmarks/results/e2e_cg_n4096_p{1,8}.json · 2026-07-17",
    source_y=0.09,  # clear of the parity line, which sits at 5% of this axis
)

# ---------------------------------------------------------------------------
# Slot 2 (§6 B-sweep / nse-sweep) — no B-sweep JSONs exist (all runs pin one
# B per op); the real problem-size sweep is the e2e rsample encoder sweep.
SIZES = ("32", "48", "64")
rs_csr = [load(f"e2e_rsample_csr_encoder-2x{s}^3.json") for s in SIZES]
rs_coo = [load(f"e2e_rsample_coo_encoder-2x{s}^3.json") for s in SIZES]
sweep_chart(
    "nse_sweep_rsample.png",
    "e2e rsample nse-sweep — encoder 2×{32,48,64}³, B=1, f32/i64",
    "speedup CSR vs COO layout · the CSR win holds ~4× across a 8× nse range",
    "nse (log₂)",
    [r["nse"] for r in rs_csr],
    [r["speedup"] for r in rs_csr],
    MAGENTA_E2E,
    log2_x=True,
    ymax=6.0,
    source="benchmarks/results/e2e_rsample_{csr,coo}_encoder-2x{32,48,64}^3.json · 2026-07-17",
)


# ---------------------------------------------------------------------------
# Slot 3 (§6 memory B-sweep) — same substitution as slot 2: the measured
# memory sweep is over encoder size, not B. Dense-grad counterfactual is the
# computed n·n·4-byte materialisation (benchmarks.md §5), clearly labelled.
def memory_chart(fname):
    """Memory nse-sweep: 3 series → legend + direct endpoint labels."""
    fig, ax = styled_axes()

    nse = [r["nse"] for r in rs_csr]
    dense = [r["n"] * r["n"] * 4 / 1e6 for r in rs_csr]  # computed, not measured
    series = [
        ("ours (rsample, CSR encoder) peak_bwd", [r["peak_bwd_mb"] for r in rs_csr],
         MAGENTA_E2E, "-"),
        ("rsample, COO encoder peak_bwd", [r["peak_bwd_mb"] for r in rs_coo],
         YELLOW_BASE, "-"),
        ("dense-grad counterfactual n²·4B (computed)", dense, TEXT_2, (0, (4, 4))),
    ]
    for name, y, color, ls in series:
        ax.plot(nse, y, color=color, linewidth=2, linestyle=ls, marker="o",
                markersize=7, markerfacecolor=color, markeredgecolor=SURFACE,
                markeredgewidth=1.5, zorder=3, label=name)
        label = f"{y[-1] / 1024:.0f} GB" if y[-1] >= 1024 else f"{y[-1]:.0f} MB"
        ax.annotate(label, xy=(nse[-1], y[-1]), xytext=(6, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=9, color=TEXT, fontweight="bold")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(nse)
    ax.get_xaxis().set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M"))
    ax.minorticks_off()
    ax.set_xlim(right=nse[-1] * 2.6)  # room for endpoint labels
    ax.set_xlabel("nse (log₂)", fontsize=9, color=TEXT_2)
    ax.set_ylabel("peak backward memory, MB (log)", fontsize=9, color=TEXT_2)
    ax.set_title("peak_bwd · encoder 2×{32,48,64}³, B=1, f32/i64 · CSR ≈ 0.56× COO at every size",
                 fontsize=9, color=TEXT_2, loc="left", pad=4)
    fig.suptitle("memory nse-sweep — e2e rsample backward", fontsize=11,
                 color=TEXT, x=0.125, ha="left", y=0.97)
    # framed in surface ink so the dashed counterfactual line doesn't strike
    # through the legend text (dummy chart's frameon=False had no collision)
    leg = ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=1.0,
                    facecolor=SURFACE, edgecolor="none")
    for t in leg.get_texts():
        t.set_color(TEXT_2)
    source_note(ax, "benchmarks/results/e2e_rsample_*_encoder-2x{32,48,64}^3.json · 2026-07-17",
                loc="right")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / fname, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / fname)


memory_chart("memory_nse_sweep_rsample.png")

# ---------------------------------------------------------------------------
# Slot 4 (§6 distribution strip) — the seglse density-sweep has no sweep
# JSONs yet; the strip chart is the §6 type the full table's real data does
# support: per-row speedup, one hue per family, geo-means marked.
STRIP_ROWS = [
    # (family, label, json, in vendor geo-mean, in batched/fused geo-mean)
    ("sddmm", "sddmm B=1 vs cuSPARSE", "sddmm_custom_20260716T153519888592+0000.json", True, False),
    ("spsm", "spsm cold vs cuSPARSE", "spsm_cold_custom_20260716T184319519807+0000.json", True, False),
    ("spsm", "spsm warm vs cuSPARSE", "spsm_warm_custom_20260716T184319519807+0000.json", True, False),
    ("spmm", "spmm B=8 vs block-diag", "spmm_custom_20260716T171149781803+0000.json", False, True),
    ("sddmm", "sddmm B=8 vs pure-torch", "sddmm_custom_20260716T155529170866+0000.json", False, True),
    ("spsm", "spsm batched vs looped", "spsm_batched_custom_20260716T184319519807+0000.json", False, True),
    ("grouped_gemm", "grouped_gemm vs mm-loop", "grouped_gemm_custom_20260717T174200830899+0000.json", False, True),
    ("coo2csr", "coo2csr vs pure-torch", "coo2csr_custom_20260717T180049584105+0000.json", False, True),
    ("seglse", "seglse vs pytorch_scatter", "seglse_custom_20260715T052716070439+0000_row0.json", False, True),
    ("seglse", "seglse vs oracle-A", "seglse_custom_20260715T052716070439+0000_row1.json", False, True),
    ("seglse", "seglse_bidir vs 2× seglse", "seglse_bidir_vs_2x_seglse.json", False, True),
    ("seglse", "seglse_bidir vs scatter", "seglse_bidir_vs_pytorch_scatter.json", False, True),
    ("seglse", "seglse_bidir vs oracle-A", "seglse_bidir_vs_oracle.json", False, True),
    ("e2e", "rsample CSR vs COO", "e2e_rsample_csr_encoder-2x64^3.json", False, True),
]
FAMILY_COLORS = {
    "spmm": BLUE_SPMM, "sddmm": BLUE_SPMM, "seglse": AQUA_SEGLSE,
    "spsm": RED_SPSM, "grouped_gemm": PURPLE_GG, "coo2csr": YELLOW_BASE,
    "e2e": MAGENTA_E2E,
}


def strip_chart(fname):
    fig, ax = styled_axes(figsize=(7, 4.6))
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.grid(axis="y", visible=False)

    rows = [(fam, lbl, load(f)["speedup"], v, b) for fam, lbl, f, v, b in STRIP_ROWS]
    lanes = ["spmm", "sddmm", "spsm", "grouped_gemm", "coo2csr", "seglse", "e2e"]
    for i, (fam, lbl, sp, _v, _b) in enumerate(rows):
        y = lanes.index(fam)
        color = FAMILY_COLORS[fam]
        ax.plot([sp], [y], marker="o", markersize=8, markerfacecolor=color,
                markeredgecolor=SURFACE, markeredgewidth=1.5, color=color, zorder=3)

    # parity reference — vertical here (x is the speedup axis); labelled at the
    # foot of the line so it can't collide with the geo-mean labels up top
    ax.axvline(1.0, color=TEXT_2, linewidth=1, linestyle=(0, (4, 4)), zorder=1)
    ax.annotate("parity 1.0× ", xy=(1.0, 0.02), xycoords=("data", "axes fraction"),
                ha="right", va="bottom", fontsize=8, color=TEXT_2)

    # geo-means marked (benchmarks.md §1: geometric mean, never arithmetic)
    gm_vendor = geomean([sp for _f, _l, sp, v, _b in rows if v])
    gm_batched = geomean([sp for _f, _l, sp, _v, b in rows if b])
    for gm, name in ((gm_vendor, "geo-mean vendor"), (gm_batched, "geo-mean batched/fused")):
        ax.axvline(gm, color=TEXT, linewidth=1.2, linestyle=(0, (1, 2)), zorder=2)
        ax.annotate(f"{name} {gm:.2f}×", xy=(gm, 1.02),
                    xycoords=("data", "axes fraction"), ha="left", va="bottom",
                    fontsize=8, color=TEXT, fontweight="bold", rotation=0)

    # direct label the extreme point (seglse vs pytorch_scatter)
    top = max(rows, key=lambda r: r[2])
    ax.annotate(f"{top[2]:.0f}×", xy=(top[2], lanes.index(top[0])), xytext=(0, 10),
                textcoords="offset points", ha="center", fontsize=9,
                color=TEXT, fontweight="bold")

    ax.set_xscale("log", base=2)
    ax.set_xlim(0.5, 200)
    ax.set_xticks([0.5, 1, 2, 4, 8, 16, 32, 64, 128])
    ax.get_xaxis().set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.minorticks_off()
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels(lanes)
    ax.set_ylim(-0.6, len(lanes) - 0.4)
    ax.set_xlabel("speedup vs row's named baseline (log₂)", fontsize=9, color=TEXT_2)
    ax.set_title("one dot per §5 table row · latest JSON per op/baseline · synthetic tier",
                 fontsize=9, color=TEXT_2, loc="left", pad=14)
    fig.suptitle("distribution strip — per-row speedup, geo-means marked",
                 fontsize=11, color=TEXT, x=0.125, ha="left", y=0.97)
    source_note(ax, "benchmarks/results/*.json (14 rows) · 2026-07-15..17", loc="right")

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / fname, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / fname)
    print(f"geo-mean vendor-baseline rows:  {gm_vendor:.3f}×")
    print(f"geo-mean batched/fused rows:    {gm_batched:.3f}×")


strip_chart("distribution_strip.png")
