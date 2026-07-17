"""benchmarks/viz.py — sweep chart renderer.

spec/benchmarks.md §6: "The table is the record; sweeps are the argument
... The suite's viz script (successor of
``visualize_benchmark_results.py``) renders the chart set below from the
CSV [here: from the JSON records ``benchmarks/results.py`` writes]." Chart
rules: log2 x-axes, y = speedup vs named baseline with a parity line at
1.0x, one hue per kernel family (fixed assignment, never recycled), no dual
axes, endpoint direct-labelled, dark/light both rendered.

Styling matches ``spec/images/make_real_charts.py`` (palette,
spine/grid treatment, parity line, endpoint labels) so real charts drop in
without a restyle once kernel data exists.

Only single-series speedup-vs-baseline line charts are implemented (the
p-sweep/B-sweep/nse-sweep/density-sweep shape from benchmarks.md §6's
table); the 3-series memory B-sweep chart is future work once there's a
custom-kernel row to compare against a block-diag path and a dense-grad
counterfactual (none of those exist yet -- Phase 3).

Rows with no ``speedup`` (e.g. this commit's oracle-only baseline row,
which has no kernel to compare against) are skipped, not plotted as a fake
1.0x -- there is deliberately nothing to chart yet, and :func:`render_all`
says so rather than fabricating a chart.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set before this import)

RESULTS_DIR = Path(__file__).parent / "results"
OUT_DIR = Path(__file__).parent / "charts"

# Palette -- shared with spec/images/make_real_charts.py so real
# charts match the spec's sketches exactly. One fixed hue per kernel family,
# never recycled (benchmarks.md §6).
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e7e6e2"
FAMILY_COLORS = {
    "SpMM": "#2a78d6",
    "SDDMM": "#2a78d6",
    "seglse": "#1baf7a",
    "spsm": "#c0392b",
    "grouped_gemm": "#8e5fd6",
    "coo2csr": "#eda100",
}
_DEFAULT_COLOR = "#eda100"


def load_results(results_dir: Path = RESULTS_DIR) -> list[dict]:
    if not results_dir.exists():
        return []
    records = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            records.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return records


def _style_axes(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)


def speedup_sweep_chart(
    records: list[dict],
    *,
    x_field: str,
    title: str,
    subtitle: str,
    xlabel: str,
    fname: str,
    log2_x: bool = True,
    out_dir: Path = OUT_DIR,
) -> Optional[Path]:
    """One family's speedup-vs-baseline line, x = ``x_field`` (e.g. "p",
    "n"). Returns ``None`` (writes nothing) if no record has both
    ``x_field`` and a non-null ``speedup``."""
    pts = [
        (r[x_field], r["speedup"], r.get("family"))
        for r in records
        if r.get(x_field) is not None and r.get("speedup") is not None
    ]
    if not pts:
        return None
    pts.sort(key=lambda t: t[0])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    color = FAMILY_COLORS.get(pts[0][2], _DEFAULT_COLOR)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    _style_axes(ax)

    ax.axhline(1.0, color=TEXT_2, linewidth=1, linestyle=(0, (4, 4)), zorder=1)
    ax.annotate(
        "parity 1.0×",
        xy=(0.99, 1.0),
        xycoords=("axes fraction", "data"),
        ha="right",
        va="bottom",
        fontsize=8,
        color=TEXT_2,
    )

    ax.plot(
        xs,
        ys,
        color=color,
        linewidth=2,
        marker="o",
        markersize=7,
        markerfacecolor=color,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        zorder=3,
    )
    ax.annotate(
        f"{ys[-1]:.2f}×",
        xy=(xs[-1], ys[-1]),
        xytext=(0, 9),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        color=TEXT,
        fontweight="bold",
    )

    if log2_x:
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.minorticks_off()

    ax.set_xlabel(xlabel, fontsize=9, color=TEXT_2)
    ax.set_ylabel("speedup ×", fontsize=9, color=TEXT_2)
    ax.set_title(subtitle, fontsize=9, color=TEXT_2, loc="left", pad=4)
    fig.suptitle(title, fontsize=11, color=TEXT, x=0.125, ha="left", y=0.97)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / fname
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return out_path


# The spec's chart set (benchmarks.md §6), reduced to what a flat JSON-record
# reader can drive generically; each entry becomes one speedup_sweep_chart()
# call once matching rows exist.
_CHART_SPECS = (
    dict(x_field="p", title="p-sweep", subtitle="speedup vs baseline", xlabel="n_rhs p (log₂)", fname="p_sweep.png"),
    dict(
        x_field="n",
        title="nse-sweep",
        subtitle="speedup vs baseline",
        xlabel="problem size n (log₂)",
        fname="n_sweep.png",
    ),
)


def render_all(results_dir: Path = RESULTS_DIR, out_dir: Path = OUT_DIR) -> list[Path]:
    """Render the spec's chart set from whatever's in ``results_dir``.
    Returns only the charts actually written -- skips any chart with no
    matching (``speedup``-bearing) rows."""
    records = load_results(results_dir)
    written = []
    for spec in _CHART_SPECS:
        path = speedup_sweep_chart(records, out_dir=out_dir, **spec)
        if path is not None:
            written.append(path)
    return written


if __name__ == "__main__":
    paths = render_all()
    if not paths:
        print(
            "no chartable rows in benchmarks/results/ yet (no result carries a "
            "`speedup` -- expected until a tsgu:: kernel lands, Phase 3)"
        )
    for p in paths:
        print("wrote", p)
