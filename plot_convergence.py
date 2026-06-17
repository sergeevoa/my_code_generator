#!/usr/bin/env python3
"""
Plot convergence curves from a summary JSON file.

Usage:
    python plot_convergence.py path/to/summary.json
    python plot_convergence.py C:/results/experiment_summary.json
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Constants ─────────────────────────────────────────────────────────────────

TITLE   = "Кривые сходимости агентов"

LABEL_B0 = "Агент в базовой конфигурации"
LABEL_B1 = "Агент с трассами исполнения"

COLOR_B0 = "#1a6faf"   # насыщенный синий
COLOR_B1 = "#c0392b"   # насыщенный красный

MARKER_B0 = "o"
MARKER_B1 = "s"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_path(arg: str) -> Path:
    """Return an existing Path for *arg*."""
    p = Path(arg).resolve()
    if not p.exists():
        print(f"Файл не найден: {arg}", file=sys.stderr)
        sys.exit(1)
    return p


def _extract_curve(curve: dict) -> tuple[list[int], list[float], list[float] | None]:
    """
    Parse a convergence_curve dict.

    Values may be plain floats (SEED mode) or {"mean": ..., "std": ...} (AGGREGATE).
    Returns (iterations, means, stds_or_None).
    """
    iters, means, stds = [], [], []
    has_std = False

    for key, val in sorted(curve.items(), key=lambda kv: int(kv[0].split("_")[1])):
        k = int(key.split("_")[1])
        iters.append(k)
        if isinstance(val, dict):
            has_std = True
            means.append(float(val["mean"]))
            stds.append(float(val.get("std", 0.0)))
        else:
            means.append(float(val))

    return iters, means, (stds if has_std else None)


def _fmt(v: float) -> str:
    """Format probability value: drop superfluous trailing zeros."""
    s = f"{v:.3f}"
    return s.rstrip("0").rstrip(".")


# ── Core plot ─────────────────────────────────────────────────────────────────

def _annotate(ax, xs, ys, color: str, offset_y: int) -> None:
    for x, y in zip(xs, ys):
        ax.annotate(
            _fmt(y),
            xy=(x, y),
            xytext=(0, offset_y),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color=color,
        )


def build_plot(summary: dict, out_path: Path) -> None:
    b0_curve_raw = summary.get("B0", {}).get("convergence_curve", {})
    b1_curve_raw = summary.get("B1", {}).get("convergence_curve", {})

    if not b0_curve_raw or not b1_curve_raw:
        print("В файле нет данных convergence_curve для B0 и/или B1.", file=sys.stderr)
        sys.exit(1)

    iters0, means0, stds0 = _extract_curve(b0_curve_raw)
    iters1, means1, stds1 = _extract_curve(b1_curve_raw)

    fig, ax = plt.subplots(figsize=(10, 6))

    # ── B0 ──────────────────────────────────────────────────────────────────
    ax.plot(
        iters0, means0,
        marker=MARKER_B0, color=COLOR_B0,
        linewidth=2.5, markersize=9,
        label=LABEL_B0, zorder=4,
    )
    if stds0:
        lo = [m - s for m, s in zip(means0, stds0)]
        hi = [m + s for m, s in zip(means0, stds0)]
        ax.fill_between(iters0, lo, hi, color=COLOR_B0, alpha=0.14, zorder=2)

    # ── B1 ──────────────────────────────────────────────────────────────────
    ax.plot(
        iters1, means1,
        marker=MARKER_B1, color=COLOR_B1,
        linewidth=2.5, markersize=9,
        label=LABEL_B1, zorder=4,
    )
    if stds1:
        lo = [m - s for m, s in zip(means1, stds1)]
        hi = [m + s for m, s in zip(means1, stds1)]
        ax.fill_between(iters1, lo, hi, color=COLOR_B1, alpha=0.14, zorder=2)

    # ── Labels above B1, below B0 so they don't overlap ────────────────────
    _annotate(ax, iters1, means1, color=COLOR_B1, offset_y=+13)
    _annotate(ax, iters0, means0, color=COLOR_B0, offset_y=-18)

    # ── Axes & grid ─────────────────────────────────────────────────────────
    all_iters = sorted(set(iters0 + iters1))
    ax.set_xticks(all_iters)
    ax.set_xticklabels([str(k) for k in all_iters], fontsize=11)
    ax.set_xlabel("Итерация отладки", fontsize=13)
    ax.set_ylabel("P(решено за ≤ k итераций)", fontsize=13)
    ax.set_ylim(0.0, 1.08)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.grid(True, linestyle="--", alpha=0.45, zorder=1)
    ax.set_axisbelow(True)

    # ── Title ───────────────────────────────────────────────────────────────
    mode = summary.get("mode", "")
    seed = summary.get("seed")
    subtitle = f"Seed {seed}" if mode == "SEED" and seed is not None else None

    ax.set_title(
        TITLE + (f"\n({subtitle})" if subtitle else ""),
        fontsize=15,
        fontweight="bold",
        pad=12,
    )

    # ── Legend ──────────────────────────────────────────────────────────────
    ax.legend(fontsize=12, loc="lower right", framealpha=0.92, edgecolor="#cccccc")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"График сохранён: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Построить график кривых сходимости из QuixBugs summary-файла.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "summary_file",
        help="Путь к summary JSON-файлу",
    )
    args = parser.parse_args()

    src = _resolve_path(args.summary_file)

    with open(src, encoding="utf-8") as fh:
        summary = json.load(fh)

    out = src.parent / (src.stem + "_convergence.png")
    build_plot(summary, out)


if __name__ == "__main__":
    main()
