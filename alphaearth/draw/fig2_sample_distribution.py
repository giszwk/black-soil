#!/usr/bin/env python3
"""Draw the spatial distribution of black-soil sampling points."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from shapely.geometry import Point


REPO_ROOT = Path(__file__).resolve().parents[2]
ALPHAEARTH_ROOT = Path(__file__).resolve().parents[1]
AOI_PATH = REPO_ROOT / "data" / "songnen_plain" / "songnen_wgs84.shp"
SAMPLE_CSV = REPO_ROOT / "data" / "黑土层采样数据" / "黑土层采样数据.csv"
OUTPUT_DIR = ALPHAEARTH_ROOT / "draw" / "outputs"
OUTPUT_PNG = OUTPUT_DIR / "fig2_sample_distribution.png"
OUTPUT_PDF = OUTPUT_DIR / "fig2_sample_distribution.pdf"

FIGURE_DPI = 600
FONT_SIZE_TITLE = 11
FONT_SIZE_LABEL = 9
FONT_SIZE_TICK = 8
FONT_SIZE_LEGEND = 8


def configure_style() -> None:
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font in ["PingFang HK", "Songti SC", "Arial Unicode MS", "Hiragino Sans GB", "STHeiti"]:
        if font in fonts:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "font.size": FONT_SIZE_TICK,
            "axes.labelsize": FONT_SIZE_LABEL,
            "axes.titlesize": FONT_SIZE_TITLE,
            "xtick.labelsize": FONT_SIZE_TICK,
            "ytick.labelsize": FONT_SIZE_TICK,
            "legend.fontsize": FONT_SIZE_LEGEND,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_scale_bar(ax: plt.Axes, length_km: int = 100) -> None:
    """Add an approximate scale bar for maps in EPSG:4326."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    x_start = xlim[1] - x_range * 0.28
    y_start = ylim[0] + y_range * 0.055
    length_deg = length_km / (111.0 * 0.67)
    height = y_range * 0.01
    segments = 4
    seg_width = length_deg / segments

    for idx in range(segments):
        rect = Rectangle(
            (x_start + idx * seg_width, y_start),
            seg_width,
            height,
            facecolor="black" if idx % 2 == 0 else "white",
            edgecolor="black",
            linewidth=0.6,
            zorder=5,
        )
        ax.add_patch(rect)

    ax.text(x_start, y_start - height * 0.9, "0", ha="center", va="top", fontsize=FONT_SIZE_TICK)
    ax.text(
        x_start + length_deg,
        y_start - height * 0.9,
        f"{length_km} km",
        ha="center",
        va="top",
        fontsize=FONT_SIZE_TICK,
    )


def add_north_arrow(ax: plt.Axes) -> None:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    x = xlim[1] - x_range * 0.08
    y = ylim[1] - y_range * 0.18
    length = y_range * 0.075
    half_width = x_range * 0.018

    ax.annotate(
        "",
        xy=(x, y + length),
        xytext=(x, y),
        arrowprops={"facecolor": "black", "edgecolor": "black", "width": 3, "headwidth": 12},
        zorder=6,
    )
    ax.text(x, y + length + y_range * 0.018, "N", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.plot([x - half_width, x + half_width], [y, y], color="black", linewidth=0.8, zorder=6)


def load_data() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    aoi = gpd.read_file(AOI_PATH).to_crs("EPSG:4326")
    samples = pd.read_csv(SAMPLE_CSV, encoding="utf-8-sig")
    samples = samples.dropna(subset=["经度", "纬度"]).copy()
    sample_gdf = gpd.GeoDataFrame(
        samples,
        geometry=[Point(lon, lat) for lon, lat in zip(samples["经度"], samples["纬度"], strict=True)],
        crs="EPSG:4326",
    )
    aoi_union = aoi.geometry.union_all()
    sample_gdf = sample_gdf[sample_gdf.geometry.within(aoi_union)].copy()
    return aoi, sample_gdf


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    aoi, samples = load_data()
    bounds = aoi.total_bounds
    pad_x = (bounds[2] - bounds[0]) * 0.06
    pad_y = (bounds[3] - bounds[1]) * 0.06

    fig, ax = plt.subplots(figsize=(6.2, 7.0), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F7F7F7")

    aoi.plot(ax=ax, facecolor="#F1F4EA", edgecolor="#7A7A7A", linewidth=0.45, zorder=1)
    aoi.boundary.plot(ax=ax, edgecolor="#2E2E2E", linewidth=1.0, zorder=2)
    samples.plot(
        ax=ax,
        color="#D55E00",
        markersize=22,
        edgecolor="white",
        linewidth=0.45,
        alpha=0.9,
        zorder=3,
    )

    ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
    ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("黑土层采样点空间分布", fontweight="bold", pad=8)
    ax.grid(True, color="white", linewidth=0.8, zorder=0)
    ax.tick_params(direction="out", length=3, width=0.8)

    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#333333")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#D55E00",
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=6,
            label=f"Sampling points (n={len(samples)})",
        ),
        Line2D([0], [0], color="#2E2E2E", linewidth=1.0, label="Songnen Plain boundary"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", frameon=True, framealpha=0.95, edgecolor="#CCCCCC")

    add_scale_bar(ax, length_km=100)
    add_north_arrow(ax)

    fig.savefig(OUTPUT_PNG, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
