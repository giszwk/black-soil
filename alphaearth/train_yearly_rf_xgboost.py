#!/usr/bin/env python3
"""Train yearly RF and XGBoost regressors from AlphaEarth point embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path
from string import ascii_uppercase

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "alphaearth" / "data"
OUT_ROOT = ROOT / "alphaearth" / "outputs_yearly"
TARGETS = ["pH值", "全碳(g/kg)", "有机碳(g/kg)", "容重(g/cm3)", "N(g/kg)"]
EMBEDDING_BANDS = [f"A{i:02d}" for i in range(64)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, required=True)
    return parser.parse_args()


def configure_plot_style() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    preferred_fonts = [
        "PingFang HK",
        "Songti SC",
        "Arial Unicode MS",
        "Hiragino Sans GB",
        "STHeiti",
    ]
    for font in preferred_fonts:
        if font in available_fonts:
            sns.set_theme(style="ticks", context="paper", font=font)
            break
    else:
        sns.set_theme(style="ticks", context="paper")

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def build_rf() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )


def build_xgboost() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=2.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )


def evaluate_model(
    df: pd.DataFrame,
    year: int,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_df = df.dropna(subset=[*EMBEDDING_BANDS, *TARGETS]).copy()
    x = model_df[EMBEDDING_BANDS]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    predictions = model_df[["样点", "点号", "经度", "纬度"]].copy()
    metrics = []

    for target in TARGETS:
        y = model_df[target].astype(float)
        model = build_rf() if model_name == "rf" else build_xgboost()
        pred = cross_val_predict(model, x, y, cv=cv, n_jobs=-1)
        predictions[f"{target}_observed"] = y.to_numpy()
        predictions[f"{target}_predicted"] = pred
        metrics.append(
            {
                "year": year,
                "model": model_name,
                "target": target,
                "n": len(y),
                "r2": r2_score(y, pred),
                "rmse": root_mean_squared_error(y, pred),
                "mae": mean_absolute_error(y, pred),
            }
        )

    return pd.DataFrame(metrics), predictions


def plot_observed_vs_predicted(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    year: int,
    model_name: str,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), constrained_layout=True)
    axes = axes.ravel()
    metric_map = metrics.set_index("target").to_dict(orient="index")

    for idx, (ax, target) in enumerate(zip(axes, TARGETS, strict=False)):
        obs = predictions[f"{target}_observed"]
        pred = predictions[f"{target}_predicted"]
        low = min(obs.min(), pred.min())
        high = max(obs.max(), pred.max())
        pad = (high - low) * 0.08 if high > low else 1
        low -= pad
        high += pad
        slope, intercept = np.polyfit(obs, pred, deg=1)

        ax.scatter(
            obs,
            pred,
            s=18,
            facecolor="#0072B2",
            edgecolor="white",
            linewidth=0.35,
            alpha=0.78,
            rasterized=True,
        )
        ax.plot(
            [low, high],
            [low, high],
            color="#4D4D4D",
            linewidth=0.9,
            linestyle=(0, (4, 3)),
            label="1:1 line",
        )
        ax.plot(
            [low, high],
            [slope * low + intercept, slope * high + intercept],
            color="#D55E00",
            linewidth=1.1,
            label="OLS fit",
        )
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(target)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        ax.text(
            -0.14,
            1.08,
            ascii_uppercase[idx],
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            fontweight="bold",
        )
        ax.text(
            0.05,
            0.95,
            f"$R^2$ = {metric_map[target]['r2']:.2f}\n"
            f"RMSE = {metric_map[target]['rmse']:.2f}\n"
            f"MAE = {metric_map[target]['mae']:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7.5,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#BDBDBD",
                "linewidth": 0.5,
                "alpha": 0.9,
            },
        )
        ax.tick_params(direction="out", length=3, width=0.8)
        ax.grid(True, color="#E5E5E5", linewidth=0.55)
        sns.despine(ax=ax)

    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="center", frameon=False, title="Reference")
    title_model = "Random forest" if model_name == "rf" else "XGBoost"
    fig.suptitle(f"AlphaEarth {year} embeddings: {title_model} cross-validated regression", fontsize=10)
    fig.savefig(out_dir / "observed_vs_predicted.png", dpi=600)
    fig.savefig(out_dir / "observed_vs_predicted.pdf")
    plt.close(fig)


def run_year(year: int, model_name: str) -> pd.DataFrame:
    input_csv = DATA_DIR / f"alphaearth_{year}_sample_embeddings.csv"
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    out_dir = OUT_ROOT / str(year) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    metrics, predictions = evaluate_model(df, year, model_name)
    metrics.to_csv(out_dir / "model_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out_dir / "cross_validated_predictions.csv", index=False, encoding="utf-8-sig")
    plot_observed_vs_predicted(predictions, metrics, year, model_name, out_dir)
    return metrics


def main() -> None:
    configure_plot_style()
    all_metrics = []
    for year in parse_args().years:
        for model_name in ["rf", "xgboost"]:
            metrics = run_year(year, model_name)
            all_metrics.append(metrics)
            print(metrics.to_string(index=False))

    summary = pd.concat(all_metrics, ignore_index=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_ROOT / "yearly_model_metrics.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
