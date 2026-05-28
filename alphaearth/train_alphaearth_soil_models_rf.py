#!/usr/bin/env python3
"""Train soil property regressors from local AlphaEarth embedding samples."""

from __future__ import annotations

from pathlib import Path
from string import ascii_uppercase

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "alphaearth" / "data"
OUT_DIR = ROOT / "alphaearth" / "outputs"
INPUT = DATA_DIR / "alphaearth_2018_sample_embeddings.csv"
TARGETS = ["pH值", "全碳(g/kg)", "有机碳(g/kg)", "容重(g/cm3)", "N(g/kg)"]
EMBEDDING_BANDS = [f"A{i:02d}" for i in range(64)]


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
            return
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams["axes.unicode_minus"] = False


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=500,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_and_evaluate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [col for col in [*EMBEDDING_BANDS, *TARGETS] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    model_df = df.dropna(subset=[*EMBEDDING_BANDS, *TARGETS]).copy()
    x = model_df[EMBEDDING_BANDS]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    metrics = []
    predictions = model_df[["样点", "点号", "经度", "纬度"]].copy()
    importances = []

    for target in TARGETS:
        y = model_df[target].astype(float)
        model = build_model()
        pred = cross_val_predict(model, x, y, cv=cv, n_jobs=-1)
        predictions[f"{target}_observed"] = y.to_numpy()
        predictions[f"{target}_predicted"] = pred
        metrics.append(
            {
                "target": target,
                "n": len(y),
                "r2": r2_score(y, pred),
                "rmse": root_mean_squared_error(y, pred),
                "mae": mean_absolute_error(y, pred),
            }
        )

        fitted = build_model().fit(x, y)
        result = permutation_importance(
            fitted,
            x,
            y,
            n_repeats=20,
            random_state=42,
            n_jobs=-1,
            scoring="neg_root_mean_squared_error",
        )
        top_idx = np.argsort(result.importances_mean)[-10:][::-1]
        for idx in top_idx:
            importances.append(
                {
                    "target": target,
                    "band": EMBEDDING_BANDS[idx],
                    "importance": result.importances_mean[idx],
                }
            )

    return pd.DataFrame(metrics), predictions, pd.DataFrame(importances)


def plot_observed_vs_predicted(predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
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
            f"$R^2$ = {metric_map[target]['r2']:.2f}\nRMSE = {metric_map[target]['rmse']:.2f}\nMAE = {metric_map[target]['mae']:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7.5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BDBDBD", "linewidth": 0.5, "alpha": 0.9},
        )
        ax.tick_params(direction="out", length=3, width=0.8)
        ax.grid(True, color="#E5E5E5", linewidth=0.55)
        sns.despine(ax=ax)

    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="center", frameon=False, title="Reference")
    fig.suptitle("AlphaEarth 2018 embeddings: random forest cross-validated regression", fontsize=10)
    fig.savefig(OUT_DIR / "observed_vs_predicted.png", dpi=600)
    fig.savefig(OUT_DIR / "observed_vs_predicted.pdf")
    plt.close(fig)


def plot_metrics(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    sns.barplot(data=metrics, x="target", y="r2", ax=axes[0], color="#2a9d8f")
    sns.barplot(data=metrics, x="target", y="rmse", ax=axes[1], color="#e76f51")
    axes[0].set_ylim(min(-0.2, metrics["r2"].min() - 0.05), 1)
    axes[0].set_ylabel("Cross-validated R2")
    axes[1].set_ylabel("Cross-validated RMSE")
    for ax in axes:
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)
    fig.savefig(OUT_DIR / "model_metrics.png", dpi=220)
    plt.close(fig)


def plot_importance(importances: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    top = importances.sort_values(["target", "importance"], ascending=[True, False])
    sns.barplot(data=top, y="band", x="importance", hue="target", ax=ax)
    ax.set_title("Top AlphaEarth Band Permutation Importance")
    ax.set_xlabel("Increase in RMSE after permutation")
    ax.set_ylabel("Embedding band")
    fig.savefig(OUT_DIR / "band_importance.png", dpi=220)
    plt.close(fig)


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing {INPUT}. Run alphaearth/download_alphaearth_2018.py first."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT, encoding="utf-8-sig")
    metrics, predictions, importances = train_and_evaluate(df)

    metrics.to_csv(OUT_DIR / "model_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUT_DIR / "cross_validated_predictions.csv", index=False, encoding="utf-8-sig")
    importances.to_csv(OUT_DIR / "band_importance.csv", index=False, encoding="utf-8-sig")

    configure_plot_style()
    plot_observed_vs_predicted(predictions, metrics)
    plot_metrics(metrics)
    plot_importance(importances)

    print(metrics.to_string(index=False))
    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
