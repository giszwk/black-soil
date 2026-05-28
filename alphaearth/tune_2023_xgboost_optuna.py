#!/usr/bin/env python3
"""Tune 2023 AlphaEarth XGBoost models with feature selection and Optuna."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from string import ascii_uppercase

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "alphaearth" / "data" / "alphaearth_2023_sample_embeddings.csv"
OUT_DIR = ROOT / "alphaearth" / "outputs_yearly" / "2023" / "xgboost_optuna"
TARGETS = ["pH值", "全碳(g/kg)", "有机碳(g/kg)", "容重(g/cm3)", "N(g/kg)"]
EMBEDDING_BANDS = [f"A{i:02d}" for i in range(64)]


def safe_name(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(" ", "_")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=80)
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


def mi_score(x: pd.DataFrame, y: pd.Series) -> np.ndarray:
    return mutual_info_regression(x, y, random_state=42)


def build_pipeline(params: dict) -> Pipeline:
    k_features = params["k_features"]
    xgb_params = {key: value for key, value in params.items() if key != "k_features"}
    return Pipeline(
        steps=[
            ("select", SelectKBest(score_func=mi_score, k=k_features)),
            ("xgb", XGBRegressor(**xgb_params)),
        ]
    )


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "k_features": trial.suggest_int("k_features", 8, 64),
        "n_estimators": trial.suggest_int("n_estimators", 120, 900),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
        "subsample": trial.suggest_float("subsample", 0.55, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.3, 20.0, log=True),
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": 1,
        "tree_method": "hist",
    }


def tune_target(
    x: pd.DataFrame,
    y: pd.Series,
    target: str,
    n_trials: int,
    cv: KFold,
) -> tuple[dict, pd.DataFrame]:
    scorer = make_scorer(root_mean_squared_error, greater_is_better=False)

    def objective(trial: optuna.Trial) -> float:
        model = build_pipeline(suggest_params(trial))
        scores = cross_val_score(model, x, y, cv=cv, scoring=scorer, n_jobs=-1)
        return -float(scores.mean())

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=f"alphaearth_2023_xgboost_{target}",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    trials = study.trials_dataframe()
    trials.to_csv(OUT_DIR / f"optuna_trials_{safe_name(target)}.csv", index=False, encoding="utf-8-sig")
    return study.best_params, trials


def evaluate_best(
    x: pd.DataFrame,
    y: pd.Series,
    target: str,
    params: dict,
    cv: KFold,
) -> tuple[np.ndarray, dict, list[str]]:
    full_params = {
        **params,
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": 1,
        "tree_method": "hist",
    }
    model = build_pipeline(full_params)
    pred = cross_val_predict(model, x, y, cv=cv, n_jobs=-1)

    fitted = model.fit(x, y)
    mask = fitted.named_steps["select"].get_support()
    selected_features = [band for band, keep in zip(EMBEDDING_BANDS, mask, strict=True) if keep]
    metrics = {
        "target": target,
        "n": len(y),
        "r2": r2_score(y, pred),
        "rmse": root_mean_squared_error(y, pred),
        "mae": mean_absolute_error(y, pred),
        "k_features": len(selected_features),
    }
    return pred, metrics, selected_features


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
            f"$R^2$ = {metric_map[target]['r2']:.2f}\n"
            f"RMSE = {metric_map[target]['rmse']:.2f}\n"
            f"MAE = {metric_map[target]['mae']:.2f}\n"
            f"k = {int(metric_map[target]['k_features'])}",
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
    fig.suptitle("AlphaEarth 2023 embeddings: tuned XGBoost cross-validated regression", fontsize=10)
    fig.savefig(OUT_DIR / "observed_vs_predicted.png", dpi=600)
    fig.savefig(OUT_DIR / "observed_vs_predicted.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", message="`sklearn.utils.parallel.delayed`")

    df = pd.read_csv(INPUT, encoding="utf-8-sig").dropna(subset=[*EMBEDDING_BANDS, *TARGETS])
    x = df[EMBEDDING_BANDS]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    predictions = df[["样点", "点号", "经度", "纬度"]].copy()
    metrics = []
    selected_rows = []
    best_params = {}

    for target in TARGETS:
        y = df[target].astype(float)
        params, _ = tune_target(x, y, target, args.trials, cv)
        pred, metric, selected_features = evaluate_best(x, y, target, params, cv)
        predictions[f"{target}_observed"] = y.to_numpy()
        predictions[f"{target}_predicted"] = pred
        metrics.append(metric)
        best_params[target] = params
        selected_rows.extend({"target": target, "band": band} for band in selected_features)
        print(pd.DataFrame([metric]).to_string(index=False))

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUT_DIR / "model_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUT_DIR / "cross_validated_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selected_rows).to_csv(OUT_DIR / "selected_features.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "best_params.json").write_text(
        json.dumps(best_params, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    configure_plot_style()
    plot_observed_vs_predicted(predictions, metrics_df)


if __name__ == "__main__":
    main()
