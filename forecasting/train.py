"""Run from the repository root with python -m forecasting.train."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .load_data import load_data, rain_threshold, features
from .sarimax import ORDERS, fit_sarimax
from .predict import forecast_frame


def metrics(y, p):
    """Calculate prediction errors, explicitly handling zero actuals."""

    y = np.asarray(y)
    p = np.asarray(p)

    error = np.abs(y - p)
    positive = y > 0

    return {
        "n": len(y),
        "mae": float(error.mean()),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "wape_pct": (
            float(100 * error.sum() / y.sum())
            if y.sum()
            else np.nan
        ),
        "mape_positive_hours_pct": (
            float(
                100 * np.mean(
                    error[positive] / y[positive]
                )
            )
            if positive.any()
            else np.nan
        ),
        "zero_actual_hours": int((~positive).sum()),
    }


def origins(start, stop, folds):
    """Choose non-overlapping seven-day evaluation windows."""

    available = list(range(start, stop - 168 + 1, 168))

    if len(available) < folds:
        raise ValueError(
            "Not enough complete weeks for the requested folds."
        )

    positions = np.unique(
        np.linspace(
            0,
            len(available) - 1,
            folds,
            dtype=int,
        )
    )

    return [available[i] for i in positions]


def run(
    data_path,
    output="artifacts/forecasting",
    folds=3,
    train_days=90,
    quick=False,
    validation_start=None,
    test_start=None,
):
    """Train candidates, evaluate forecasts, and export results."""

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)

    # Use supplied dates, or chronological percentage boundaries.
    if validation_start:
        a = int(
            df.index.searchsorted(
                pd.Timestamp(validation_start)
            )
        )
    else:
        a = int(len(df) * 0.70)

    if test_start:
        b = int(
            df.index.searchsorted(
                pd.Timestamp(test_start)
            )
        )
    else:
        b = int(len(df) * 0.85)

    if not 168 < a < b < len(df):
        raise ValueError(
            "Invalid chronological split boundaries."
        )

    val_origins = origins(a, b, folds)
    test_origins = origins(b, len(df), folds)

    # Freeze the threshold before validation and testing.
    threshold = rain_threshold(df.iloc[:a])

    modes = ["calendar", "weather", "weather_lag"]
    orders = ORDERS[:1] if quick else ORDERS

    specs = []

    for mode in modes:
        for i, (order, seasonal) in enumerate(orders):
            specs.append({
                "name": f"sarimax_{mode}_{i}",
                "kind": "sarimax",
                "mode": mode,
                "order": order,
                "seasonal_order": seasonal,
            })

    specs.extend([
        {
            "name": "weekly_naive",
            "kind": "naive",
            "mode": "calendar",
        },
        {
            "name": "hour_of_week_mean",
            "kind": "mean",
            "mode": "calendar",
        },
        {
            "name": "gradient_boosted_weather_lag",
            "kind": "boost",
            "mode": "weather_lag",
        },
    ])

    # Features do not use future arrival counts.
    # Weather evaluation is conditional on observed future rain.
    xs = {
        mode: features(df, threshold, mode)
        for mode in modes
    }

    event_flags = features(
        df,
        threshold,
        "weather_lag",
    )["post_rain_surge"]

    records = []
    failures = []
    predictions = []

    def evaluate(spec, origin, phase):
        """Fit using preceding observations and forecast 168 hours."""

        lo = max(0, origin - train_days * 24)

        y = df["total_arrivals"].iloc[lo:origin]
        truth = df["total_arrivals"].iloc[
            origin:origin + 168
        ]

        x = xs[spec["mode"]]

        fit_seconds = 0.0
        prediction_seconds = 0.0
        warnings_list = []
        result = None

        start = perf_counter()

        if spec["kind"] == "sarimax":
            result, fit_seconds, warnings_list = fit_sarimax(
                y,
                x.iloc[lo:origin],
                spec["order"],
                spec["seasonal_order"],
            )

            frame, prediction_seconds = forecast_frame(
                result,
                x.iloc[origin:origin + 168],
            )

        else:
            if spec["kind"] == "naive":
                # Repeat the preceding week's hourly observations.
                pred = y.iloc[-168:].to_numpy()

            elif spec["kind"] == "mean":
                # Average historical arrivals for each hour of week.
                hour_of_week = (
                    y.index.dayofweek * 24
                    + y.index.hour
                )

                averages = y.groupby(hour_of_week).mean()

                future_hour_of_week = (
                    truth.index.dayofweek * 24
                    + truth.index.hour
                )

                pred = averages.reindex(
                    future_hour_of_week
                ).to_numpy()

            else:
                # Fixed comparison model, not an exhaustive search.
                model = HistGradientBoostingRegressor(
                    loss="poisson",
                    max_iter=150,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    early_stopping=False,
                    random_state=42,
                )

                model.fit(
                    x.iloc[lo:origin],
                    y,
                )

                fit_seconds = perf_counter() - start
                start = perf_counter()

                pred = model.predict(
                    x.iloc[origin:origin + 168]
                )

            prediction_seconds = perf_counter() - start

            frame = pd.DataFrame({
                "timestamp": truth.index,
                "forecast": np.maximum(0, pred),
            })

        frame["actual"] = truth.to_numpy()

        frame["post_rain_surge"] = event_flags.iloc[
            origin:origin + 168
        ].to_numpy()

        frame["lead_hour"] = np.arange(1, 169)
        frame["origin"] = truth.index[0]
        frame["model"] = spec["name"]
        frame["phase"] = phase

        for horizon in [24, 72, 168]:
            sub = frame.iloc[:horizon]

            row = {
                "phase": phase,
                "model": spec["name"],
                "origin": str(truth.index[0]),
                "horizon": horizon,
                "fit_seconds": fit_seconds,
                "prediction_seconds": prediction_seconds,
                "warnings": " | ".join(warnings_list),
                **metrics(
                    sub["actual"],
                    sub["forecast"],
                ),
            }

            if "lower" in sub.columns:
                inside = (
                    (sub["actual"] >= sub["lower"])
                    & (sub["actual"] <= sub["upper"])
                )

                row["coverage_95_pct"] = (
                    100 * inside.mean()
                )

                row["mean_interval_width"] = (
                    sub["upper"] - sub["lower"]
                ).mean()

            records.append(row)

        predictions.append(frame)

        return result, frame

    # PHASE 1: Validation and model selection.
    for spec in specs:
        print(
            "Validation:",
            spec["name"],
            flush=True,
        )

        for origin in val_origins:
            try:
                evaluate(spec, origin, "validation")

            except Exception as exc:
                failures.append({
                    "phase": "validation",
                    "model": spec["name"],
                    "origin": str(df.index[origin]),
                    "error": str(exc),
                })

    if not records:
        raise RuntimeError("All candidates failed.")

    val = pd.DataFrame(records)

    rank = (
        val[val["horizon"] == 168]
        .groupby("model")
        .agg(
            mae=("mae", "mean"),
            folds=("origin", "nunique"),
        )
        .reset_index()
    )

    # Only select candidates that completed every validation fold.
    rank = (
        rank[rank["folds"] == folds]
        .sort_values(["mae", "model"])
    )

    rank.to_csv(
        out / "validation_ranking.csv",
        index=False,
    )

    winners = []

    for mode in modes:
        names = {
            spec["name"]
            for spec in specs
            if (
                spec["kind"] == "sarimax"
                and spec["mode"] == mode
            )
        }

        eligible = rank[
            rank["model"].isin(names)
        ]

        if not eligible.empty:
            winners.append(
                eligible.iloc[0]["model"]
            )

    if not winners:
        pd.DataFrame(failures).to_csv(
            out / "failures.csv",
            index=False,
        )

        raise RuntimeError(
            "No SARIMAX candidate completed every "
            "validation fold. Inspect failures.csv."
        )

    selected_name = (
        rank[rank["model"].isin(winners)]
        .iloc[0]["model"]
    )

    test_names = winners + [
        "weekly_naive",
        "hour_of_week_mean",
        "gradient_boosted_weather_lag",
    ]

    # PHASE 2: Test evaluation.
    # Model selection is already frozen.
    demo = None

    for spec in specs:
        if spec["name"] not in test_names:
            continue

        print(
            "Test:",
            spec["name"],
            flush=True,
        )

        for origin in test_origins:
            try:
                result, frame = evaluate(
                    spec,
                    origin,
                    "test",
                )

                if (
                    spec["name"] == selected_name
                    and origin == test_origins[-1]
                ):
                    demo = frame

                    result.save(
                        out / "sarimax_historical_demo.pkl"
                    )

            except Exception as exc:
                failures.append({
                    "phase": "test",
                    "model": spec["name"],
                    "origin": str(df.index[origin]),
                    "error": str(exc),
                })

    # Save fold-level results and any failures.
    pd.DataFrame(records).to_csv(
        out / "fold_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        failures,
        columns=["phase", "model", "origin", "error"],
    ).to_csv(
        out / "failures.csv",
        index=False,
    )

    pred_df = pd.concat(
        predictions,
        ignore_index=True,
    )

    pred_df.to_csv(
        out / "backtest_predictions.csv",
        index=False,
    )

    # Aggregate results separately for each forecast horizon.
    summary = []

    for (phase, name), group in pred_df.groupby(
        ["phase", "model"]
    ):
        for horizon in [24, 72, 168]:
            sub = group[
                group["lead_hour"] <= horizon
            ]

            completed_folds = sub["origin"].nunique()

            row = {
                "phase": phase,
                "model": name,
                "horizon": horizon,
                "folds": completed_folds,
                "complete": completed_folds == folds,
                **metrics(
                    sub["actual"],
                    sub["forecast"],
                ),
            }

            row["mae_target_met"] = row["mae"] <= 2
            row["rmse_target_met"] = row["rmse"] <= 3
            row["wape_target_met"] = row["wape_pct"] < 15

            # Ordinary MAPE is undefined when actual arrivals are zero.
            if row["zero_actual_hours"] == 0:
                row["mape_target_met"] = (
                    row["mape_positive_hours_pct"] < 15
                )
            else:
                row["mape_target_met"] = None

            if "lower" in sub.columns:
                valid = sub.dropna(
                    subset=["lower", "upper"]
                )

                if len(valid):
                    inside = (
                        (valid["actual"] >= valid["lower"])
                        & (valid["actual"] <= valid["upper"])
                    )

                    row["coverage_95_pct"] = (
                        100 * inside.mean()
                    )

            summary.append(row)

    pd.DataFrame(summary).to_csv(
        out / "metrics_summary.csv",
        index=False,
    )

    # Examine performance specifically during post-rain hours.
    events = []

    for (phase, name), group in pred_df.groupby(
        ["phase", "model"]
    ):
        sub = group[
            group["post_rain_surge"] == 1
        ]

        if len(sub):
            events.append({
                "phase": phase,
                "model": name,
                **metrics(
                    sub["actual"],
                    sub["forecast"],
                ),
            })

    pd.DataFrame(events).to_csv(
        out / "post_rain_metrics.csv",
        index=False,
    )

    selected_spec = next(
        spec
        for spec in specs
        if spec["name"] == selected_name
    )

    metadata = {
        "selected_sarimax": selected_name,
        "selected_by": (
            "Mean validation 168-hour MAE; "
            "complete folds only"
        ),
        "spec": selected_spec,
        "validation_start": str(df.index[a]),
        "test_start": str(df.index[b]),
        "validation_origins": [
            str(df.index[i])
            for i in val_origins
        ],
        "test_origins": [
            str(df.index[i])
            for i in test_origins
        ],
        "rain_threshold_mm": threshold,
        "training_window_days": train_days,
        "quick_mode": quick,
        "data_sha256": hashlib.sha256(
            Path(data_path).read_bytes()
        ).hexdigest(),
        "weather_mode": (
            "Conditional backtest using "
            "observed future precipitation"
        ),
        "forecast_mode": (
            "Historical demonstration, not a live forecast"
        ),
        "timezone": (
            "Asia/Manila, stored as naive local "
            "hourly timestamps"
        ),
        "interval": (
            "Nominal 95% SARIMAX predictive bounds "
            "clipped at zero; weather uncertainty excluded"
        ),
        "data_provenance": (
            "User-supplied data; original HILOM generator "
            "produces synthetic patient counts"
        ),
        "python": platform.python_version(),
        "failures": len(failures),
    }

    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    if demo is None:
        raise RuntimeError(
            "Selected model failed at final test origin. "
            "Metrics saved; no demo export."
        )

    forecast_columns = [
        "timestamp",
        "forecast",
        "lower",
        "upper",
    ]

    demo[forecast_columns].to_parquet(
        out / "forecast.parquet",
        index=False,
    )

    demo[forecast_columns].to_csv(
        out / "forecast.csv",
        index=False,
    )

    # Save an actual-versus-predicted chart.
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(
        demo["timestamp"],
        demo["actual"],
        label="Actual synthetic arrivals",
        alpha=0.6,
    )

    ax.plot(
        demo["timestamp"],
        demo["forecast"],
        label="SARIMAX expected arrivals",
    )

    ax.fill_between(
        demo["timestamp"],
        demo["lower"],
        demo["upper"],
        alpha=0.18,
        label="Nominal 95% interval",
    )

    ax.set(
        ylabel="Patients/hour",
        title=(
            "Historical holdout: "
            "conditional on observed weather"
        ),
    )

    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(
        out / "forecast_evaluation.png",
        dpi=180,
    )

    plt.close(fig)

    print(
        "Completed. Review metrics_summary.csv, "
        "failures.csv and metadata.json "
        "before reporting results."
    )

    return pd.DataFrame(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
    )

    parser.add_argument(
        "--output",
        default="artifacts/forecasting",
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--train-days",
        type=int,
        default=90,
    )

    parser.add_argument("--validation-start")
    parser.add_argument("--test-start")
    parser.add_argument("--quick", action="store_true")

    args = parser.parse_args()

    run(
        data_path=args.data,
        output=args.output,
        folds=args.folds,
        train_days=args.train_days,
        quick=args.quick,
        validation_start=args.validation_start,
        test_start=args.test_start,
    )
