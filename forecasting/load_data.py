"""Load hourly data and prepare forecasting features."""

from pathlib import Path

import numpy as np
import pandas as pd


def load_data(path):
    """Read the dataset and validate its hourly observations."""

    path = Path(path)

    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)

    required = {
        "timestamp",
        "total_arrivals",
        "precipitation_mm",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.empty:
        raise ValueError("The dataset is empty.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if df["timestamp"].isna().any():
        raise ValueError("Missing timestamps.")

    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps.")

    df = df.sort_values("timestamp").set_index("timestamp")
    df.index = df.index.as_unit("ns")

    if df.index.tz is not None:
        df.index = (
            df.index
            .tz_convert("Asia/Manila")
            .tz_localize(None)
        )

    expected_index = pd.date_range(
        start=df.index[0],
        periods=len(df),
        freq="h",
    )

    if not df.index.equals(expected_index):
        raise ValueError(
            "Timestamps must form an uninterrupted hourly series."
        )

    for column in ["total_arrivals", "precipitation_mm"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

        if not np.isfinite(df[column]).all():
            raise ValueError(
                f"{column} contains missing or non-finite values."
            )

        if (df[column] < 0).any():
            raise ValueError(
                f"{column} contains negative values."
            )

    if not np.equal(
        df["total_arrivals"],
        np.floor(df["total_arrivals"]),
    ).all():
        raise ValueError(
            "Observed arrivals must be integer counts."
        )

    if not (df["precipitation_mm"] > 0).any():
        raise ValueError(
            "All rainfall is zero. Check whether the generator "
            "used its API-failure fallback."
        )

    return df.asfreq("h")


def rain_threshold(training):
    """Calculate the heavy-rain threshold from training data."""

    wet_hours = training.loc[
        training["precipitation_mm"] >= 1,
        "precipitation_mm",
    ]

    if wet_hours.empty:
        raise ValueError(
            "No wet training hours: cannot estimate "
            "the weather experiment."
        )

    return float(wet_hours.quantile(0.98))


def features(df, threshold, mode="calendar"):
    """
    Construct predictors without using patient-arrival targets.

    Modes:
        calendar: hour-of-day and day-of-week indicators.
        weather: calendar plus precipitation and heavy rain.
        weather_lag: weather plus a post-rain indicator.

    Include the preceding two historical hours when preparing
    future weather features so rainfall lags cross the boundary.
    """

    valid_modes = {
        "calendar",
        "weather",
        "weather_lag",
    }

    if mode not in valid_modes:
        raise ValueError(f"Unknown feature mode: {mode}")

    index = df.index

    x = pd.DataFrame(
        {"constant": np.ones(len(df))},
        index=index,
    )

    # Hour 0 is the reference category.
    for hour in range(1, 24):
        x[f"hour_{hour}"] = (
            index.hour == hour
        ).astype(float)

    # Monday (0) is the reference category.
    for day in range(1, 7):
        x[f"dow_{day}"] = (
            index.dayofweek == day
        ).astype(float)

    if mode != "calendar":
        heavy = df["precipitation_mm"].ge(threshold)

        x["precipitation_mm"] = (
            df["precipitation_mm"].astype(float)
        )

        x["is_heavy_rain"] = heavy.astype(float)

        if mode == "weather_lag":
            recent_heavy_rain = (
                heavy.shift(1, fill_value=False)
                | heavy.shift(2, fill_value=False)
            )

            x["post_rain_surge"] = (
                (~heavy) & recent_heavy_rain
            ).astype(float)

    return x.astype(float)
