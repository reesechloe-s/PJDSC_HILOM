"""Generate and validate dashboard-compatible forecasts."""

from time import perf_counter

import numpy as np
import pandas as pd

from .load_data import features


def validate_forecast(df, expected_rows=168):
    """Check forecast length, timestamps, values, and bounds."""

    required = {
        "timestamp",
        "forecast",
        "lower",
        "upper",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing forecast columns: {missing}")

    if df.empty:
        raise ValueError("The forecast is empty.")

    if len(df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} forecast rows; "
            f"received {len(df)}."
        )

    timestamps = pd.DatetimeIndex(
        pd.to_datetime(df["timestamp"])
    ).as_unit("ns")

    expected_index = pd.date_range(
        start=timestamps[0],
        periods=len(timestamps),
        freq="h",
    )

    if not timestamps.equals(expected_index):
        raise ValueError(
            "Forecast timestamps must be consecutive hours."
        )

    values = df[["forecast", "lower", "upper"]]

    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(
            "Forecast contains missing or non-finite values."
        )

    if (values < 0).any().any():
        raise ValueError(
            "Forecast values must be nonnegative."
        )

    valid_bounds = (
        (df["lower"] <= df["forecast"])
        & (df["forecast"] <= df["upper"])
    )

    if not valid_bounds.all():
        raise ValueError(
            "Prediction bounds do not contain "
            "the point forecast."
        )


def forecast_frame(result, future_x):
    """
    Generate expected hourly arrivals and nominal 95% bounds.

    Negative predictions and bounds are clipped at zero.
    The same rule must be used during evaluation.
    """

    start = perf_counter()

    prediction = result.get_forecast(
        steps=len(future_x),
        exog=future_x,
    )

    mean = np.asarray(
        prediction.predicted_mean
    )

    interval = np.asarray(
        prediction.conf_int(alpha=0.05)
    )

    output = pd.DataFrame({
        "timestamp": future_x.index,
        "forecast": np.maximum(0, mean),
        "lower": np.maximum(0, interval[:, 0]),
        "upper": np.maximum(0, interval[:, 1]),
    })

    validate_forecast(
        output,
        expected_rows=len(future_x),
    )

    elapsed = perf_counter() - start

    return output, elapsed


def predict_future(
    result,
    history,
    future_weather,
    threshold,
    mode,
):
    """
    Forecast the 168 hours immediately after history.

    The fitted model must end at the same cutoff as history.

    Supply future precipitation explicitly:
    observed weather for historical evaluation, or
    specified weather inputs for a future scenario.

    Timestamps must use local Manila time.
    """

    future = future_weather.copy()

    if "timestamp" in future.columns:
        future = future.set_index(
            pd.to_datetime(future.pop("timestamp"))
        )

    future.index = pd.DatetimeIndex(
        future.index
    ).as_unit("ns")

    expected_index = pd.date_range(
        start=history.index[-1] + pd.Timedelta(hours=1),
        periods=168,
        freq="h",
    )

    if not future.index.equals(expected_index):
        raise ValueError(
            "Supply 168 consecutive hours immediately "
            "after the training history."
        )

    rain = future["precipitation_mm"]

    if not np.isfinite(rain).all() or (rain < 0).any():
        raise ValueError(
            "Future precipitation must be finite "
            "and nonnegative."
        )

    # Preserve the preceding two hours for weather lags.
    combined_weather = pd.concat([
        history[["precipitation_mm"]].tail(2),
        future[["precipitation_mm"]],
    ])

    future_x = features(
        combined_weather,
        threshold=threshold,
        mode=mode,
    ).iloc[-168:]

    return forecast_frame(
        result,
        future_x,
    )
