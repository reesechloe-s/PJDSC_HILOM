"""SARIMAX model configurations and fitting."""

from time import perf_counter
import warnings

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX


# Each entry contains:
# (p, d, q), (P, D, Q, seasonal_period)
#
# Calendar and weather predictors are supplied separately.
# The final candidate tests daily residual dependence.
ORDERS = [
    ((0, 0, 0), (0, 0, 0, 0)),
    ((1, 0, 0), (0, 0, 0, 0)),
    ((0, 0, 1), (0, 0, 0, 0)),
    ((1, 0, 0), (1, 0, 0, 24)),
]


def fit_sarimax(
    y,
    x,
    order,
    seasonal_order,
    maxiter=150,
):
    """
    Fit one SARIMAX candidate.

    Parameters
    ----------
    y:
        Historical hourly arrival counts.
    x:
        Historical predictors aligned with y.
    order:
        Nonseasonal configuration (p, d, q).
    seasonal_order:
        Seasonal configuration (P, D, Q, s).

    Returns
    -------
    result:
        Fitted model.
    elapsed:
        Fitting time in seconds.
    warning_messages:
        Warnings recorded during fitting.
    """

    start = perf_counter()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        model = SARIMAX(
            endog=y.astype(float),
            exog=x,
            order=tuple(order),
            seasonal_order=tuple(seasonal_order),
            trend="n",
            enforce_stationarity=True,
            enforce_invertibility=True,
        )

        result = model.fit(
            disp=False,
            maxiter=maxiter,
        )

    converged = result.mle_retvals.get(
        "converged",
        False,
    )

    if not converged:
        raise RuntimeError(
            "SARIMAX did not converge; candidate excluded."
        )

    if not np.isfinite(result.params).all():
        raise RuntimeError(
            "Model contains non-finite parameters."
        )

    elapsed = perf_counter() - start

    warning_messages = [
        str(warning.message)
        for warning in caught
    ]

    return result, elapsed, warning_messages
