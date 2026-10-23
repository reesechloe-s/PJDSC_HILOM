"""Closed-form M/M/c (Erlang C) and M/G/c (Allen-Cunneen) helpers.

Pure Python, no third-party dependencies, so it can be unit-tested anywhere and
used as an independent cross-check on the discrete-event simulation.

Units: arrival rates are patients/hour, service times are minutes, waits are minutes.
"""
from __future__ import annotations

import math

import pandas as pd


def erlang_b(servers: int, offered_load: float) -> float:
    """Blocking probability via the numerically stable recursion."""
    if servers < 1:
        raise ValueError("servers must be >= 1")
    b = 1.0
    for k in range(1, servers + 1):
        b = offered_load * b / (k + offered_load * b)
    return b


def erlang_c(servers: int, offered_load: float) -> float:
    """P(an arriving patient has to wait) for an M/M/c queue."""
    if offered_load <= 0:
        return 0.0
    if offered_load >= servers:
        return 1.0
    rho = offered_load / servers
    b = erlang_b(servers, offered_load)
    return b / (1.0 - rho * (1.0 - b))


def mmc_metrics(
    arrival_rate_per_hr: float,
    mean_service_min: float,
    servers: int,
    service_cv2: float = 1.0,
) -> dict:
    """Steady-state metrics. service_cv2=1 is M/M/c; other values use the
    Allen-Cunneen M/G/c approximation Wq ~ Wq(M/M/c) * (1 + cv2) / 2."""
    if arrival_rate_per_hr < 0 or mean_service_min <= 0 or servers < 1:
        raise ValueError("need arrival_rate >= 0, mean_service_min > 0, servers >= 1")
    lam_per_min = arrival_rate_per_hr / 60.0
    a = lam_per_min * mean_service_min          # offered load in Erlangs
    rho = a / servers
    stable = rho < 1.0
    p_wait = erlang_c(servers, a)
    if not stable:
        wq = math.inf
    elif lam_per_min == 0:
        wq = 0.0
    else:
        mu = 1.0 / mean_service_min
        wq = p_wait / (servers * mu - lam_per_min)
        wq *= (1.0 + service_cv2) / 2.0
    return {
        "offered_load": a,
        "utilization": rho,
        "stable": stable,
        "p_wait": p_wait,
        "mean_wait_min": wq,
        "mean_queue_len": (lam_per_min * wq) if math.isfinite(wq) else math.inf,
    }


def mmc_table(
    arrivals_per_hr: pd.Series, mean_service_min: float, servers: int,
    service_cv2: float = 1.0,
) -> pd.DataFrame:
    rows = [
        mmc_metrics(float(x), mean_service_min, servers, service_cv2)
        for x in arrivals_per_hr
    ]
    return pd.DataFrame(rows, index=arrivals_per_hr.index)


def effective_arrival_rate(
    arrival_rate_per_hr: float, divertible_fraction: float, diversion_acceptance: float = 1.0
) -> float:
    """Arrival rate left at the hospital after diverting a share of the divertible class."""
    for name, v in (("divertible_fraction", divertible_fraction),
                    ("diversion_acceptance", diversion_acceptance)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    return arrival_rate_per_hr * (1.0 - divertible_fraction * diversion_acceptance)


def servers_needed(
    arrival_rate_per_hr: float, mean_service_min: float, target_wait_min: float,
    max_servers: int = 2000,
) -> int:
    """Smallest c with mean M/M/c wait <= target."""
    c = max(1, math.ceil(arrival_rate_per_hr / 60.0 * mean_service_min))
    while c <= max_servers:
        m = mmc_metrics(arrival_rate_per_hr, mean_service_min, c)
        if m["stable"] and m["mean_wait_min"] <= target_wait_min:
            return c
        c += 1
    raise ValueError("no feasible server count under max_servers")