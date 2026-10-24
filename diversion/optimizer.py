"""Diversion Load-Balancing Optimizer for Emergency Department Decision Support."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from simulation.engine import SimulationResult


@dataclass
class OptimizationSummary:
    """Structured summary of optimization results for LLM policy reporting."""

    total_excess_patients: int
    total_allocated_patients: int
    unhandled_overflow: int
    facility_allocations: pd.DataFrame
    hourly_allocation_log: pd.DataFrame
    llm_context_str: str


def optimize_diversion_allocations(
    simulation_result: SimulationResult,
    facilities_df: pd.DataFrame,
    max_occupancy_threshold: float = 0.85,
    provider_threshold: float = 2.0,  # CHANGED: Added missing parameter
) -> OptimizationSummary:
    """
    Solves a Linear Program to load-balance excess non-urgent patients across
    receiving facilities, minimizing total patient travel time while respecting
    facility capacities.
    """
    if facilities_df.empty or simulation_result.hourly.empty:
        empty_alloc = facilities_df.copy() if not facilities_df.empty else pd.DataFrame()
        if not empty_alloc.empty:
            empty_alloc["recommended_referrals"] = 0
        return OptimizationSummary(
            total_excess_patients=0,
            total_allocated_patients=0,
            unhandled_overflow=0,
            facility_allocations=empty_alloc,
            hourly_allocation_log=pd.DataFrame(),
            llm_context_str="No simulation or facility data provided for optimization.",
        )

    hourly = simulation_result.hourly.copy()
    patients = simulation_result.patients.copy()

    # CHANGED: Composite threshold logic for saturation
    hourly["utilization"] = hourly["mean_bed_utilization"].fillna(0.0)
    
    # Safely fetch provider queue if it exists in the hourly logs, otherwise default to 0
    # Update "mean_doctor_queue_length" if your engine.py uses a different column name
    provider_queue = hourly.get("mean_doctor_queue_length", pd.Series([0] * len(hourly))) 
    
    # Identify hours where EITHER bed utilization OR provider queue exceeded the thresholds
    saturated_hours = hourly[
        (hourly["utilization"] >= max_occupancy_threshold) | 
        (provider_queue >= provider_threshold)
    ]

    # Calculate non-urgent excess demand during saturated hours
    hourly_log_list: list[dict[str, Any]] = []
    facility_totals = {
        fac_id: 0 for fac_id in facilities_df["facility_id"].values
    }

    travel_costs = facilities_df["travel_time_mins"].to_numpy(dtype=float)
    capacities = facilities_df["max_simultaneous_patients"].to_numpy(dtype=float)
    facility_ids = facilities_df["facility_id"].values
    facility_names = facilities_df["facility_name"].values

    total_excess = 0
    total_allocated = 0

    for _, hour_row in saturated_hours.iterrows():
        timestamp = hour_row["timestamp"]
        hour_idx = int(hour_row["timestamp"].hour) if hasattr(hour_row["timestamp"], "hour") else 0

        # Non-urgent patients arriving during this saturated hour
        if not patients.empty:
            p_hour = patients[
                (patients["arrival_time"].floordiv(60) == hour_idx)
                & (patients["acuity"] == "non-urgent")
            ]
            excess_count = len(p_hour)
        else:
            excess_count = int(hour_row.get("non_urgent_count", 0))

        if excess_count <= 0 or sum(capacities) <= 0:
            continue

        total_excess += excess_count

        # Linear Program Setup:
        # Objective: Minimize c^T * x (Total Travel Time)
        # Subject to: sum(x) <= excess_count, 0 <= x_i <= capacity_i
        c = travel_costs
        bounds = [(0, cap) for cap in capacities]
        A_eq = [np.ones(len(c))]
        b_eq = [min(excess_count, sum(capacities))]

        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

        if res.success:
            allocations = np.round(res.x).astype(int)
        else:
            allocations = np.zeros(len(c), dtype=int)

        allocated_this_hour = int(sum(allocations))
        total_allocated += allocated_this_hour

        for fac_id, alloc in zip(facility_ids, allocations):
            facility_totals[fac_id] += int(alloc)

        hourly_log_list.append(
            {
                "timestamp": timestamp,
                "bed_utilization_pct": round(float(hour_row["utilization"]) * 100, 1),
                "excess_non_urgent_patients": excess_count,
                "allocated_to_network": allocated_this_hour,
                "unhandled_overflow": excess_count - allocated_this_hour,
            }
        )

    # Build facility summary dataframe
    facility_summary = facilities_df.copy()
    facility_summary["recommended_referrals"] = facility_summary["facility_id"].map(
        facility_totals
    ).fillna(0).astype(int)

    unhandled = total_excess - total_allocated

    # Construct the structured LLM prompt context string
    llm_lines = [
        "=== DIVERSION LOAD-BALANCING OPTIMIZATION SUMMARY ===",
        f"- Saturated Hours Identified (Beds >= {max_occupancy_threshold * 100:.0f}% or Providers >= {provider_threshold}): {len(saturated_hours)} hours",
        f"- Total Excess Non-Urgent Patients Requiring Redirection: {total_excess}",
        f"- Total Patients Successfully Allocated to Network: {total_allocated}",
        f"- Unhandled Overflow (Network Capacity Deficit): {unhandled}",
        "",
        "FACILITY ALLOCATION BREAKDOWN:",
    ]

    for _, f_row in facility_summary.iterrows():
        llm_lines.append(
            f"  * {f_row['facility_name']} (Travel: {f_row['travel_time_mins']} min, Max Cap: {f_row['max_simultaneous_patients']}): "
            f"{f_row['recommended_referrals']} referrals recommended"
        )

    llm_context_str = "\n".join(llm_lines)

    return OptimizationSummary(
        total_excess_patients=total_excess,
        total_allocated_patients=total_allocated,
        unhandled_overflow=unhandled,
        facility_allocations=facility_summary,
        hourly_allocation_log=pd.DataFrame(hourly_log_list),
        llm_context_str=llm_context_str,
    )