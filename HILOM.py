from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from simulation.config import DEFAULT_CONFIG, SimulationConfig
from simulation.engine3 import SimulationResult, run_simulation
from facility_map import render_facility_map

PROJECT_DIR = Path(__file__).resolve().parent
FACILITIES_PATH = PROJECT_DIR / "data" / "alternative_facilities.parquet"
FORECAST_PATH = PROJECT_DIR / "artifacts" / "forecasting" / "forecast.parquet"
RANDOM_SEED = 42

st.set_page_config(page_title="ED Load Balancer", layout="wide")
st.title("Emergency Department Load Balancer")
st.caption("Decision-support tool using SARIMAX forecast data.")


@st.cache_data
def load_facilities() -> pd.DataFrame:
    return pd.read_parquet(FACILITIES_PATH)


@st.cache_data
def load_forecast() -> pd.DataFrame:
    df = pd.read_parquet(FORECAST_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _dashboard_metrics(
    result: SimulationResult, diversion_threshold: float
) -> pd.DataFrame:
    hourly = result.hourly.copy()
    hourly["utilization"] = hourly["mean_bed_utilization"].fillna(0.0)
    hourly["wait_min"] = hourly["mean_total_wait"].fillna(0.0)
    hourly["saturated"] = hourly["utilization"] >= diversion_threshold
    hourly["divert_pct"] = 0.0

    if not result.patients.empty:
        patients = result.patients.copy()
        patients["_hour"] = patients["arrival_time"].floordiv(60).astype(int)
        nonurgent = patients.loc[patients["acuity"] == "non-urgent"]
        grouped = nonurgent.groupby("_hour")
        nonurgent_count = grouped.size()
        diverted_count = grouped["diversion_accepted"].sum()
        for hour in nonurgent_count.index:
            if hour < len(hourly) and nonurgent_count.loc[hour] > 0:
                hourly.loc[hour, "divert_pct"] = (
                    100.0 * diverted_count.get(hour, 0) / nonurgent_count.loc[hour]
                )

    return hourly


@st.cache_data
def run_dashboard_simulations(
    forecast_df: pd.DataFrame,
    facilities: pd.DataFrame,
    beds: int,
    nurses: int,
    doctors: int,
    diversion_threshold: float,
    wait_target: float,
    abandonment_limit: float,
    triage_min: float,
    triage_max: float,
    treatment_mean: float,
) -> tuple[SimulationResult, SimulationResult]:
    
    baseline_config = SimulationConfig(
        num_beds=beds,
        num_nurses=nurses,
        num_doctors=doctors,
        diversion_occupancy_threshold=None,
        wait_target_minutes=wait_target,
        abandonment_limit_minutes=abandonment_limit,
        triage_min_minutes=triage_min,
        triage_max_minutes=triage_max,
        treatment_mean_minutes=treatment_mean,
        acuity_probabilities=DEFAULT_CONFIG.acuity_probabilities,
        treatment_std_dev_minutes=DEFAULT_CONFIG.treatment_std_dev_minutes,
    )
    diversion_config = SimulationConfig(
        num_beds=beds,
        num_nurses=nurses,
        num_doctors=doctors,
        diversion_occupancy_threshold=diversion_threshold,
        wait_target_minutes=wait_target,
        abandonment_limit_minutes=abandonment_limit,
        triage_min_minutes=triage_min,
        triage_max_minutes=triage_max,
        treatment_mean_minutes=treatment_mean,
        acuity_probabilities=baseline_config.acuity_probabilities,
        treatment_std_dev_minutes=baseline_config.treatment_std_dev_minutes,
    )

    baseline = run_simulation(
        forecast_df, config=baseline_config, facilities=facilities, seed=RANDOM_SEED
    )
    diversion = run_simulation(
        forecast_df, config=diversion_config, facilities=facilities, seed=RANDOM_SEED
    )
    return baseline, diversion


st.sidebar.header("Controls")

# Resource Capacities
beds = st.sidebar.slider(
    "Bed capacity", min_value=1, max_value=400, value=DEFAULT_CONFIG.num_beds
)
nurses = st.sidebar.slider(
    "Triage nurses on duty", min_value=1, max_value=200, value=DEFAULT_CONFIG.num_nurses
)
doctors = st.sidebar.slider(
    "Doctors on duty", min_value=1, max_value=200, value=DEFAULT_CONFIG.num_doctors
)

# Thresholds & Limits
threshold_pct = st.sidebar.slider(
    "Diversion threshold (% of capacity)", 
    min_value=50, 
    max_value=100, 
    value=int((DEFAULT_CONFIG.diversion_occupancy_threshold or 0.85) * 100)
)
diversion_threshold = threshold_pct / 100.0

wait_target = st.sidebar.slider(
    "Wait-time target (minutes)", 
    min_value=0, 
    max_value=120, 
    value=int(DEFAULT_CONFIG.wait_target_minutes)
)
abandonment_limit = st.sidebar.slider(
    "Abandonment limit (minutes)", 
    min_value=30, 
    max_value=1440, 
    value=int(DEFAULT_CONFIG.abandonment_limit_minutes)
)

# Service Times
triage_min = st.sidebar.slider(
    "Triage min time (mins)", 
    min_value=1.0, 
    max_value=10.0, 
    value=float(DEFAULT_CONFIG.triage_min_minutes)
)
triage_max = st.sidebar.slider(
    "Triage max time (mins)", 
    min_value=float(triage_min), 
    max_value=30.0, 
    value=float(DEFAULT_CONFIG.triage_max_minutes)
)
treatment_mean = st.sidebar.slider(
    "Mean treatment time (minutes)", 
    min_value=0, 
    max_value=240, 
    value=int(DEFAULT_CONFIG.treatment_mean_minutes)
)

forecast_df = load_forecast()
facilities_raw = load_facilities()

# Editable Facilities Dataframe
st.subheader("Network Facility Capacities")
st.write("Edit the simultaneous capacity of nearby facilities. Set to 0 to disable diversion to a specific hospital.")
edited_facilities = st.data_editor(
    facilities_raw,
    column_config={
        "simultaneous_capacity": st.column_config.NumberColumn(
            "Capacity (Beds)",
            help="Maximum diverted patients they can handle at once",
            min_value=0,
            max_value=100,
            step=1,
        )
    },
    disabled=["facility_id", "facility_name", "travel_time_mins"],
    hide_index=True,
)

# Run simulations using the EDITED facilities and new slider parameters
baseline_result, diversion_result = run_dashboard_simulations(
    forecast_df,
    edited_facilities,
    beds,
    nurses,
    doctors,
    diversion_threshold,
    wait_target,
    abandonment_limit,
    triage_min,
    triage_max,
    treatment_mean,
)

baseline_df = _dashboard_metrics(baseline_result, diversion_threshold)
sim_df = _dashboard_metrics(diversion_result, diversion_threshold)
facility_results = edited_facilities.merge(
    diversion_result.facilities[
        ["facility_id", "referrals_accepted"]
    ],
    on="facility_id",
    how="left",
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Forecast", "Queue Simulation", "Diversion", "Policy Memo"]
)

with tab1:
    st.subheader("Projected hourly arrivals (7 days)")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=forecast_df["timestamp"],
            y=forecast_df["upper"],
            line=dict(width=0),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["timestamp"],
            y=forecast_df["lower"],
            fill="tonexty",
            line=dict(width=0),
            name="Forecast range",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["timestamp"],
            y=forecast_df["forecast"],
            name="Forecast",
        )
    )
    fig.update_layout(xaxis_title="Time", yaxis_title="Patients per hour")
    st.plotly_chart(fig, width="stretch")

with tab2:
    st.subheader("DES waiting time and bed utilization")
    peak_wait = sim_df["wait_min"].max()
    peak_util = sim_df["utilization"].max() * 100
    sat_hours = int(sim_df["saturated"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Peak mean wait", f"{peak_wait:.0f} min")
    c2.metric("Peak sampled bed utilization", f"{peak_util:.0f}%")
    c3.metric("Hours at diversion threshold", sat_hours)

    if peak_wait > wait_target:
        st.warning(f"Projected mean waits exceed the {wait_target}-minute target.")
    else:
        st.success("Projected mean waits stay within target.")

    wait_fig = go.Figure()
    wait_fig.add_trace(
        go.Scatter(
            x=baseline_df["timestamp"],
            y=baseline_df["wait_min"],
            name="Baseline (no diversion)",
        )
    )
    wait_fig.add_trace(
        go.Scatter(
            x=sim_df["timestamp"],
            y=sim_df["wait_min"],
            name="DES with diversion",
        )
    )
    wait_fig.add_hline(y=wait_target, line_dash="dash")
    wait_fig.update_layout(xaxis_title="Time", yaxis_title="Mean wait (minutes)")
    st.plotly_chart(wait_fig, width="stretch")

    utilization_fig = go.Figure(
        go.Scatter(
            x=sim_df["timestamp"],
            y=sim_df["utilization"] * 100,
            name="Bed utilization",
        )
    )
    utilization_fig.add_hline(y=threshold_pct, line_dash="dash")
    utilization_fig.update_layout(
        xaxis_title="Time", yaxis_title="Sampled bed utilization (%)"
    )
    st.plotly_chart(utilization_fig, width="stretch")

with tab3:
    st.subheader("Simulated diversion")
    st.metric(
        "Peak non-urgent diversion",
        f"{sim_df['divert_pct'].max():.0f}% of non-urgent patients",
    )
    st.metric(
        "Patients diverted",
        f"{int(diversion_result.patients['diversion_accepted'].sum())}",
    )
    st.write("Alternative facilities and accepted referrals")
    render_facility_map(facility_results)
    
    with st.expander("View Raw Patient Simulation Logs"):
        st.dataframe(diversion_result.patients, use_container_width=True)

with tab4:
    st.subheader("Draft policy memo")
    st.info("Justin's LLM output will appear here.")
    memo = st.text_area(
        "Memo draft (for human review)", "Placeholder memo text...", height=200
    )
    st.button("Generate memo")