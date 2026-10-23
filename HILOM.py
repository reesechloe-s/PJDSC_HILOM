import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from simulation.config import DEFAULT_CONFIG, SimulationConfig
from simulation.engine import SimulationResult, run_simulation
from facility_map import render_facility_map
from kpis import compare_scenarios, hourly_dashboard_metrics, scenario_kpis
from memo import build_memo_facts, generate_memo
from queueing_theory import mmc_table
from memo import build_memo_facts, generate_memo

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


def _get_api_key():
    """Key from the environment or Streamlit secrets; None means 'use the template memo'."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        return None


def _slider_default(value, lo, hi):
    """Clamp a config default into the slider range (Streamlit raises if it is outside)."""
    return min(max(value, lo), hi)


@st.cache_data
def run_dashboard_simulations(
    forecast_df: pd.DataFrame,
    facilities: pd.DataFrame,
    beds: int,
    nurses: int,
    doctors: int,
    max_occupancy: float,
    wait_target: float,
    max_wait_tolerance: float,
    triage_min: float,
    triage_max: float,
    treatment_mean: float,
) -> tuple[SimulationResult, SimulationResult]:
    
    baseline_config = SimulationConfig(
        num_beds=beds,
        num_nurses=nurses,
        num_doctors=doctors,
        max_occupancy_before_diversion=None,
        wait_target_minutes=wait_target,
        max_wait_tolerance=max_wait_tolerance,
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
        max_occupancy_before_diversion=max_occupancy,
        wait_target_minutes=wait_target,
        max_wait_tolerance=max_wait_tolerance,
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
trigger_pct = st.sidebar.slider(
    "Redirection Trigger (% of capacity reached)", 
    min_value=50, 
    max_value=100, 
    value=_slider_default(
        int(round((DEFAULT_CONFIG.max_occupancy_before_diversion or 0.85) * 100)), 50, 100
    ),
)
max_occupancy = trigger_pct / 100.0

wait_target = st.sidebar.slider(
    "Wait-time target (minutes)", 
    min_value=0, 
    max_value=120, 
    value=_slider_default(int(DEFAULT_CONFIG.wait_target_minutes), 0, 120),
)
max_wait_tolerance = st.sidebar.slider(
    "Walkout time limit (minutes)", 
    min_value=30, 
    max_value=1440, 
    value=_slider_default(int(DEFAULT_CONFIG.max_wait_tolerance), 30, 1440),
)

# Service Times
triage_min = st.sidebar.slider(
    "Triage min time (mins)", 
    min_value=1.0, 
    max_value=10.0, 
    value=_slider_default(float(DEFAULT_CONFIG.triage_min_minutes), 1.0, 10.0),
)
triage_max = st.sidebar.slider(
    "Triage max time (mins)", 
    min_value=float(triage_min), 
    max_value=30.0, 
    value=_slider_default(float(DEFAULT_CONFIG.triage_max_minutes), float(triage_min), 30.0),
)
treatment_mean = st.sidebar.slider(
    "Mean treatment time (minutes)", 
    min_value=1,  # 0 would make service times degenerate
    max_value=240, 
    value=_slider_default(int(DEFAULT_CONFIG.treatment_mean_minutes), 1, 240),
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
    max_occupancy,
    wait_target,
    max_wait_tolerance,
    triage_min,
    triage_max,
    treatment_mean,
)

baseline_df = hourly_dashboard_metrics(baseline_result.hourly, baseline_result.patients, max_occupancy)
sim_df = hourly_dashboard_metrics(diversion_result.hourly, diversion_result.patients, max_occupancy)
facility_results = edited_facilities.merge(
    diversion_result.facilities[
        ["facility_id", "referrals_accepted"]
    ],
    on="facility_id",
    how="left",
)
facility_results["referrals_accepted"] = (
    facility_results["referrals_accepted"].fillna(0).astype(int)
)

baseline_kpis = scenario_kpis(baseline_df, baseline_result.patients)
optimized_kpis = scenario_kpis(sim_df, diversion_result.patients)
kpi_table = compare_scenarios(baseline_kpis, optimized_kpis)

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

    st.markdown("**Proposal targets: baseline vs. diversion (same seed, same demand)**")
    st.dataframe(
        kpi_table.style.format({"Result": "{:.2f}", "Target": "{:.2f}"}, na_rep="n/a"),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Reductions are fractions (0.10 = 10%). Capacity violation = hour with all beds "
        "occupied. One seed is one sample path; treat PASS/FAIL as indicative."
    )

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
    utilization_fig.add_hline(y=trigger_pct, line_dash="dash")
    utilization_fig.update_layout(
        xaxis_title="Time", yaxis_title="Sampled bed utilization (%)"
    )
    st.plotly_chart(utilization_fig, width="stretch")

    with st.expander("Queueing-theory cross-check (analytic M/M/c on beds)"):
        analytic = mmc_table(
            forecast_df["forecast"].clip(lower=0), float(treatment_mean), int(beds)
        )
        peak_i = analytic["utilization"].idxmax()
        a1, a2, a3 = st.columns(3)
        a1.metric("Peak offered load (Erlangs)", f"{analytic['offered_load'].max():.1f}")
        a2.metric("Peak analytic utilization", f"{analytic['utilization'].max():.0%}")
        a3.metric(
            "Peak analytic mean wait",
            "unstable" if not analytic.loc[peak_i, "stable"]
            else f"{analytic.loc[peak_i, 'mean_wait_min']:.1f} min",
        )
        st.caption(
            "Steady-state M/M/c using forecast arrivals, mean treatment time and bed count. "
            "Ignores triage, acuity mix and diversion, so it should bracket, not equal, the DES."
        )

with tab3:
    st.subheader("Simulated diversion")
    st.metric(
        "Peak non-urgent diversion",
        f"{sim_df['divert_pct'].max():.0f}% of non-urgent patients",
    )
    patients_log = diversion_result.patients
    diverted_total = (
        int(patients_log["diversion_accepted"].sum())
        if "diversion_accepted" in patients_log.columns else 0
    )
    st.metric("Patients diverted", f"{diverted_total}")
    st.write("Alternative facilities and accepted referrals")
    render_facility_map(facility_results)
    
    with st.expander("View Raw Patient Simulation Logs"):
        st.dataframe(patients_log, width="stretch")

with tab4:
    st.subheader("Draft policy memo")
    st.caption(
        "Drafted only from the simulation figures in this dashboard. "
        "A human must review before any use."
    )
    diverted_total = (
        int(diversion_result.patients["diversion_accepted"].sum())
        if "diversion_accepted" in diversion_result.patients.columns else 0
    )
    memo_facts = build_memo_facts(
        params={
            "beds": beds, "nurses": nurses, "doctors": doctors,
            "trigger_pct": trigger_pct, "wait_target_min": wait_target,
            "walkout_limit_min": max_wait_tolerance, "mean_treatment_min": treatment_mean,
        },
        kpi_table=kpi_table,
        baseline_kpis=baseline_kpis,
        optimized_kpis=optimized_kpis,
        sim_df=sim_df,
        facility_results=facility_results,
        diverted_total=diverted_total,
        forecast_df=forecast_df,
    )
    facts_sig = hashlib.sha256(
        json.dumps(memo_facts, sort_keys=True, default=str).encode()
    ).hexdigest()

    if st.button("Generate memo"):
        with st.spinner("Drafting memo..."):
            result = generate_memo(memo_facts, api_key=_get_api_key())
        st.session_state["memo_text"] = result.text
        st.session_state["memo_source"] = result.source
        st.session_state["memo_note"] = result.note
        st.session_state["memo_unverified"] = result.unverified_numbers
        st.session_state["memo_sig"] = facts_sig

    st.session_state.setdefault("memo_text", "")
    if st.session_state.get("memo_source") == "template":
        st.info(st.session_state.get("memo_note") or "Template draft.")
    if st.session_state.get("memo_unverified"):
        st.warning(
            "Numbers in this draft that are not in the simulation results: "
            + ", ".join(st.session_state["memo_unverified"])
            + ". Check them before use."
        )
    if st.session_state.get("memo_sig") not in (None, facts_sig):
        st.warning("Settings or results changed since this memo was drafted. Regenerate it.")
    st.text_area("Memo draft (for human review)", key="memo_text", height=420)