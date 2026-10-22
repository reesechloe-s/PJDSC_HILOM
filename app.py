import numpy as np                       
import pandas as pd                      # DataFrames (tables)
import streamlit as st           

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
FACILITIES_PATH = PROJECT_DIR / "data" / "alternative_facilities.parquet"
        # dashboard lib

# page setup: title, layout, etc.
st.set_page_config(page_title="ED Load Balancer", layout="wide")  
st.title("Emergency Department Load Balancer")                    
st.caption("Decision-support tool. Demo data only.")              


@st.cache_data
def load_facilities():
    return pd.read_parquet(FACILITIES_PATH)

@st.cache_data
def make_mock_forecast():
    # 7 days of hourly timestamps starting from the current hour
    start = pd.Timestamp.now().floor("h")
    times = pd.date_range(start, periods=7 * 24, freq="h")
    hours = np.arange(len(times))

    # fake arrivals: a daily wave (peak mid-day) + small random noise
    rng = np.random.default_rng(42)  # fixed seed so the fake data doesn't change
    forecast = 20 + 8 * np.sin(2 * np.pi * (hours % 24 - 6) / 24) + rng.normal(0, 1.5, len(times))
    forecast = np.clip(forecast, 1, None)  # arrivals can't be negative
    
    # same shape as Kath's future output: timestamp, forecast, lower, upper
    return pd.DataFrame({
        "timestamp": times,
        "forecast": forecast,
        "lower": forecast * 0.85,   # fake lower bound of the forecast range
        "upper": forecast * 1.15,   # fake upper bound
    })


def run_simulation(forecast_df, beds, staff, diversion_threshold):
    # FAKE stand-in for Chloe's real function. Same inputs/outputs idea.
    df = forecast_df.copy()                                # don't change the original table
    capacity = min(staff * 2.0, beds / 2.0)                # patients per hour the ED can handle
    df["utilization"] = df["forecast"] / capacity          # 1.0 means completely full
    df["wait_min"] = 5 + 90 * np.clip(df["utilization"] - 0.7, 0, None)  # waits grow once busy
    df["saturated"] = df["utilization"] >= diversion_threshold            # True when over the limit
    # share of patients to divert so utilization drops back to the threshold
    df["divert_pct"] = np.where(df["saturated"], 1 - diversion_threshold / df["utilization"], 0) * 100
    return df


# sidebar controls
st.sidebar.header("Controls")                                                     # sidebar heading
beds = st.sidebar.slider("Bed capacity", 10, 200, 60)                             # returns a number
staff = st.sidebar.slider("Staff on duty", 5, 100, 25)                            # returns a number
threshold_pct = st.sidebar.slider("Diversion threshold (% of capacity)", 50, 100, 85)
wait_target = st.sidebar.slider("Wait-time target (minutes)", 15, 120, 45)
diversion_threshold = threshold_pct / 100                                         # 85 -> 0.85

# run the models (reruns automatically when a slider moves)
forecast_df = make_mock_forecast()                                      # Kath's part (fake for now)
sim_df = run_simulation(forecast_df, beds, staff, diversion_threshold)  # Chloe's part (fake for now)
facilities = load_facilities()                                          # the parquet file or fake backup

# tab layouts
tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "Queue Simulation", "Diversion", "Policy Memo"])

# tab1: forecast chart
with tab1:
    st.subheader("Projected hourly arrivals (7 days)")
    fig = go.Figure()                                                      # empty chart
    fig.add_trace(go.Scatter(x=sim_df["timestamp"], y=sim_df["upper"],
                             line=dict(width=0), showlegend=False))        # invisible top edge of the band
    fig.add_trace(go.Scatter(x=sim_df["timestamp"], y=sim_df["lower"],
                             fill="tonexty", line=dict(width=0),
                             name="Forecast range"))                       # shaded band down to the lower edge
    fig.add_trace(go.Scatter(x=sim_df["timestamp"], y=sim_df["forecast"],
                             name="Forecast"))                             # the main forecast line
    fig.update_layout(xaxis_title="Time", yaxis_title="Patients per hour") # axis labels
    st.plotly_chart(fig, use_container_width=True)                         # show the chart

# tab2: queue sim results
with tab2:
    st.subheader("Waiting time and utilization")
    peak_wait = sim_df["wait_min"].max()                               # worst wait in the week
    peak_util = sim_df["utilization"].max() * 100                      # busiest hour, as a %
    sat_hours = int(sim_df["saturated"].sum())                         # count of saturated hours
    c1, c2, c3 = st.columns(3)                                         # three side-by-side boxes
    c1.metric("Peak wait", f"{peak_wait:.0f} min")                     # big number cards
    c2.metric("Peak utilization", f"{peak_util:.0f}%")
    c3.metric("Saturated hours", sat_hours)
    if peak_wait > wait_target:                                        # compare with the slider target
        st.warning(f"Projected wait exceeds the {wait_target}-minute target.")
    else:
        st.success("Projected waits stay within target.")
    wait_fig = go.Figure(go.Scatter(x=sim_df["timestamp"], y=sim_df["wait_min"], name="Wait"))
    wait_fig.add_hline(y=wait_target, line_dash="dash")                # dashed target line
    wait_fig.update_layout(xaxis_title="Time", yaxis_title="Wait (minutes)")
    st.plotly_chart(wait_fig, use_container_width=True)

# tab3: diversion and map 
with tab3:
    st.subheader("Recommended diversion")
    st.metric("Peak diversion needed", f"{sim_df['divert_pct'].max():.0f}% of non-critical patients")
    st.write("Alternative facilities")
    st.map(facilities, latitude="lat", longitude="lon")                # plots the points on a map
    st.dataframe(facilities, use_container_width=True)                 # same data as a table

# tab4: policy memo placeholder
with tab4:
    st.subheader("Draft policy memo")
    st.info("Justin's LLM output will appear here.")
    memo = st.text_area("Memo draft (for human review)", "Placeholder memo text...", height=200)
    st.button("Generate memo")                                         # does nothing yet