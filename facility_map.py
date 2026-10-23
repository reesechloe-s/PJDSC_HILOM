import pydeck as pdk
import streamlit as st


def render_facility_map(facilities):
    df = facilities.dropna(subset=["lat", "lon"])  # skip rows with no coordinates
    if df.empty:
        st.error("No facilities with coordinates to show.")
        return

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[lon, lat]",
        get_radius=150,
        get_fill_color=[220, 50, 50, 200],
        pickable=True,  # lets the hover tooltip work
    )
    view = pdk.ViewState(
        latitude=df["lat"].mean(), longitude=df["lon"].mean(), zoom=12
    )

    text = "{facility_name}\n{facility_type}\n{travel_time_mins} min away"
    if "referrals_accepted" in df.columns:  # only from the simulation merge
        text += "\n{referrals_accepted} referrals accepted"
    tooltip = {"text": text}

    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip))  # type: ignore
    st.dataframe(df, width="stretch")