import os

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="Armillaria Risk GeoAI", layout="wide")

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 0rem !important;
        }
        header {
            visibility: hidden !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

BACKEND_HOST = os.getenv("BACKEND_HOST", "http://localhost:8000")
PREDICT_URL = f"{BACKEND_HOST}/predict"
FEEDBACK_URL = f"{BACKEND_HOST}/feedback"

if "map_center" not in st.session_state:
    st.session_state.map_center = [47.68, 16.59]
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 11
if "target_lat" not in st.session_state:
    st.session_state.target_lat = 47.68
if "target_lon" not in st.session_state:
    st.session_state.target_lon = 16.59
if "inference_results" not in st.session_state:
    st.session_state.inference_results = None

st.title("Autonomous GeoAI Inference & Explainability")
st.markdown(
    "Select a location on the map. The system extracts STAC data, predicts risk via XGBoost, "
    "and explains the prediction using SHAP."
)


def plot_risk_gauge(probability):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 32}},
            title={"text": "Infection Risk", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "rgba(0,0,0,0)"},
                "steps": [
                    {"range": [0, 33], "color": "#2ecc71"},
                    {"range": [33, 66], "color": "#f1c40f"},
                    {"range": [66, 100], "color": "#e74c3c"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": probability * 100,
                },
            },
        )
    )
    fig.update_layout(height=180, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def plot_shap_bar(shap_dict):
    df = pd.DataFrame(list(shap_dict.items()), columns=["Feature", "SHAP Value"])
    df["Abs_SHAP"] = df["SHAP Value"].abs()
    df = df.sort_values(by="Abs_SHAP", ascending=True)
    df["Impact"] = df["SHAP Value"].apply(lambda x: "Increases Risk" if x > 0 else "Decreases Risk")

    fig = px.bar(
        df,
        x="SHAP Value",
        y="Feature",
        orientation="h",
        color="Impact",
        color_discrete_map={"Increases Risk": "#ff4b4b", "Decreases Risk": "#0068c9"},
    )
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        xaxis_title="SHAP Value",
        yaxis_title="",
    )
    return fig


top_col1, top_col2 = st.columns([2, 1])

with top_col1:
    m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom)

    folium.TileLayer("OpenStreetMap", name="Topography").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite (Esri)",
        overlay=False,
    ).add_to(m)
    folium.LayerControl().add_to(m)

    folium.Marker(
        [st.session_state.target_lat, st.session_state.target_lon],
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

    map_data = st_folium(m, height=380, use_container_width=True)

    if map_data:
        if map_data.get("center"):
            st.session_state.map_center = [map_data["center"]["lat"], map_data["center"]["lng"]]
        if map_data.get("zoom"):
            st.session_state.map_zoom = map_data["zoom"]

        if map_data.get("last_clicked"):
            new_lat = map_data["last_clicked"]["lat"]
            new_lon = map_data["last_clicked"]["lng"]

            if new_lat != st.session_state.target_lat or new_lon != st.session_state.target_lon:
                st.session_state.target_lat = new_lat
                st.session_state.target_lon = new_lon
                st.rerun()

    with st.form("field_form"):
        st.markdown("#### Field Validation")
        tree_status = st.selectbox("Observed Status", ["Healthy (0)", "Infected (1)"])
        notes = st.text_input("Field Notes (e.g., Pinus sylvestris roots)")

        if st.form_submit_button("Submit Ground Truth"):
            status_val = 1 if "1" in tree_status else 0
            payload = {
                "lat": st.session_state.target_lat,
                "lon": st.session_state.target_lon,
                "status": status_val,
                "notes": notes,
            }
            try:
                res = requests.post(FEEDBACK_URL, json=payload)
                if res.status_code == 200:
                    st.success("Feedback saved to database!")
                else:
                    st.error(f"Failed to submit: {res.text}")
            except Exception as ex:
                st.error(f"Cannot reach backend: {ex}")

with top_col2:
    st.subheader("Spatial Analysis")
    st.info(
        f"Target: Lat {st.session_state.target_lat:.4f} | Lon {st.session_state.target_lon:.4f}"
    )

    if st.button("Run Spatial Inference", type="primary", use_container_width=True):
        with st.status("Initiating GeoAI Pipeline...", expanded=True) as status:
            try:
                st.write("Fetching STAC & TerraClimate data...")
                response = requests.post(
                    PREDICT_URL,
                    json={"lat": st.session_state.target_lat, "lon": st.session_state.target_lon},
                )
                response.raise_for_status()

                st.session_state.inference_results = response.json()
                status.update(label="Inference Complete!", state="complete", expanded=False)

            except requests.exceptions.RequestException as e:
                status.update(label="Inference Failed", state="error", expanded=True)
                st.error(f"API Error: {e}")

if st.session_state.inference_results:
    st.markdown("---")
    res = st.session_state.inference_results

    bottom_col1, bottom_col2, bottom_col3 = st.columns([1, 2, 1])

    with bottom_col1:
        st.markdown("#### Overall Risk")
        st.plotly_chart(plot_risk_gauge(res.get("risk_probability", 0.0)), use_container_width=True)

    with bottom_col2:
        st.markdown("#### Model Explainability (SHAP)")
        st.plotly_chart(plot_shap_bar(res.get("shap_values", {})), use_container_width=True)

    with bottom_col3:
        feats = res.get("extracted_features", {})
        st.markdown("#### Environmental Features")
        st.write("")
        c1, c2 = st.columns(2)
        c1.metric("Elevation", f"{feats.get('elevation_m', 0):.1f} m")
        c1.metric("BIO1 Mean Temp", f"{feats.get('bio1_temp_c', 0):.2f} °C")
        c2.metric("Landcover", f"Class {feats.get('landcover_class', 0)}")
        c2.metric("BIO12 Annual Precip", f"{feats.get('bio12_precip_mm', 0):.1f} mm")
