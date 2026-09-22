import os

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Armillaria Risk GeoAI",
    page_icon="🌲",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.2rem !important;
            padding-bottom: 2rem !important;
        }
        header {
            visibility: hidden !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.4rem !important;
            font-weight: 600 !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.85rem !important;
            color: #94a3b8 !important;
        }
        .risk-banner {
            padding: 8px 14px;
            border-radius: 8px;
            font-weight: 600;
            text-align: center;
            margin-top: 4px;
            margin-bottom: 8px;
        }
        .risk-high {
            background-color: rgba(231, 76, 60, 0.15);
            color: #e74c3c;
            border: 1px solid rgba(231, 76, 60, 0.4);
        }
        .risk-medium {
            background-color: rgba(241, 196, 15, 0.15);
            color: #f1c40f;
            border: 1px solid rgba(241, 196, 15, 0.4);
        }
        .risk-low {
            background-color: rgba(46, 204, 113, 0.15);
            color: #2ecc71;
            border: 1px solid rgba(46, 204, 113, 0.4);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

BACKEND_HOST = os.getenv("BACKEND_HOST", "http://localhost:8000")
PREDICT_URL = f"{BACKEND_HOST}/predict"
FEEDBACK_URL = f"{BACKEND_HOST}/feedback"

ESA_LANDCOVER_MAP = {
    10: "Tree Cover / Forest",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up / Urban",
    60: "Bare / Sparse",
    70: "Snow and Ice",
    80: "Permanent Water",
    90: "Herbaceous Wetland",
    95: "Mangroves",
    100: "Moss and Lichen",
}

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
    "Select a target location on the map to query Copernicus DEM, ESA WorldCover, and TerraClimate Zarr stores, "
    "run an XGBoost spatial inference, and interpret the outcome via SHAP."
)


def plot_risk_gauge(probability: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 34, "family": "Inter, sans-serif"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
                "bar": {"color": "#3b82f6", "thickness": 0.25},
                "steps": [
                    {"range": [0, 33], "color": "rgba(46, 204, 113, 0.45)"},
                    {"range": [33, 66], "color": "rgba(241, 196, 15, 0.45)"},
                    {"range": [66, 100], "color": "rgba(231, 76, 60, 0.45)"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 3},
                    "thickness": 0.8,
                    "value": probability * 100,
                },
            },
        )
    )
    fig.update_layout(
        height=210,
        margin=dict(l=15, r=15, t=20, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
    )
    return fig


def plot_shap_bar(shap_dict: dict):
    friendly_names = {
        "bio1_mean_temp": "Mean Annual Temp (BIO1)",
        "bio1_temp_c": "Mean Annual Temp (BIO1)",
        "bio4_temp_seasonality": "Temp Seasonality (BIO4)",
        "bio12_annual_precip": "Annual Precip (BIO12)",
        "bio12_precip_mm": "Annual Precip (BIO12)",
        "bio15_precip_seasonality": "Precip Seasonality (BIO15)",
        "elevation": "Elevation (DEM)",
        "elevation_m": "Elevation (DEM)",
        "landcover_class": "Landcover (WorldCover)",
        "slope_deg": "Terrain Slope",
    }

    df = pd.DataFrame(
        [
            {
                "Feature": friendly_names.get(k, k),
                "SHAP Value": v,
            }
            for k, v in shap_dict.items()
        ]
    )
    df["Abs_SHAP"] = df["SHAP Value"].abs()
    df = df.sort_values(by="Abs_SHAP", ascending=True)
    df["Impact"] = df["SHAP Value"].apply(lambda x: "Increases Risk" if x > 0 else "Decreases Risk")

    fig = px.bar(
        df,
        x="SHAP Value",
        y="Feature",
        orientation="h",
        color="Impact",
        color_discrete_map={"Increases Risk": "#ef4444", "Decreases Risk": "#3b82f6"},
    )
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        xaxis_title="SHAP Attribution (Log-odds impact)",
        yaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.15)", zerolinecolor="#64748b")
    fig.update_yaxes(showgrid=False)
    return fig


top_col1, top_col2 = st.columns([2, 1], gap="medium")

with top_col1:
    m = folium.Map(
        location=st.session_state.map_center,
        zoom_start=st.session_state.map_zoom,
        tiles=None,
    )

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite (Esri)",
        overlay=False,
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Topography (OSM)",
        overlay=False,
        control=True,
    ).add_to(m)

    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    folium.Marker(
        [st.session_state.target_lat, st.session_state.target_lon],
        popup=f"Target: {st.session_state.target_lat:.4f}, {st.session_state.target_lon:.4f}",
        tooltip="Selected point for inference",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

    map_data = st_folium(
        m,
        height=420,
        use_container_width=True,
        returned_objects=["last_clicked", "center", "zoom"],
        key="armillaria_leaflet_map",
    )

    if map_data and map_data.get("last_clicked"):
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lon = map_data["last_clicked"]["lng"]

        if (
            abs(clicked_lat - st.session_state.target_lat) > 1e-5
            or abs(clicked_lon - st.session_state.target_lon) > 1e-5
        ):
            st.session_state.target_lat = clicked_lat
            st.session_state.target_lon = clicked_lon

            if map_data.get("center"):
                st.session_state.map_center = [
                    map_data["center"]["lat"],
                    map_data["center"]["lng"],
                ]
            if map_data.get("zoom"):
                st.session_state.map_zoom = map_data["zoom"]

            st.rerun()

with top_col2:
    st.subheader("Spatial Analysis")
    st.info(
        f"Selected Point Coordinates:\n\n**Latitude:** `{st.session_state.target_lat:.5f}`\n\n"
        f"**Longitude:** `{st.session_state.target_lon:.5f}`"
    )

    if st.button("Run Spatial Inference", type="primary", use_container_width=True):
        with st.status("Initiating GeoAI Pipeline...", expanded=True) as status:
            try:
                st.write("Querying STAC assets & streaming TerraClimate Zarr...")
                response = requests.post(
                    PREDICT_URL,
                    json={"lat": st.session_state.target_lat, "lon": st.session_state.target_lon},
                    timeout=180,
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
    prob = float(res.get("risk_probability", 0.0))

    bottom_col1, bottom_col2, bottom_col3 = st.columns([1.1, 1.8, 1.4], gap="medium")

    with bottom_col1:
        st.markdown("### Infection Risk")
        st.plotly_chart(
            plot_risk_gauge(prob),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        if prob >= 0.66:
            st.markdown(
                '<div class="risk-banner risk-high">⚠️ High Pathogen Suitability</div>',
                unsafe_allow_html=True,
            )
        elif prob >= 0.33:
            st.markdown(
                '<div class="risk-banner risk-medium">⚡ Moderate Infection Risk</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="risk-banner risk-low">Unfavorable Habitat / Low Risk</div>',
                unsafe_allow_html=True,
            )

    with bottom_col2:
        st.markdown("### Feature Attribution (SHAP)")
        st.plotly_chart(
            plot_shap_bar(res.get("shap_values", {})),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with bottom_col3:
        st.markdown("### Extracted Features")
        feats = res.get("extracted_features", {})

        lc_code = int(feats.get("landcover_class", 0))
        lc_label = ESA_LANDCOVER_MAP.get(lc_code, f"Class {lc_code}")

        c1, c2 = st.columns(2)
        with c1:
            elev_val = feats.get("elevation_m") or feats.get("elevation", 0)
            c1.metric("Elevation", f"{elev_val:.1f} m")
            bio1_val = feats.get("bio1_temp_c") or feats.get("bio1_mean_temp", 0)
            c1.metric("Mean Temp (BIO1)", f"{bio1_val:.2f} °C")
            bio4_val = feats.get("bio4_temp_seasonality", 0)
            if bio4_val:
                c1.metric("Temp Seasonality", f"{bio4_val:.0f}")

        with c2:
            c2.metric("Land Cover", lc_label)
            bio12_val = feats.get("bio12_precip_mm") or feats.get("bio12_annual_precip", 0)
            c2.metric("Annual Precip (BIO12)", f"{bio12_val:.1f} mm")
            bio15_val = feats.get("bio15_precip_seasonality", 0)
            if bio15_val:
                c2.metric("Precip Seasonality", f"{bio15_val:.1f}")