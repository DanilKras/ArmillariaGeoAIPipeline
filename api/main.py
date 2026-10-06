import os
import urllib.parse
from contextlib import asynccontextmanager
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
import shap
import urllib3
import xarray as xr
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .schemas import (
    ExtractedFeatures,
    InferenceRequest,
    InferenceResponse,
    RiskLevel,
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

FEATURE_COLUMNS = [
    "elevation",
    "landcover_class",
    "bio1_mean_temp",
    "bio4_temp_seasonality",
    "bio12_annual_precip",
    "bio15_precip_seasonality",
]

RISK_TIFF_PATH = os.getenv("RISK_MAP_PATH", "data/03_results/maps/sopron_armillaria_risk.tif")
MODEL_PATH = os.getenv("MODEL_PATH", "models/armillaria_spatial_xgb.json")

app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    app_state["model"] = model
    app_state["explainer"] = shap.TreeExplainer(model)

    if Path(RISK_TIFF_PATH).exists():
        app_state["raster_src"] = rasterio.open(RISK_TIFF_PATH)

    yield

    if "raster_src" in app_state:
        app_state["raster_src"].close()
    app_state.clear()


app = FastAPI(
    title="Dual-Engine GeoAI Inference API",
    description="Local Precomputed (<50ms) vs Cloud Streaming (~30s) Inference",
    lifespan=lifespan
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    first_msg = errors[0].get("msg", "Validation failed") if errors else "Invalid spatial input"
    return JSONResponse(status_code=422, content={"detail": first_msg.replace("Value error, ", "")})


@app.post("/predict/local", response_model=InferenceResponse)
def predict_local(request: InferenceRequest):
    src = app_state.get("raster_src")
    if not src:
        raise HTTPException(status_code=500, detail="Local raster not loaded. Run generate_risk_map.py first.")

    try:
        sample_gen = src.sample([(request.lon, request.lat)])
        sampled_val = float(next(sample_gen)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Raster sampling error: {e}")

    if np.isclose(sampled_val, -9999.0) or sampled_val < 0 or np.isnan(sampled_val):
        return InferenceResponse(
            risk_probability=0.0,
            risk_level=RiskLevel.NON_FOREST,
            is_forest=False,
            extracted_features=None,
            shap_values={},
        )

    prob = float(sampled_val)
    level = RiskLevel.HIGH if prob >= 0.66 else RiskLevel.MEDIUM if prob >= 0.33 else RiskLevel.LOW

    # TODO: Replace hardcode when multi-band local TIFF is ready
    sample_features = {
        "elevation": 420.5,
        "landcover_class": 10,
        "bio1_mean_temp": 9.82,
        "bio4_temp_seasonality": 685.2,
        "bio12_annual_precip": 710.4,
        "bio15_precip_seasonality": 24.8,
    }

    input_df = pd.DataFrame([sample_features])[FEATURE_COLUMNS]
    shap_vals = app_state["explainer"](input_df).values[0]

    return InferenceResponse(
        risk_probability=round(prob, 4),
        risk_level=level,
        is_forest=True,
        extracted_features=ExtractedFeatures(**{
            "elevation_m": sample_features["elevation"],
            "landcover_class": sample_features["landcover_class"],
            "bio1_temp_c": sample_features["bio1_mean_temp"],
            "bio4_seasonality": sample_features["bio4_temp_seasonality"],
            "bio12_precip_mm": sample_features["bio12_annual_precip"],
            "bio15_precip_seas": sample_features["bio15_precip_seasonality"]
        }),
        shap_values={col: round(float(val), 4) for col, val in zip(FEATURE_COLUMNS, shap_vals)},
    )


@app.post("/predict/cloud", response_model=InferenceResponse)
def predict_cloud(request: InferenceRequest):
    lat, lon = request.lat, request.lon
    point_geom = {"type": "Point", "coordinates": [lon, lat]}

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    env_kwargs = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": "tif",
        "GDAL_HTTP_UNSAFESSL": "YES",
    }

    try:
        elevation = 0.0
        search_dem = catalog.search(collections=["cop-dem-glo-30"], intersects=point_geom)
        items_dem = list(search_dem.items())
        if items_dem:
            with rasterio.Env(**env_kwargs):
                with rasterio.open(items_dem[0].assets["data"].href) as src:
                    elevation = float(next(src.sample([(lon, lat)]))[0])

        landcover = 0
        search_lc = catalog.search(collections=["esa-worldcover"], intersects=point_geom)
        items_lc = list(search_lc.items())
        if items_lc:
            with rasterio.Env(**env_kwargs):
                with rasterio.open(items_lc[0].assets["map"].href) as src:
                    landcover = int(next(src.sample([(lon, lat)]))[0])

        collection_clim = catalog.get_collection("terraclimate")
        zarr_url = collection_clim.assets["zarr-https"].href
        signed_url = planetary_computer.sign(zarr_url)

        parsed = urllib.parse.urlsplit(signed_url)
        base_endpoint = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        sas_params = dict(urllib.parse.parse_qsl(parsed.query))

        store = fsspec.get_mapper(base_endpoint, params=sas_params)
        ds = xr.open_zarr(store, consolidated=True)

        ds_clim = ds[["tmax", "tmin", "ppt"]].sel(time=slice("2015-01-01", "2025-12-31"))
        extracted = ds_clim.sel(lat=lat, lon=lon, method="nearest").compute()

        tmean = (extracted["tmax"] + extracted["tmin"]) * 0.5
        bio1 = float(tmean.mean(dim="time").values)
        bio4 = float((tmean.groupby("time.month").mean(dim="time").std(dim="month").values) * 100)

        ppt = extracted["ppt"]
        bio12 = float(ppt.groupby("time.year").sum(dim="time").mean(dim="year").values)
        monthly_ppt = ppt.groupby("time.month").mean(dim="time")
        bio15 = float((monthly_ppt.std(dim="month") / (monthly_ppt.mean(dim="month") + 1e-6)).values * 100)

        dynamic_features = {
            "elevation": elevation,
            "landcover_class": landcover,
            "bio1_mean_temp": bio1,
            "bio4_temp_seasonality": bio4,
            "bio12_annual_precip": bio12,
            "bio15_precip_seasonality": bio15,
        }

        input_df = pd.DataFrame([dynamic_features])[FEATURE_COLUMNS]
        prob = float(app_state["model"].predict_proba(input_df)[:, 1][0])
        shap_vals = app_state["explainer"](input_df).values[0]
        level = RiskLevel.HIGH if prob >= 0.66 else RiskLevel.MEDIUM if prob >= 0.33 else RiskLevel.LOW

        return InferenceResponse(
            risk_probability=round(prob, 4),
            risk_level=level if landcover == 10 else RiskLevel.NON_FOREST,
            is_forest=(landcover == 10),
            extracted_features=ExtractedFeatures(**{
                "elevation_m": dynamic_features["elevation"],
                "landcover_class": dynamic_features["landcover_class"],
                "bio1_temp_c": dynamic_features["bio1_mean_temp"],
                "bio4_seasonality": dynamic_features["bio4_temp_seasonality"],
                "bio12_precip_mm": dynamic_features["bio12_annual_precip"],
                "bio15_precip_seas": dynamic_features["bio15_precip_seasonality"]
            }),
            shap_values={col: round(float(val), 4) for col, val in zip(FEATURE_COLUMNS, shap_vals)},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cloud streaming failed: {str(e)}")
