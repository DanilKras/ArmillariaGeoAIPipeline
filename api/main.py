import os
import sys
import urllib.parse
from contextlib import asynccontextmanager
from pathlib import Path

import dask
import fsspec
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
import shap
import xarray as xr
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pystac_client.stac_api_io import StacApiIO

sys.modules["distributed"] = None


class InferenceRequest(BaseModel):
    lat: float = Field(..., description="Latitude in WGS84", ge=-90, le=90)
    lon: float = Field(..., description="Longitude in WGS84", ge=-180, le=180)


class InferenceResponse(BaseModel):
    risk_probability: float
    risk_level: str
    extracted_features: dict
    shap_values: dict


class GroundTruthPoint(BaseModel):
    lat: float
    lon: float
    status: int
    notes: str = ""


app_state = {}

FEATURE_COLUMNS = [
    "elevation",
    "landcover_class",
    "bio1_mean_temp",
    "bio4_temp_seasonality",
    "bio12_annual_precip",
    "bio15_precip_seasonality",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = "models/armillaria_spatial_xgb.json"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    app_state["model"] = model

    explainer = shap.TreeExplainer(model)
    app_state["explainer"] = explainer

    stac_io = StacApiIO()
    stac_io.session.verify = False
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
        stac_io=stac_io,
    )
    app_state["catalog"] = catalog

    yield
    app_state.clear()


app = FastAPI(title="Explainable GeoAI API", lifespan=lifespan)


def extract_elevation(lat: float, lon: float, catalog) -> float:
    point_geom = {"type": "Point", "coordinates": [lon, lat]}
    search_dem = catalog.search(collections=["cop-dem-glo-30"], intersects=point_geom)
    items_dem = list(search_dem.items())

    if not items_dem:
        return 0.0

    dem_href = items_dem[0].assets["data"].href
    env_kwargs = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "GDAL_HTTP_UNSAFESSL": "YES"}

    with rasterio.Env(**env_kwargs):
        with rasterio.open(dem_href) as src:
            for val in src.sample([(lon, lat)]):
                return float(val[0])
    return 0.0


def extract_climate_bioclim(lat: float, lon: float, catalog) -> tuple:
    collection = catalog.get_collection("terraclimate")
    zarr_url = collection.assets["zarr-https"].href
    signed_url = planetary_computer.sign(zarr_url)

    parsed = urllib.parse.urlsplit(signed_url)
    base_endpoint = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    sas_params = dict(urllib.parse.parse_qsl(parsed.query))

    store = fsspec.get_mapper(base_endpoint, params=sas_params)
    ds = xr.open_zarr(store, consolidated=True, mask_and_scale=False)
    ds_clim = ds[["tmax", "tmin", "ppt"]].sel(time=slice("2011-01-01", "2020-12-31"))

    lat_da = xr.DataArray([lat], dims="points")
    lon_da = xr.DataArray([lon], dims="points")

    with dask.config.set(scheduler="synchronous"):
        pt_clim = ds_clim.sel(lat=lat_da, lon=lon_da, method="nearest").compute()

    tmean = (pt_clim["tmax"] + pt_clim["tmin"]) * 0.5 * 0.1
    ppt = pt_clim["ppt"]

    bio1 = float(tmean.mean(dim="time").values.squeeze())
    monthly_tmean = tmean.groupby("time.month").mean(dim="time")
    bio4 = float(monthly_tmean.std(dim="month").values.squeeze()) * 100.0

    annual_ppt = ppt.groupby("time.year").sum(dim="time")
    bio12 = float(annual_ppt.mean(dim="year").values.squeeze())

    monthly_ppt = ppt.groupby("time.month").mean(dim="time")
    ppt_mean = monthly_ppt.mean(dim="month")
    ppt_std = monthly_ppt.std(dim="month")
    bio15 = float(((ppt_std / (ppt_mean + 1e-5)) * 100.0).values.squeeze())

    return bio1, bio4, bio12, bio15


def extract_landcover(lat: float, lon: float, catalog) -> float:
    point_geom = {"type": "Point", "coordinates": [lon, lat]}
    search = catalog.search(collections=["esa-worldcover"], intersects=point_geom)
    items = list(search.items())

    if not items:
        return 40.0

    map_href = items[0].assets["map"].href
    env_kwargs = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "GDAL_HTTP_UNSAFESSL": "YES"}

    with rasterio.Env(**env_kwargs):
        with rasterio.open(map_href) as src:
            for val in src.sample([(lon, lat)]):
                return float(val[0])
    return 40.0


@app.post("/predict", response_model=InferenceResponse)
async def predict_risk(request: InferenceRequest):
    model = app_state.get("model")
    explainer = app_state.get("explainer")
    catalog = app_state.get("catalog")

    if not model or not catalog or not explainer:
        raise HTTPException(status_code=500, detail="Services not initialized")

    try:
        elev = extract_elevation(request.lat, request.lon, catalog)
        bio1, bio4, bio12, bio15 = extract_climate_bioclim(request.lat, request.lon, catalog)
        lc = extract_landcover(request.lat, request.lon, catalog)

        input_dict = {
            "elevation": elev,
            "landcover_class": lc,
            "bio1_mean_temp": bio1,
            "bio4_temp_seasonality": bio4,
            "bio12_annual_precip": bio12,
            "bio15_precip_seasonality": bio15,
        }
        input_data = pd.DataFrame([input_dict])[FEATURE_COLUMNS]

        prob = float(model.predict_proba(input_data)[0][1])
        level = "High" if prob >= 0.66 else "Medium" if prob >= 0.33 else "Low"

        shap_vals = explainer(input_data)
        vals = shap_vals.values[0]

        shap_dict = {col: float(val) for col, val in zip(FEATURE_COLUMNS, vals)}

        return InferenceResponse(
            risk_probability=prob,
            risk_level=level,
            extracted_features={
                "elevation_m": round(elev, 1),
                "landcover_class": int(lc),
                "bio1_temp_c": round(bio1, 2),
                "bio4_seasonality": round(bio4, 1),
                "bio12_precip_mm": round(bio12, 1),
                "bio15_precip_seas": round(bio15, 1),
            },
            shap_values=shap_dict,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
def log_ground_truth(point: GroundTruthPoint):
    csv_path = Path("data/01_raw/field_feedback.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    new_data = pd.DataFrame(
        [
            {
                "lat": point.lat,
                "lon": point.lon,
                "target": point.status,
                "notes": point.notes,
                "timestamp": pd.Timestamp.now(),
            }
        ]
    )

    if csv_path.exists():
        new_data.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        new_data.to_csv(csv_path, mode="w", header=True, index=False)

    return {"status": "success", "message": "Field observation recorded successfully."}