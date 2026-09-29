import urllib.parse
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
import rasterio.warp
import urllib3
import xarray as xr
import xgboost as xgb
from pystac_client.stac_api_io import StacApiIO
from rasterio.enums import Resampling

from utils.config_loader import load_config

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def generate_risk_map() -> None:
    config = load_config()

    model_path = config["paths"]["model_weights"]
    output_tif = config["paths"]["risk_map_tif"]
    bbox = config["spatial"]["bbox"]
    min_lon, min_lat, max_lon, max_lat = bbox

    print(f"Generating risk map for region: {config['project']['region']}")

    stac_io = StacApiIO()
    stac_io.session.verify = False
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
        stac_io=stac_io,
    )

    env_kwargs = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": "tif",
        "GDAL_HTTP_UNSAFESSL": "YES",
    }

    print("Reading Copernicus DEM...")
    search_dem = catalog.search(collections=["cop-dem-glo-30"], bbox=bbox)
    items_dem = list(search_dem.items())
    if not items_dem:
        raise ValueError("DEM tile not found for bbox.")

    dem_href = items_dem[0].assets["data"].href

    with rasterio.Env(**env_kwargs):
        with rasterio.open(dem_href) as src:
            py1, px1 = src.index(min_lon, max_lat)
            py2, px2 = src.index(max_lon, min_lat)
            window = rasterio.windows.Window.from_slices((py1, py2), (px1, px2))

            elevation_matrix = src.read(1, window=window).astype(np.float32)
            transform = src.window_transform(window)
            dem_crs = src.crs
            dem_shape = elevation_matrix.shape

            cols, rows = np.meshgrid(np.arange(window.width), np.arange(window.height))
            lons, lats = transform * (cols, rows)
            lons = np.array(lons, dtype=np.float32)
            lats = np.array(lats, dtype=np.float32)

    print("Reading and aligning ESA WorldCover...")
    search_lc = catalog.search(collections=["esa-worldcover"], bbox=bbox)
    items_lc = list(search_lc.items())
    if not items_lc:
        raise ValueError("ESA WorldCover tile not found.")

    items_lc.sort(key=lambda x: x.datetime if x.datetime else x.id, reverse=True)
    lc_href = items_lc[0].assets["map"].href

    with rasterio.Env(**env_kwargs):
        with rasterio.open(lc_href) as src:
            landcover_matrix = np.empty(dem_shape, dtype=np.float32)
            rasterio.warp.reproject(
                source=rasterio.band(src, 1),
                destination=landcover_matrix,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dem_crs,
                resampling=Resampling.nearest,
            )

    valid_mask = (elevation_matrix > 0) & (landcover_matrix == 10)
    valid_lats = lats[valid_mask]
    valid_lons = lons[valid_mask]
    print("Reading and interpolating TerraClimate variables...")
    collection_clim = catalog.get_collection("terraclimate")
    zarr_url = collection_clim.assets["zarr-https"].href
    signed_url = planetary_computer.sign(zarr_url)

    parsed_url = urllib.parse.urlsplit(signed_url)
    base_url = urllib.parse.urlunsplit(
        (parsed_url.scheme, parsed_url.netloc, parsed_url.path, "", "")
    )
    sas_dict = dict(urllib.parse.parse_qsl(parsed_url.query))

    store = fsspec.get_mapper(base_url, params=sas_dict)
    ds = xr.open_zarr(store, consolidated=True)
    ds_clim = ds[["tmax", "tmin", "ppt"]].sel(time=slice("2011-01-01", "2020-12-31"))

    lats_da = xr.DataArray(valid_lats, dims="points")
    lons_da = xr.DataArray(valid_lons, dims="points")

    extracted = ds_clim.interp(lat=lats_da, lon=lons_da, method="linear").compute()

    tmean = (extracted["tmax"] + extracted["tmin"]) * 0.5 * 0.1
    ppt = extracted["ppt"]

    bio1 = tmean.mean(dim="time").values
    bio4 = (tmean.groupby("time.month").mean(dim="time").std(dim="month").values) * 100
    bio12 = ppt.groupby("time.year").sum(dim="time").mean(dim="year").values

    monthly_ppt = ppt.groupby("time.month").mean(dim="time")
    bio15 = (monthly_ppt.std(dim="month") / (monthly_ppt.mean(dim="month") + 1e-6) * 100).values

    X_valid = pd.DataFrame(
        {
            "elevation": elevation_matrix[valid_mask],
            "landcover_class": landcover_matrix[valid_mask],
            "bio1_mean_temp": bio1,
            "bio4_temp_seasonality": bio4,
            "bio12_annual_precip": bio12,
            "bio15_precip_seasonality": bio15,
        }
    )

    print(f"Loading model from: {model_path}")
    clf = xgb.XGBClassifier()
    clf.load_model(model_path)

    probabilities = clf.predict_proba(X_valid)[:, 1]

    risk_matrix = np.full(dem_shape, -9999.0, dtype=np.float32)
    risk_matrix[valid_mask] = probabilities

    Path(output_tif).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_tif,
        "w",
        driver="GTiff",
        height=dem_shape[0],
        width=dem_shape[1],
        count=1,
        dtype=np.float32,
        crs=dem_crs,
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(risk_matrix, 1)

    print(f"Risk map successfully written to: {output_tif}")


if __name__ == "__main__":
    generate_risk_map()
