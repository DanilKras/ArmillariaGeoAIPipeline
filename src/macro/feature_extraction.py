import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
import urllib3
import xarray as xr
from pystac_client.stac_api_io import StacApiIO
from tqdm import tqdm

from utils.config_loader import load_config
from utils.logger import get_logger

logger = get_logger(__name__)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def extract_point_from_raster(row, catalog, env_kwargs):
    """
    Extracts elevation and landcover class for a given point (lat, lon) from the specified STAC catalog.
    """
    lat, lon = row["lat"], row["lon"]
    point_geom = {"type": "Point", "coordinates": [lon, lat]}

    elevation = None
    landcover = None

    try:
        search_dem = catalog.search(collections=["cop-dem-glo-30"], intersects=point_geom)
        items_dem = list(search_dem.items())
        if items_dem:
            dem_href = items_dem[0].assets["data"].href
            with rasterio.Env(**env_kwargs):
                with rasterio.open(dem_href) as src:
                    for val in src.sample([(lon, lat)]):
                        elevation = float(val[0])

        search_lc = catalog.search(collections=["esa-worldcover"], intersects=point_geom)
        items_lc = list(search_lc.items())
        if items_lc:
            lc_href = items_lc[0].assets["map"].href
            with rasterio.Env(**env_kwargs):
                with rasterio.open(lc_href) as src:
                    for val in src.sample([(lon, lat)]):
                        landcover = int(val[0])

    except Exception as e:
        logger.warning(f"Failed to extract DEM/LC for gbif_id {row.get('gbif_id')}: {e}")

    return row["gbif_id"], elevation, landcover


def extract_sentinel2_indices(row, catalog, config):
    """
    Draft: Extraction of optical indices (NDVI, NDMI) with cloud filtering.
    """
    lat, lon = row["lat"], row["lon"]
    point_geom = {"type": "Point", "coordinates": [lon, lat]}

    start_date = config["temporal"]["start_date"]
    end_date = config["temporal"]["end_date"]
    max_cloud = config["sentinel2"]["max_cloud_cover"]

    try:
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            intersects=point_geom,
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": max_cloud}}
        )
        items = list(search.items())

        if not items:
            return row["gbif_id"], np.nan, np.nan

        # TODO: Sentinel-2 logic

    except Exception as e:
        logger.warning(f"Failed to extract Sentinel-2 for gbif_id {row.get('gbif_id')}: {e}")

    return row["gbif_id"], None, None


def run_advanced_feature_extraction() -> pd.DataFrame:
    config = load_config()

    input_path = config["paths"]["features_combined"]
    output_path = Path(config["paths"]["features_advanced"])
    checkpoint_path = Path(config["paths"]["features_terrain_checkpoint"])

    start_date = config["temporal"]["start_date"]
    end_date = config["temporal"]["end_date"]
    climate_vars = config["climate"]["variables"]

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

    if checkpoint_path.exists():
        logger.info(f"Loading cached terrain features from: {checkpoint_path}")
        df = pd.read_parquet(checkpoint_path)
    else:
        logger.info(f"Loading input samples: {input_path}")
        df = pd.read_csv(input_path)

        logger.info(f"Querying DEM & Landcover for {len(df)} points...")
        terrain_results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(extract_point_from_raster, row, catalog, env_kwargs)
                for _, row in df.iterrows()
            ]
            for f in tqdm(as_completed(futures), total=len(df), desc="DEM & WorldCover"):
                gbif_id, elev, lc = f.result()
                terrain_results.append(
                    {"gbif_id": gbif_id, "elevation": elev, "landcover_class": lc}
                )

        df = df.merge(pd.DataFrame(terrain_results), on="gbif_id", how="left")
        df = df.dropna(subset=["elevation", "landcover_class"])
        df = df[df["elevation"] > 0]

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(checkpoint_path, engine="pyarrow", compression="snappy")
        logger.info(f"Saved terrain checkpoint to: {checkpoint_path}")

    logger.info(f"Streaming TerraClimate variables ({climate_vars}) from {start_date} to {end_date}...")
    collection = catalog.get_collection("terraclimate")
    zarr_url = collection.assets["zarr-https"].href
    signed_url = planetary_computer.sign(zarr_url)

    parsed = urllib.parse.urlsplit(signed_url)
    base_endpoint = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    sas_params = dict(urllib.parse.parse_qsl(parsed.query))

    store = fsspec.get_mapper(base_endpoint, params=sas_params)
    ds = xr.open_zarr(store, consolidated=True)

    ds_clim = ds[climate_vars].sel(time=slice(start_date, end_date))

    lats_da = xr.DataArray(df["lat"].values, dims="points")
    lons_da = xr.DataArray(df["lon"].values, dims="points")

    extracted = ds_clim.sel(lat=lats_da, lon=lons_da, method="nearest").compute()

    if "tmax" in climate_vars and "tmin" in climate_vars:
        tmean = (extracted["tmax"] + extracted["tmin"]) * 0.5
        df["bio1_mean_temp"] = tmean.mean(dim="time").values
        df["bio4_temp_seasonality"] = (
                                          tmean.groupby("time.month").mean(dim="time").std(dim="month").values
                                      ) * 100

    if "ppt" in climate_vars:
        ppt = extracted["ppt"]
        annual_ppt = ppt.groupby("time.year").sum(dim="time")
        df["bio12_annual_precip"] = annual_ppt.mean(dim="year").values

        monthly_ppt = ppt.groupby("time.month").mean(dim="time")
        cv_ppt = (monthly_ppt.std(dim="month") / (monthly_ppt.mean(dim="month") + 1e-6)) * 100
        df["bio15_precip_seasonality"] = cv_ppt.values

    required_cols = ["elevation", "landcover_class"]
    if "tmax" in climate_vars and "tmin" in climate_vars:
        required_cols.extend(["bio1_mean_temp", "bio4_temp_seasonality"])
    if "ppt" in climate_vars:
        required_cols.extend(["bio12_annual_precip", "bio15_precip_seasonality"])

    df_final = df.dropna(subset=required_cols)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(output_path, engine="pyarrow", compression="snappy")

    logger.info(f"Features successfully saved to: {output_path} (Records: {len(df_final)})")
    return df_final


if __name__ == "__main__":
    run_advanced_feature_extraction()
