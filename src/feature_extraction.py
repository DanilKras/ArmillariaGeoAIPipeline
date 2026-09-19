import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fsspec
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
import urllib3
import xarray as xr
from pystac_client.stac_api_io import StacApiIO
from tqdm import tqdm

from utils.config_loader import load_config

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def get_point_elevation_and_landcover(row, catalog):
    """
    Fetches elevation (Copernicus DEM 30m) and landcover class (ESA WorldCover 10m)
    for a single point via Planetary Computer STAC.
    """
    lat, lon = row["lat"], row["lon"]
    point_geom = {"type": "Point", "coordinates": [lon, lat]}

    elevation = None
    landcover = None

    env_kwargs = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": "tif",
        "GDAL_HTTP_UNSAFESSL": "YES",
    }

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

    except Exception:
        pass

    return row["gbif_id"], elevation, landcover


def run_advanced_feature_extraction() -> pd.DataFrame:
    config = load_config()

    input_path = config["paths"]["features_combined"]
    output_path = Path(config["paths"]["features_advanced"])
    checkpoint_path = Path(config["paths"]["features_terrain_checkpoint"])

    stac_io = StacApiIO()
    stac_io.session.verify = False
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
        stac_io=stac_io,
    )

    if checkpoint_path.exists():
        print(f"Loading cached terrain features from: {checkpoint_path}")
        df = pd.read_parquet(checkpoint_path)
    else:
        print(f"Loading input samples: {input_path}")
        df = pd.read_csv(input_path)

        print(f"Querying DEM & Landcover for {len(df)} points...")
        terrain_results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(get_point_elevation_and_landcover, row, catalog)
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
        print(f"Saved terrain checkpoint to: {checkpoint_path}")

    print("Streaming bioclimatic variables from TerraClimate Zarr...")
    collection = catalog.get_collection("terraclimate")
    zarr_url = collection.assets["zarr-https"].href
    signed_url = planetary_computer.sign(zarr_url)

    parsed = urllib.parse.urlsplit(signed_url)
    base_endpoint = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    sas_params = dict(urllib.parse.parse_qsl(parsed.query))

    store = fsspec.get_mapper(base_endpoint, params=sas_params)
    ds = xr.open_zarr(store, consolidated=True)
    ds_clim = ds[["tmax", "tmin", "ppt"]].sel(time=slice("2011-01-01", "2020-12-31"))

    lats_da = xr.DataArray(df["lat"].values, dims="points")
    lons_da = xr.DataArray(df["lon"].values, dims="points")

    extracted = ds_clim.sel(lat=lats_da, lon=lons_da, method="nearest").compute()

    tmean = (extracted["tmax"] + extracted["tmin"]) * 0.5 * 0.1
    ppt = extracted["ppt"]

    df["bio1_mean_temp"] = tmean.mean(dim="time").values
    df["bio4_temp_seasonality"] = (
        tmean.groupby("time.month").mean(dim="time").std(dim="month").values
    ) * 100

    annual_ppt = ppt.groupby("time.year").sum(dim="time")
    df["bio12_annual_precip"] = annual_ppt.mean(dim="year").values

    monthly_ppt = ppt.groupby("time.month").mean(dim="time")
    cv_ppt = (monthly_ppt.std(dim="month") / (monthly_ppt.mean(dim="month") + 1e-6)) * 100
    df["bio15_precip_seasonality"] = cv_ppt.values

    df_final = df.dropna(
        subset=["elevation", "landcover_class", "bio1_mean_temp", "bio12_annual_precip"]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(output_path, engine="pyarrow", compression="snappy")

    print(f"Features successfully saved to: {output_path} (Records: {len(df_final)})")
    return df_final


if __name__ == "__main__":
    run_advanced_feature_extraction()
