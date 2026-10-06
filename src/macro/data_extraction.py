import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pygbif import occurrences as occ
from pygbif import species

from utils.config_loader import load_config
from utils.logger import get_logger

logger = get_logger(__name__)


def fetch_gbif_occurrences(target_records: int = 5000) -> pd.DataFrame | None:
    config = load_config()
    target_species = config["project"]["target_species"]
    output_path = Path(config["paths"]["raw_data"])
    start_year = config["temporal"]["start_date"][:4]
    end_year = config["temporal"]["end_date"][:4]

    logger.info(f"Resolving taxonKey for: {target_species}...")
    match = species.name_backbone(species=target_species)
    taxon_key = match.get("usageKey") or match.get("usage", {}).get("key")

    if not taxon_key:
        logger.error(f"taxonKey for '{target_species}' not found.")
        return None

    limit = 300
    offset = 0
    raw_records = []

    search_params = {
        "taxonKey": taxon_key,
        "hasCoordinate": True,
        "hasGeospatialIssue": False,
        "continent": "EUROPE",
        "coordinateUncertaintyInMeters": "0,250",
        "year": f"{start_year},{end_year}",
        "basisOfRecord": [
            "HUMAN_OBSERVATION",
            "OBSERVATION",
            "PRESERVED_SPECIMEN",
            "MACHINE_OBSERVATION",
        ],
        "limit": limit,
    }

    logger.info(f"Fetching up to {target_records} occurrence records from GBIF...")
    while len(raw_records) < target_records:
        response = occ.search(**search_params, offset=offset)
        records = response.get("results", [])

        if not records:
            break

        raw_records.extend(records)
        offset += limit
        time.sleep(0.2)

        if response.get("endOfRecords") or len(raw_records) >= response.get("count", 0):
            break

    raw_records = raw_records[:target_records]

    parsed_data = [
        {
            "gbif_id": r.get("key"),
            "species": r.get("species", target_species),
            "country": r.get("country"),
            "lat": r.get("decimalLatitude"),
            "lon": r.get("decimalLongitude"),
            "year": r.get("year"),
            "basis_of_record": r.get("basisOfRecord"),
            "coordinate_uncertainty_m": r.get("coordinateUncertaintyInMeters"),
            "target": 1,
        }
        for r in raw_records
    ]

    df = pd.DataFrame(parsed_data)
    initial_count = len(df)

    df = df.drop_duplicates(subset=["lat", "lon"]).dropna(subset=["lat", "lon"])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".parquet":
        df.to_parquet(output_path, engine="pyarrow", compression="snappy")
    else:
        df.to_csv(output_path, index=False)

    logger.info(f"Downloaded {initial_count} records. Retained {len(df)} unique coordinates.")
    logger.info(f"Saved raw points to: {output_path}")

    return df


def extract_from_local(file_path: Path, target_species: str) -> pd.DataFrame:
    if file_path.suffix in [".gpkg", ".shp", ".geojson"]:
        gdf = gpd.read_file(file_path)
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        df = pd.DataFrame({
            "gbif_id": [f"local_{i}" for i in range(len(gdf))],
            "species": target_species,
            "country": "Hungary",
            "lat": gdf.geometry.y,
            "lon": gdf.geometry.x,
            "year": gdf.get("year", None),
            "basis_of_record": "FIELD_OBSERVATION",
            "coordinate_uncertainty_m": 10.0,
            "target": 1,
        })
    else:
        df = pd.read_csv(file_path)
        df["target"] = 1

    return df


def extract_all_raw_data() -> pd.DataFrame:
    config = load_config()
    source_type = config.get("data_source", {}).get("type", "gbif")
    target_species = config["project"]["target_species"]
    output_path = Path(config["paths"]["raw_data"])

    if source_type == "local":
        local_path_str = config.get("data_source", {}).get("local_path")
        if local_path_str and Path(local_path_str).exists():
            df = extract_from_local(Path(local_path_str), target_species)
        else:
            logger.warning("Local file not specified or doesn't exist. Falling back to GBIF...")
            df = fetch_gbif_occurrences()
    else:
        df = fetch_gbif_occurrences()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        df.to_parquet(output_path, engine="pyarrow", compression="snappy")
    else:
        df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    fetch_gbif_occurrences()
