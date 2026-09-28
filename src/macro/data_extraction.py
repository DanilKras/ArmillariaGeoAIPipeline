import time
from pathlib import Path

import pandas as pd
from pygbif import occurrences as occ
from pygbif import species

from utils.config_loader import load_config


def fetch_gbif_occurrences(target_records: int = 5000) -> pd.DataFrame | None:
    config = load_config()
    target_species = config["project"]["target_species"]
    output_path = Path(config["paths"]["raw_data"])

    print(f"Resolving taxonKey for: {target_species}...")
    match = species.name_backbone(species=target_species)
    taxon_key = match.get("usageKey") or match.get("usage", {}).get("key")

    if not taxon_key:
        print(f"Error: taxonKey for '{target_species}' not found.")
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
        "year": "2015,2026",
        "basisOfRecord": [
            "HUMAN_OBSERVATION",
            "OBSERVATION",
            "PRESERVED_SPECIMEN",
            "MACHINE_OBSERVATION",
        ],
        "limit": limit,
    }

    print(f"Fetching up to {target_records} occurrence records from GBIF...")
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

    print(f"Downloaded {initial_count} records. Retained {len(df)} unique coordinates.")
    print(f"Saved raw points to: {output_path}")

    return df


if __name__ == "__main__":
    fetch_gbif_occurrences()
