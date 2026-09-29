from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from utils.config_loader import load_config


def generate_ecological_pseudo_absences(
    min_dist_meters: int = 5000,
    max_extent_buffer_m: int = 200000,
    seed: int = 42,
) -> pd.DataFrame:
    config = load_config()
    rng = np.random.default_rng(seed)

    positive_path = config["paths"]["clean_gpkg"]
    output_path = Path(config["paths"]["features_combined"])
    crs_projected = config["spatial"]["crs_projected"]
    crs_wgs84 = config["spatial"]["crs_wgs84"]

    print("Loading presence locations (Class 1)...")
    gdf_pos = gpd.read_file(positive_path)
    num_points = len(gdf_pos)

    if num_points == 0:
        raise ValueError("Presence dataset is empty.")

    gdf_pos_metric = gdf_pos.to_crs(crs_projected)
    pos_coords = np.column_stack([gdf_pos_metric.geometry.x, gdf_pos_metric.geometry.y])

    tree = KDTree(pos_coords)

    minx, maxx = (
        pos_coords[:, 0].min() - max_extent_buffer_m,
        pos_coords[:, 0].max() + max_extent_buffer_m,
    )
    miny, maxy = (
        pos_coords[:, 1].min() - max_extent_buffer_m,
        pos_coords[:, 1].max() + max_extent_buffer_m,
    )

    print(f"Generating {num_points} pseudo-absences with distance >= {min_dist_meters}m...")
    neg_coords = []

    while len(neg_coords) < num_points:
        needed = num_points - len(neg_coords)
        batch_size = max(needed * 2, 1000)

        rand_x = rng.uniform(minx, maxx, size=batch_size)
        rand_y = rng.uniform(miny, maxy, size=batch_size)
        candidates = np.column_stack([rand_x, rand_y])

        distances, _ = tree.query(candidates)
        valid_candidates = candidates[distances >= min_dist_meters]

        neg_coords.extend(valid_candidates[:needed])

    neg_coords = np.array(neg_coords)

    gdf_neg = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(neg_coords[:, 0], neg_coords[:, 1]),
        crs=crs_projected,
    ).to_crs(crs_wgs84)

    gdf_pos_gps = gdf_pos_metric.to_crs(crs_wgs84)

    df_pos = pd.DataFrame(
        {
            "gbif_id": gdf_pos_gps.get("gbif_id", [f"pos_{i}" for i in range(num_points)]),
            "species": "Armillaria ostoyae",
            "country": gdf_pos_gps.get("country", None),
            "lat": gdf_pos_gps.geometry.y,
            "lon": gdf_pos_gps.geometry.x,
            "year": gdf_pos_gps.get("year", None),
            "basis_of_record": gdf_pos_gps.get("basis_of_record", "OBSERVATION"),
            "coordinate_uncertainty_m": gdf_pos_gps.get("coordinate_uncertainty_m", None),
            "target": 1,
        }
    )

    df_neg = pd.DataFrame(
        {
            "gbif_id": [f"synth_{i}" for i in range(num_points)],
            "species": "Pseudo_absence",
            "country": None,
            "lat": gdf_neg.geometry.y,
            "lon": gdf_neg.geometry.x,
            "year": None,
            "basis_of_record": "SYNTHETIC",
            "coordinate_uncertainty_m": None,
            "target": 0,
        }
    )

    df_combined = pd.concat([df_pos, df_neg], ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(output_path, index=False)
    print(f"Saved {len(df_combined)} samples (1:1 balanced) to {output_path}")

    return df_combined


if __name__ == "__main__":
    generate_ecological_pseudo_absences()
