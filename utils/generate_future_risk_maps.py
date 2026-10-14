from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xgboost as xgb

from utils.config_loader import load_config
from utils.logger import get_logger

logger = get_logger(__name__)


def generate_future_projections():
    config = load_config()

    feature_columns = config["features"]["columns"]
    model_path = Path(config["paths"]["model_weights"])
    features_stack_tif = Path(
        config["paths"].get("feature_stack_tif", "data/03_results/maps/sopron_features_stack.tif")
    )

    logger.info(f"Loading trained XGBoost model from {model_path}...")
    clf = xgb.XGBClassifier()
    clf.load_model(str(model_path))

    if not features_stack_tif.exists():
        raise FileNotFoundError(f"Feature stack not found at {features_stack_tif}")

    with rasterio.open(features_stack_tif) as src:
        profile = src.profile.copy()
        dem_shape = (src.height, src.width)
        stack = src.read()
        valid_mask = (stack[0] > 0) & (stack[4] == 10)

    elevation = stack[0][valid_mask]
    slope_deg = stack[1][valid_mask]
    northness = stack[2][valid_mask]
    twi = stack[3][valid_mask]
    landcover = stack[4][valid_mask]
    bio1_base = stack[5][valid_mask]
    bio4_base = stack[6][valid_mask]
    bio12_base = stack[7][valid_mask]
    bio15_base = stack[8][valid_mask]
    soil_ph = stack[9][valid_mask]
    soil_clay = stack[10][valid_mask]

    scenarios = {
        "ssp245_2050": {
            "name": "SSP2-4.5 (2050)",
            "delta_bio1": 1.6,
            "delta_bio4": 25.0,
            "delta_bio12": -45.0,
            "delta_bio15": 3.2,
        },
        "ssp585_2050": {
            "name": "SSP5-8.5 (2050)",
            "delta_bio1": 2.8,
            "delta_bio4": 45.0,
            "delta_bio12": -85.0,
            "delta_bio15": 6.5,
        },
    }

    base_features = {
        "elevation": elevation,
        "slope_deg": slope_deg,
        "northness_aspect": northness,
        "twi_moisture_index": twi,
        "landcover_class": landcover,
        "bio1_mean_temp": bio1_base,
        "bio4_temp_seasonality": bio4_base,
        "bio12_annual_precip": bio12_base,
        "bio15_precip_seasonality": bio15_base,
        "soil_ph": soil_ph,
        "soil_clay_percent": soil_clay,
    }

    X_base = pd.DataFrame(base_features)[feature_columns]
    base_probs = clf.predict_proba(X_base)[:, 1]

    for key, sc in scenarios.items():
        logger.info(f"Computing SDM projections for scenario: {sc['name']}...")

        future_features = {
            "elevation": elevation,
            "slope_deg": slope_deg,
            "northness_aspect": northness,
            "twi_moisture_index": twi,
            "landcover_class": landcover,
            "bio1_mean_temp": bio1_base + sc["delta_bio1"],
            "bio4_temp_seasonality": bio4_base + sc["delta_bio4"],
            "bio12_annual_precip": np.maximum(bio12_base + sc["delta_bio12"], 200.0),
            "bio15_precip_seasonality": bio15_base + sc["delta_bio15"],
            "soil_ph": soil_ph,
            "soil_clay_percent": soil_clay,
        }

        X_future = pd.DataFrame(future_features)[feature_columns]
        future_probs = clf.predict_proba(X_future)[:, 1]

        risk_matrix = np.full(dem_shape, -9999.0, dtype=np.float32)
        risk_matrix[valid_mask] = future_probs

        out_path = Path(f"data/03_results/maps/sopron_armillaria_risk_{key}.tif")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        profile.update(count=1, dtype=rasterio.float32, nodata=-9999.0)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(risk_matrix, 1)

        if key == "ssp585_2050":
            delta_matrix = np.full(dem_shape, -9999.0, dtype=np.float32)
            delta_matrix[valid_mask] = future_probs - base_probs
            delta_path = Path("data/03_results/maps/sopron_armillaria_delta_ssp585_2050.tif")
            with rasterio.open(delta_path, "w", **profile) as dst:
                dst.write(delta_matrix, 1)
            logger.info(f"Delta risk map written to: {delta_path}")

    logger.info("All CMIP6 future risk maps generated successfully.")


if __name__ == "__main__":
    generate_future_projections()
