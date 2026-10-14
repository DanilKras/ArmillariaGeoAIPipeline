from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from utils.config_loader import load_config
from utils.logger import get_logger

logger = get_logger(__name__)


def train_spatial_sdm():
    config = load_config()

    features = config["features"]["columns"]
    data_path = Path(config["paths"]["features_advanced"])
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {data_path}")

    df = pd.read_parquet(data_path)

    missing_cols = set(features) - set(df.columns)
    if missing_cols:
        raise KeyError(f"Missing expected features in parquet dataset: {missing_cols}")

    block_size = config["spatial"]["block_size_meters"]
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs=config["spatial"]["crs_wgs84"],
    ).to_crs(config["spatial"]["crs_projected"])

    block_x = (gdf.geometry.x // block_size).astype(int)
    block_y = (gdf.geometry.y // block_size).astype(int)
    gdf["spatial_block_id"] = block_x.astype(str) + "_" + block_y.astype(str)

    n_splits = config.get("cv_params", {}).get("n_splits", 5)
    gkf = GroupKFold(n_splits=n_splits)
    scores = []

    logger.info(
        f"Starting Spatial GroupKFold CV ({n_splits} splits) across {len(gdf)} records using {len(features)} features..."
    )

    for fold, (train_idx, val_idx) in enumerate(gkf.split(gdf, groups=gdf["spatial_block_id"])):
        train_data = gdf.iloc[train_idx]
        val_data = gdf.iloc[val_idx]

        if len(val_data["target"].unique()) < 2:
            logger.warning(
                f"Fold {fold + 1} contains only a single class. Skipping ROC-AUC calculation."
            )
            continue

        model = xgb.XGBClassifier(**config["model_params"])
        model.fit(train_data[features], train_data["target"])

        preds_proba = model.predict_proba(val_data[features])[:, 1]
        auc = roc_auc_score(val_data["target"], preds_proba)
        scores.append(auc)
        logger.info(f"Fold {fold + 1} | Spatial ROC-AUC: {auc:.3f}")

    mean_auc = pd.Series(scores).mean()
    logger.info(f"Mean Spatial ROC-AUC ({len(features)} features): {mean_auc:.3f}")

    final_model = xgb.XGBClassifier(**config["model_params"])
    final_model.fit(gdf[features], gdf["target"])

    model_path = Path(config["paths"]["model_weights"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(model_path))
    logger.info(f"Model weights saved to {model_path}")

    explainer = shap.TreeExplainer(final_model)
    sample_df = gdf[features].sample(min(1000, len(gdf)), random_state=42)
    shap_values = explainer(sample_df)

    shap_path = Path(config["paths"]["shap_summary_plot"])
    shap_path.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(shap_values, sample_df, show=False)
    plt.savefig(shap_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP summary plot successfully saved to {shap_path}")


if __name__ == "__main__":
    train_spatial_sdm()
