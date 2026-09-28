from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from utils.config_loader import load_config


def train_spatial_sdm():
    config = load_config()

    df = pd.read_parquet(config["paths"]["features_advanced"])
    features = [
        "elevation",
        "landcover_class",
        "bio1_mean_temp",
        "bio4_temp_seasonality",
        "bio12_annual_precip",
        "bio15_precip_seasonality",
    ]

    block_size = config["spatial"]["block_size_meters"]
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs=config["spatial"]["crs_wgs84"],
    ).to_crs(config["spatial"]["crs_projected"])

    block_x = (gdf.geometry.x // block_size).astype(int)
    block_y = (gdf.geometry.y // block_size).astype(int)
    gdf["spatial_block_id"] = block_x.astype(str) + "_" + block_y.astype(str)

    gkf = GroupKFold(n_splits=config.get("cv_params", {}).get("n_splits", 5))
    scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(gdf, groups=gdf["spatial_block_id"])):
        train_data = gdf.iloc[train_idx]
        val_data = gdf.iloc[val_idx]

        model = xgb.XGBClassifier(**config["model_params"])
        model.fit(train_data[features], train_data["target"])

        preds_proba = model.predict_proba(val_data[features])[:, 1]
        auc = roc_auc_score(val_data["target"], preds_proba)
        scores.append(auc)
        print(f"Fold {fold + 1} | ROC-AUC: {auc:.3f}")

    print(f"Mean Spatial ROC-AUC: {pd.Series(scores).mean():.3f}")

    final_model = xgb.XGBClassifier(**config["model_params"])
    final_model.fit(gdf[features], gdf["target"])

    model_path = Path(config["paths"]["model_weights"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(model_path)
    print(f"Model saved to {model_path}")

    explainer = shap.TreeExplainer(final_model)
    sample_df = gdf[features].sample(min(1000, len(gdf)), random_state=42)
    shap_values = explainer(sample_df)

    shap_path = Path(config["paths"]["shap_summary_plot"])
    shap_path.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(shap_values, sample_df, show=False)
    plt.savefig(shap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP plot saved to {shap_path}")


if __name__ == "__main__":
    train_spatial_sdm()
