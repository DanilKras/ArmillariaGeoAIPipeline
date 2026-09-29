from pathlib import Path

import pytest

from utils.config_loader import load_config


@pytest.fixture
def params():
    return load_config()


def test_params_yaml_exists():
    config_path = Path("config/params.yaml")
    assert config_path.exists(), "Config file config/params.yaml not found!"


def test_project_metadata(params):
    assert params.get("project", {}).get("target_species") == "Armillaria ostoyae"
    assert params.get("project", {}).get("region") == "Sopron_Hungary"


def test_spatial_settings(params):
    spatial = params.get("spatial", {})
    assert spatial.get("crs_projected") == "EPSG:3035"
    assert spatial.get("crs_wgs84") == "EPSG:4326"
    assert "bbox" in spatial
    assert len(spatial["bbox"]) == 4


def test_model_hyperparameters(params):
    model_params = params.get("model_params", {})
    assert model_params.get("learning_rate") == 0.03
    assert model_params.get("max_depth") == 5
    assert model_params.get("eval_metric") == "auc"
