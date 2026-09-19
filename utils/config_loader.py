from pathlib import Path

import yaml


def load_config(config_name: str = "params.yaml") -> dict:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config" / config_name

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
