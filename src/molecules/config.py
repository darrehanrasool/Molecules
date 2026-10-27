"""Configuration loader and typed settings."""

from pathlib import Path
from typing import Any, Dict
import yaml


def get_project_root() -> Path:
    """Returns absolute path to repository root."""
    return Path(__file__).resolve().parent.parent.parent


def load_yaml_config(config_name: str) -> Dict[str, Any]:
    """Loads a YAML configuration file from configs/ directory."""
    root = get_project_root()
    path = root / "configs" / config_name
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


PROJECT_CONFIG = load_yaml_config("project.yaml")
DATA_CONFIG = load_yaml_config("data.yaml")
MODELS_CONFIG = load_yaml_config("models.yaml")
