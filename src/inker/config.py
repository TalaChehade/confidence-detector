from pathlib import Path
import yaml


def load_config(config_path):
    """Load the YAML experiment configuration."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_project_path(config, path_key):
    """Resolve a path from config['paths'] relative to project_dir."""
    project_dir = Path(config["paths"]["project_dir"])
    configured_path = Path(config["paths"][path_key])

    if configured_path.is_absolute():
        return str(configured_path)

    return str(project_dir / configured_path)


def detector_layers(config):
    return list(config["detector"]["hidden_layers"])
