from pathlib import Path
import yaml
import os


def load_config(config_path):
    """Load the YAML experiment configuration."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_project_path(config, path_key, create_if_missing=False):
    """
    Resolve a path from config['paths'] relative to project_dir.
    
    Args:
        config: Configuration dictionary
        path_key: Key in config['paths']
        create_if_missing: If True, create the directory if it doesn't exist
    
    Returns:
        Path as string
    """
    project_dir = Path(config["paths"]["project_dir"])
    
    # Handle case where path_key might not exist in config
    if path_key not in config["paths"]:
        # Construct default path from path_key
        configured_path = Path(f"results/{path_key}")
    else:
        configured_path = Path(config["paths"][path_key])

    if configured_path.is_absolute():
        resolved_path = str(configured_path)
    else:
        resolved_path = str(project_dir / configured_path)
    
    if create_if_missing:
        os.makedirs(resolved_path, exist_ok=True)
    
    return resolved_path


def detector_layers(config):
    return list(config["detector"]["hidden_layers"])
