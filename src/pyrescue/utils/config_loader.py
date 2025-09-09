import json

from src.pyrescue.config import DEFAULT_CONFIG


def load_config(user_config_path: str | None):
    config = DEFAULT_CONFIG.copy()

    if user_config_path:
        with open(user_config_path, "r") as f:
            user_config = json.load(f)

        _merge_dicts(config, user_config)

    return config


def _merge_dicts(target: dict, source: dict):
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            _merge_dicts(target[key], value)
        else:
            target[key] = value
