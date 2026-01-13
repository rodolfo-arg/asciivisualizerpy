from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_CONFIG_PATH = Path("config.local.json")


def load_config(path: Optional[Path | str] = None) -> Dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_config(data: Dict[str, Any], path: Optional[Path | str] = None) -> Path:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return config_path
