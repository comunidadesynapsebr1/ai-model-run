"""
Registro local de modelos: mantém um JSON com metadados de cada
modelo baixado (id, caminho, tamanho, tipo, data de download etc).
"""
import json
import time
from pathlib import Path
from typing import Optional

from .config import REGISTRY_FILE, ensure_dirs


def _load() -> dict:
    ensure_dirs()
    if not REGISTRY_FILE.exists():
        return {}
    try:
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: dict):
    ensure_dirs()
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def add_model(model_id: str, local_path: str, model_type: str = "auto",
              extra: Optional[dict] = None):
    data = _load()
    data[model_id] = {
        "local_path": local_path,
        "type": model_type,
        "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **(extra or {}),
    }
    _save(data)


def remove_model(model_id: str):
    data = _load()
    if model_id in data:
        del data[model_id]
        _save(data)


def get_model(model_id: str) -> Optional[dict]:
    return _load().get(model_id)


def list_models() -> dict:
    return _load()
