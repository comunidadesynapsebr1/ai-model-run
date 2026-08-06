"""
Configurações globais do AI Runner.
Define onde os modelos e metadados ficam armazenados localmente.
"""
import os
from pathlib import Path

# Diretório base: ~/.ai_runner
BASE_DIR = Path(os.environ.get("AI_RUNNER_HOME", Path.home() / ".ai_runner"))
MODELS_DIR = BASE_DIR / "models"
REGISTRY_FILE = BASE_DIR / "registry.json"
AGENTS_DIR = BASE_DIR / "agents"
LOGS_DIR = BASE_DIR / "logs"


def ensure_dirs():
    """Garante que todos os diretórios necessários existem."""
    for d in (BASE_DIR, MODELS_DIR, AGENTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def model_path(model_id: str) -> Path:
    """
    Converte um model_id (ex: 'meta-llama/Llama-3.2-1B') em um
    caminho local seguro dentro de MODELS_DIR.
    """
    safe_name = model_id.replace("/", "__")
    return MODELS_DIR / safe_name
