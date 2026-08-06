"""
Baixa modelos do Hugging Face Hub (ou de uma URL direta) para o
diretório local de modelos, e registra no registry.json.

Funciona com qualquer repositório do Hugging Face: LLMs, modelos de
difusão, GANs, embeddings, etc. — baixa os arquivos "crus" do repo.
"""
import shutil
from pathlib import Path

from .config import model_path, ensure_dirs
from .registry import add_model


def pull_model(model_id: str, revision: str = "main",
                allow_patterns=None, model_type: str = "auto") -> Path:
    """
    Baixa um modelo do Hugging Face Hub usando huggingface_hub.snapshot_download.

    model_id: ex. "gpt2", "meta-llama/Llama-3.2-1B-Instruct",
              "runwayml/stable-diffusion-v1-5"
    revision: branch/tag/commit do repo (default "main")
    allow_patterns: lista de padrões glob para baixar só alguns arquivos
                    (ex: ["*.safetensors", "*.json"]) — útil para
                    economizar espaço/banda.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "Pacote 'huggingface_hub' não instalado. "
            "Rode: pip install -r requirements.txt"
        ) from e

    ensure_dirs()
    dest = model_path(model_id)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Baixando '{model_id}' (revision={revision}) para {dest} ...")
    local_dir = snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=str(dest),
        allow_patterns=allow_patterns,
    )
    print("Download concluído.")

    add_model(model_id, str(local_dir), model_type=model_type)
    return Path(local_dir)


def remove_local_model(model_id: str):
    """Remove os arquivos locais de um modelo e tira do registro."""
    from .registry import remove_model
    dest = model_path(model_id)
    if dest.exists():
        shutil.rmtree(dest)
        print(f"Removido: {dest}")
    remove_model(model_id)
