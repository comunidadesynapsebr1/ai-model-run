"""
Geração de imagens com modelos de difusão (Stable Diffusion e
compatíveis) usando a biblioteca `diffusers`. Também serve para
modelos GAN simples se o usuário adaptar o carregamento (ver nota
no final do arquivo).
"""
from pathlib import Path
from typing import Optional

from .registry import get_model


class ImageRunner:
    def __init__(self, model_id: str, device: str = "auto"):
        info = get_model(model_id)
        if info is None:
            raise ValueError(
                f"Modelo '{model_id}' não encontrado localmente. "
                f"Rode primeiro: ai-runner pull {model_id}"
            )
        self.model_id = model_id
        self.local_path = info["local_path"]
        self.device = device
        self._pipe = None

    def load(self):
        if self._pipe is not None:
            return
        try:
            import torch
            from diffusers import DiffusionPipeline
        except ImportError as e:
            raise RuntimeError(
                "Pacote 'diffusers' não instalado. "
                "Rode: pip install diffusers accelerate (ou "
                "'pip install -r requirements.txt' se já estiver na lista)"
            ) from e

        print(f"Carregando pipeline de imagem para '{self.model_id}' ...")
        self._pipe = DiffusionPipeline.from_pretrained(
            self.local_path,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
        )
        if self.device == "auto":
            self._pipe = self._pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._pipe = self._pipe.to(self.device)
        print("Pipeline de imagem pronto.")

    def generate(self, prompt: str, output_path: str = "output.png",
                 negative_prompt: Optional[str] = None,
                 num_inference_steps: int = 30, guidance_scale: float = 7.5,
                 width: int = 512, height: int = 512,
                 seed: Optional[int] = None) -> Path:
        self.load()
        import torch

        generator = None
        if seed is not None:
            gen_device = "cuda" if torch.cuda.is_available() and self.device != "cpu" else "cpu"
            generator = torch.Generator(device=gen_device).manual_seed(seed)

        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        )
        image = result.images[0]
        out = Path(output_path)
        image.save(out)
        print(f"Imagem salva em: {out.resolve()}")
        return out


# Nota: para modelos GAN "crus" (ex: um .pth com um Generator PyTorch
# customizado, como o Synapse-GAN-Tiny), o diffusers não se aplica —
# nesse caso é melhor carregar a classe do Generator manualmente e
# rodar `generator(noise)` diretamente. Isso é específico de cada
# arquitetura, então fica fora do escopo genérico deste runner.
