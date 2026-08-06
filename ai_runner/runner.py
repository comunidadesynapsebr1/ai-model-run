"""
Carrega e executa modelos localmente usando a biblioteca `transformers`.
Focado em modelos de texto (causal LM), que é o caso mais comum de
"agentes de IA". Para outros tipos de modelo (imagem, áudio) o usuário
pode adaptar esta classe facilmente.
"""
from .registry import get_model


class ModelRunner:
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
        self._model = None
        self._tokenizer = None

    def load(self):
        """Carrega o modelo e o tokenizer na memória (lazy loading)."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "Pacotes 'torch' e 'transformers' não instalados. "
                "Rode: pip install -r requirements.txt"
            ) from e

        print(f"Carregando modelo de {self.local_path} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.local_path)

        device_map = "auto" if self.device == "auto" else None
        self._model = AutoModelForCausalLM.from_pretrained(
            self.local_path,
            device_map=device_map,
            torch_dtype="auto",
        )
        if device_map is None:
            self._model.to(self.device)
        print("Modelo carregado.")

    def generate(self, prompt: str, max_new_tokens: int = 200,
                 temperature: float = 0.7, top_p: float = 0.9,
                 seed: int | None = None) -> str:
        """Gera texto a partir de um prompt."""
        self.load()
        import torch

        if seed is not None:
            torch.manual_seed(seed)

        inputs = self._tokenizer(prompt, return_tensors="pt")
        if self.device != "auto":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return text
