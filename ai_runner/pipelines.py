"""
Runner genérico para tarefas além de geração de texto livre, usando o
`pipeline()` da biblioteca `transformers`. Suporta qualquer modelo que
tenha um pipeline correspondente no Hub.

Tarefas suportadas (task_type):
  - "text-generation"        -> completar/continuar um texto
  - "summarization"          -> resumir um texto
  - "translation"            -> traduzir um texto
  - "text-classification"    -> classificar um texto (ex: sentimento)
  - "zero-shot-classification" -> classificar em categorias arbitrárias
  - "question-answering"     -> responder pergunta dado um contexto
  - "fill-mask"               -> preencher [MASK] em uma frase
  - "feature-extraction"      -> gerar embeddings (vetores) de um texto
  - "automatic-speech-recognition" -> transcrever áudio para texto
  - "image-classification"    -> classificar uma imagem
  - "image-to-text"           -> gerar legenda/descrição de uma imagem
"""
from typing import Any, Optional

from .registry import get_model

SUPPORTED_TASKS = [
    "text-generation",
    "summarization",
    "translation",
    "text-classification",
    "zero-shot-classification",
    "question-answering",
    "fill-mask",
    "feature-extraction",
    "automatic-speech-recognition",
    "image-classification",
    "image-to-text",
]


class PipelineRunner:
    def __init__(self, model_id: str, task_type: str, device: str = "auto"):
        if task_type not in SUPPORTED_TASKS:
            raise ValueError(
                f"task_type '{task_type}' não suportado. Opções: {', '.join(SUPPORTED_TASKS)}"
            )
        info = get_model(model_id)
        if info is None:
            raise ValueError(
                f"Modelo '{model_id}' não encontrado localmente. "
                f"Rode primeiro: ai-runner pull {model_id}"
            )
        self.model_id = model_id
        self.local_path = info["local_path"]
        self.task_type = task_type
        self.device = device
        self._pipe = None

    def load(self):
        if self._pipe is not None:
            return
        try:
            from transformers import pipeline
            import torch
        except ImportError as e:
            raise RuntimeError(
                "Pacotes 'torch' e 'transformers' não instalados. "
                "Rode: pip install -r requirements.txt"
            ) from e

        device_arg = 0 if (self.device == "auto" and torch.cuda.is_available()) else \
            (-1 if self.device in ("auto", "cpu") else self.device)

        print(f"Carregando pipeline '{self.task_type}' para '{self.model_id}' ...")
        self._pipe = pipeline(
            task=self.task_type,
            model=self.local_path,
            tokenizer=self.local_path if self.task_type not in
            ("image-classification", "automatic-speech-recognition", "image-to-text") else None,
            device=device_arg,
        )
        print("Pipeline pronto.")

    def run(self, main_input: Any, **kwargs) -> Any:
        """
        Executa a tarefa. `main_input` é o dado principal (texto, caminho
        de arquivo de áudio, caminho/URL de imagem, etc). `kwargs` são
        parâmetros extras específicos da tarefa (ex: candidate_labels
        para zero-shot, question/context para QA).
        """
        self.load()

        if self.task_type == "question-answering":
            question = kwargs.get("question")
            context = kwargs.get("context", main_input)
            if not question:
                raise ValueError("question-answering requer --question")
            return self._pipe(question=question, context=context)

        if self.task_type == "zero-shot-classification":
            labels = kwargs.get("candidate_labels")
            if not labels:
                raise ValueError("zero-shot-classification requer --labels 'a,b,c'")
            return self._pipe(main_input, candidate_labels=labels)

        if self.task_type == "translation":
            src = kwargs.get("src_lang")
            tgt = kwargs.get("tgt_lang")
            if src and tgt:
                return self._pipe(main_input, src_lang=src, tgt_lang=tgt)
            return self._pipe(main_input)

        return self._pipe(main_input, **{k: v for k, v in kwargs.items() if v is not None})
