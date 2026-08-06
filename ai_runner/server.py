"""
Servidor HTTP simples (FastAPI) para expor um modelo local baixado
com o AI Runner. Útil para usar o modelo como backend de um agente,
de um app, ou testar via curl/Postman.

Endpoints:
  GET  /health          -> status do servidor
  GET  /model-info       -> metadados do modelo carregado
  POST /generate         -> geração livre de texto (causal LM)
                             { "prompt": "...", "max_new_tokens": 200, "seed": 42 }
  POST /predict           -> qualquer outra tarefa (summarization, translation,
                             classification, embeddings, QA, ASR, etc.), usando
                             o mesmo conjunto de tarefas do comando `ai-runner run`
                             { "input": "...", "question": "...", "context": "...",
                               "candidate_labels": [...], "src_lang": "...", "tgt_lang": "..." }
POST /generate-image      -> gera uma imagem e retorna em base64 (text-to-image)
                             { "prompt": "...", "steps": 30, "width": 512, "height": 512 }
"""
from typing import List, Optional

from .runner import ModelRunner
from .pipelines import PipelineRunner, SUPPORTED_TASKS

_runner: Optional[ModelRunner] = None
_pipeline_runner: Optional[PipelineRunner] = None
_image_runner = None
_model_id = None
_task_type = "text-generation"


def create_app(model_id: str, device: str = "auto", task_type: str = "text-generation"):
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except ImportError as e:
        raise RuntimeError(
            "Pacotes 'fastapi' e 'pydantic' não instalados. "
            "Rode: pip install -r requirements.txt"
        ) from e

    global _runner, _pipeline_runner, _image_runner, _model_id, _task_type
    _model_id = model_id
    _task_type = task_type

    if task_type == "text-generation":
        _runner = ModelRunner(model_id, device=device)
    elif task_type == "text-to-image":
        from .image_gen import ImageRunner
        _image_runner = ImageRunner(model_id, device=device)
    else:
        _pipeline_runner = PipelineRunner(model_id, task_type=task_type, device=device)

    app = FastAPI(title="AI Runner", description="Servidor local de modelos de IA")

    class GenerateRequest(BaseModel):
        prompt: str
        max_new_tokens: int = 200
        temperature: float = 0.7
        top_p: float = 0.9
        seed: Optional[int] = None

    class PredictRequest(BaseModel):
        input: str
        question: Optional[str] = None
        context: Optional[str] = None
        candidate_labels: Optional[List[str]] = None
        src_lang: Optional[str] = None
        tgt_lang: Optional[str] = None

    class ImageRequest(BaseModel):
        prompt: str
        negative_prompt: Optional[str] = None
        steps: int = 30
        guidance_scale: float = 7.5
        width: int = 512
        height: int = 512
        seed: Optional[int] = None

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_id, "task_type": task_type}

    @app.get("/model-info")
    def model_info():
        return {
            "model_id": model_id,
            "task_type": task_type,
            "device": device,
            "supported_tasks": SUPPORTED_TASKS + ["text-generation", "text-to-image"],
        }

    @app.post("/generate")
    def generate(req: GenerateRequest):
        if _runner is None:
            return {"error": "Servidor não está no modo text-generation."}
        text = _runner.generate(
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            seed=req.seed,
        )
        return {"prompt": req.prompt, "output": text}

    @app.post("/predict")
    def predict(req: PredictRequest):
        if _pipeline_runner is None:
            return {"error": "Servidor não está configurado para essa tarefa. "
                              "Suba com --type <tarefa> (ver /model-info)."}
        result = _pipeline_runner.run(
            req.input,
            question=req.question,
            context=req.context,
            candidate_labels=req.candidate_labels,
            src_lang=req.src_lang,
            tgt_lang=req.tgt_lang,
        )
        return {"input": req.input, "output": result}

    @app.post("/generate-image")
    def generate_image(req: ImageRequest):
        if _image_runner is None:
            return {"error": "Servidor não está no modo text-to-image."}
        import base64
        out_path = _image_runner.generate(
            prompt=req.prompt,
            output_path="/tmp/_ai_runner_last_image.png",
            negative_prompt=req.negative_prompt,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance_scale,
            width=req.width,
            height=req.height,
            seed=req.seed,
        )
        with open(out_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return {"prompt": req.prompt, "image_base64": b64, "format": "png"}

    return app


def run_server(model_id: str, host: str = "0.0.0.0", port: int = 8000,
                device: str = "auto", task_type: str = "text-generation"):
    try:
        import uvicorn
    except ImportError as e:
        raise RuntimeError(
            "Pacote 'uvicorn' não instalado. "
            "Rode: pip install -r requirements.txt"
        ) from e

    app = create_app(model_id, device=device, task_type=task_type)
    print(f"Servindo modelo '{model_id}' (tarefa: {task_type}) em http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
