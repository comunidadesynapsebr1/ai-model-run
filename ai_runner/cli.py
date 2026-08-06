"""
AI Runner CLI — baixe e rode modelos de IA (ou agentes) com facilidade.

Exemplos:
  ai-runner pull gpt2
  ai-runner list
  ai-runner run gpt2 --prompt "Era uma vez" --max-tokens 100
  ai-runner run facebook/bart-large-cnn --type summarization --text "texto longo..."
  ai-runner run distilbert-base-uncased-finetuned-sst-2-english --type text-classification --text "eu amei!"
  ai-runner run sentence-transformers/all-MiniLM-L6-v2 --type feature-extraction --text "frase para embedding"
  ai-runner run openai/whisper-small --type automatic-speech-recognition --audio-file fala.wav
  ai-runner image runwayml/stable-diffusion-v1-5 --prompt "um gato astronauta" --output gato.png
  ai-runner serve gpt2 --port 8000
  ai-runner agent gpt2 --task "Quanto é 23 * 7 + 10?"
  ai-runner remove gpt2
"""
import argparse
import sys

from .config import ensure_dirs
from .downloader import pull_model, remove_local_model
from .registry import list_models
from .pipelines import SUPPORTED_TASKS


def cmd_pull(args):
    patterns = args.include.split(",") if args.include else None
    pull_model(args.model_id, revision=args.revision, allow_patterns=patterns,
                model_type=args.type)


def cmd_list(args):
    models = list_models()
    if not models:
        print("Nenhum modelo baixado ainda. Use: ai-runner pull <model_id>")
        return
    print(f"{'MODEL ID':40} {'TIPO':10} {'BAIXADO EM'}")
    for model_id, info in models.items():
        print(f"{model_id:40} {info.get('type', 'auto'):10} {info.get('downloaded_at', '')}")


def cmd_run(args):
    if args.type == "text-generation" and args.prompt and not args.text:
        # Caminho rápido: geração livre de texto (causal LM), como antes.
        from .runner import ModelRunner
        runner = ModelRunner(args.model_id, device=args.device)
        output = runner.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=args.seed,
        )
        print("\n--- Saída ---")
        print(output)
        return

    # Caminho genérico: qualquer outra tarefa suportada via pipeline.
    from .pipelines import PipelineRunner
    runner = PipelineRunner(args.model_id, task_type=args.type, device=args.device)

    main_input = args.text or args.audio_file or args.image_file or args.prompt
    if main_input is None:
        print("Erro: forneça --text, --prompt, --audio-file ou --image-file conforme a tarefa.",
              file=sys.stderr)
        sys.exit(1)

    labels = args.labels.split(",") if args.labels else None
    result = runner.run(
        main_input,
        question=args.question,
        context=args.context,
        candidate_labels=labels,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
    )
    print("\n--- Resultado ---")
    print(result)


def cmd_image(args):
    from .image_gen import ImageRunner
    runner = ImageRunner(args.model_id, device=args.device)
    runner.generate(
        prompt=args.prompt,
        output_path=args.output,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )


def cmd_serve(args):
    from .server import run_server
    run_server(args.model_id, host=args.host, port=args.port, device=args.device,
               task_type=args.type)


def cmd_agent(args):
    from .agent import Agent
    agent = Agent(args.model_id, device=args.device, max_steps=args.max_steps)
    answer = agent.run(args.task)
    print("\n--- Resposta Final ---")
    print(answer)


def cmd_remove(args):
    remove_local_model(args.model_id)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="ai-runner",
        description="Baixe e rode modelos de IA (ou agentes) com facilidade.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_pull = sub.add_parser("pull", help="Baixa um modelo do Hugging Face Hub")
    p_pull.add_argument("model_id", help="ex: gpt2, meta-llama/Llama-3.2-1B-Instruct")
    p_pull.add_argument("--revision", default="main", help="branch/tag/commit (default: main)")
    p_pull.add_argument("--include", default=None,
                         help="padrões glob separados por vírgula, ex: '*.safetensors,*.json'")
    p_pull.add_argument("--type", default="text-generation",
                         choices=SUPPORTED_TASKS + ["text-to-image", "auto"],
                         help="tipo de tarefa do modelo (default: text-generation)")
    p_pull.set_defaults(func=cmd_pull)

    p_list = sub.add_parser("list", help="Lista modelos baixados localmente")
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser(
        "run", help="Roda inferência com um modelo local (texto, resumo, tradução, "
                     "classificação, embeddings, áudio, imagem, etc.)")
    p_run.add_argument("model_id")
    p_run.add_argument("--type", default="text-generation", choices=SUPPORTED_TASKS,
                        help="tipo de tarefa (default: text-generation)")
    p_run.add_argument("--prompt", default=None, help="usado em text-generation")
    p_run.add_argument("--text", default=None,
                        help="texto de entrada para summarization, translation, "
                             "classification, embeddings, fill-mask, etc.")
    p_run.add_argument("--audio-file", default=None, dest="audio_file",
                        help="caminho de arquivo .wav/.mp3 para automatic-speech-recognition")
    p_run.add_argument("--image-file", default=None, dest="image_file",
                        help="caminho/URL de imagem para image-classification/image-to-text")
    p_run.add_argument("--question", default=None, help="pergunta, para question-answering")
    p_run.add_argument("--context", default=None,
                        help="contexto, para question-answering (default: usa --text)")
    p_run.add_argument("--labels", default=None,
                        help="categorias separadas por vírgula, para zero-shot-classification")
    p_run.add_argument("--src-lang", default=None, dest="src_lang", help="idioma de origem (tradução)")
    p_run.add_argument("--tgt-lang", default=None, dest="tgt_lang", help="idioma de destino (tradução)")
    p_run.add_argument("--max-tokens", type=int, default=200, dest="max_tokens")
    p_run.add_argument("--temperature", type=float, default=0.7)
    p_run.add_argument("--seed", type=int, default=None)
    p_run.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', 'mps'")
    p_run.set_defaults(func=cmd_run)

    p_image = sub.add_parser("image", help="Gera uma imagem a partir de texto (text-to-image)")
    p_image.add_argument("model_id")
    p_image.add_argument("--prompt", required=True)
    p_image.add_argument("--negative-prompt", default=None, dest="negative_prompt")
    p_image.add_argument("--output", default="output.png")
    p_image.add_argument("--steps", type=int, default=30)
    p_image.add_argument("--guidance-scale", type=float, default=7.5, dest="guidance_scale")
    p_image.add_argument("--width", type=int, default=512)
    p_image.add_argument("--height", type=int, default=512)
    p_image.add_argument("--seed", type=int, default=None)
    p_image.add_argument("--device", default="auto")
    p_image.set_defaults(func=cmd_image)

    p_serve = sub.add_parser("serve", help="Sobe uma API HTTP local para o modelo")
    p_serve.add_argument("model_id")
    p_serve.add_argument("--type", default="text-generation",
                          choices=SUPPORTED_TASKS + ["text-to-image"],
                          help="tipo de tarefa que a API vai servir (default: text-generation)")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--device", default="auto")
    p_serve.set_defaults(func=cmd_serve)

    p_agent = sub.add_parser("agent", help="Roda um agente ReAct simples com o modelo")
    p_agent.add_argument("model_id")
    p_agent.add_argument("--task", required=True)
    p_agent.add_argument("--max-steps", type=int, default=6, dest="max_steps")
    p_agent.add_argument("--device", default="auto")
    p_agent.set_defaults(func=cmd_agent)

    p_remove = sub.add_parser("remove", help="Remove um modelo baixado localmente")
    p_remove.add_argument("model_id")
    p_remove.set_defaults(func=cmd_remove)

    return parser


def main():
    ensure_dirs()
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as e:
        print(f"Erro: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
