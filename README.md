# AI Runner

Ferramenta de linha de comando (CLI) para **baixar e rodar modelos de IA
com facilidade** — sejam modelos de texto puros ou agentes de IA que usam
ferramentas. Funciona com qualquer modelo público do Hugging Face Hub.

## O que ela faz

- **`pull`** — baixa um modelo do Hugging Face Hub para sua máquina
- **`list`** — lista os modelos já baixados
- **`run`** — roda inferência com o modelo, para várias tarefas (veja abaixo)
- **`image`** — gera imagens a partir de texto (text-to-image)
- **`serve`** — sobe uma API HTTP local (FastAPI) para o modelo, em qualquer tarefa
- **`agent`** — roda um agente estilo ReAct (raciocina + usa ferramentas, ex: calculadora)
- **`remove`** — apaga um modelo baixado do disco

Tudo fica organizado em `~/.ai_runner/` (modelos, registro, logs).

### Tarefas suportadas em `run` (via `--type`)

| `--type`                        | O que faz                              | Entrada principal |
|----------------------------------|------------------------------------------|--------------------|
| `text-generation` (default)      | continuar/completar um texto             | `--prompt`         |
| `summarization`                  | resumir um texto                         | `--text`           |
| `translation`                    | traduzir (opcional `--src-lang`/`--tgt-lang`) | `--text`       |
| `text-classification`            | classificar texto (ex: sentimento)       | `--text`           |
| `zero-shot-classification`       | classificar em categorias livres         | `--text --labels`  |
| `question-answering`             | responder pergunta com base em contexto  | `--question --context` |
| `fill-mask`                      | preencher `[MASK]` em uma frase          | `--text`           |
| `feature-extraction`             | gerar embeddings (vetores) de um texto   | `--text`           |
| `automatic-speech-recognition`   | transcrever áudio para texto             | `--audio-file`     |
| `image-classification`           | classificar uma imagem                   | `--image-file`     |
| `image-to-text`                  | gerar legenda/descrição de uma imagem    | `--image-file`     |

Para geração de imagens (text-to-image, ex: Stable Diffusion), use o comando `image` em vez de `run`.

## Instalação

### Opção 1: script automático (Linux/Mac)
```bash
chmod +x install.sh
./install.sh
source .venv/bin/activate
```

### Opção 2: manual
```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

Requer Python 3.9+. A primeira instalação baixa `torch` e `transformers`,
que são pacotes grandes — pode demorar alguns minutos.

## Uso rápido

```bash
# 1. Baixar um modelo pequeno para testar
ai-runner pull gpt2

# 2. Ver os modelos baixados
ai-runner list

# 3. Rodar uma geração de texto
ai-runner run gpt2 --prompt "Era uma vez" --max-tokens 100

# 4. Subir uma API local
ai-runner serve gpt2 --port 8000
# depois: curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d '{"prompt": "Olá"}'

# 5. Rodar um agente que usa a ferramenta de calculadora
ai-runner agent gpt2 --task "Quanto é 23 * 7 + 10?"

# 6. Remover um modelo do disco
ai-runner remove gpt2
```

## Outras tarefas além de geração de texto

```bash
# Resumir um texto
ai-runner pull facebook/bart-large-cnn --type summarization
ai-runner run facebook/bart-large-cnn --type summarization --text "texto bem longo aqui..."

# Classificação de sentimento
ai-runner pull distilbert-base-uncased-finetuned-sst-2-english --type text-classification
ai-runner run distilbert-base-uncased-finetuned-sst-2-english --type text-classification --text "eu amei esse produto!"

# Classificação zero-shot (categorias livres, sem treinar nada)
ai-runner pull facebook/bart-large-mnli --type zero-shot-classification
ai-runner run facebook/bart-large-mnli --type zero-shot-classification --text "o time perdeu de virada" --labels "esporte,política,economia"

# Perguntas e respostas (Q&A)
ai-runner pull deepset/roberta-base-squad2 --type question-answering
ai-runner run deepset/roberta-base-squad2 --type question-answering --question "Quem escreveu Dom Casmurro?" --context "Dom Casmurro foi escrito por Machado de Assis em 1899."

# Embeddings (vetores para busca semântica / RAG)
ai-runner pull sentence-transformers/all-MiniLM-L6-v2 --type feature-extraction
ai-runner run sentence-transformers/all-MiniLM-L6-v2 --type feature-extraction --text "frase de exemplo"

# Transcrição de áudio (fala -> texto)
ai-runner pull openai/whisper-small --type automatic-speech-recognition
ai-runner run openai/whisper-small --type automatic-speech-recognition --audio-file audio.wav

# Geração de imagem (texto -> imagem)
ai-runner pull runwayml/stable-diffusion-v1-5 --type text-to-image
ai-runner image runwayml/stable-diffusion-v1-5 --prompt "um gato astronauta, estilo pintura a óleo" --output gato.png --steps 30
```

Todas essas tarefas também funcionam via `ai-runner serve <modelo> --type <tarefa>`,
que expõe os endpoints `/predict` (texto/classificação/áudio/QA) ou
`/generate-image` (imagem), além de `/generate` (geração livre de texto).

## Baixando apenas alguns arquivos (economizar espaço)

```bash
ai-runner pull mistralai/Mistral-7B-Instruct-v0.3 --include "*.safetensors,*.json,*.model"
```

## Modelos sugeridos para começar (leves, rodam em CPU)

Veja `examples/models.yaml` para uma lista com `gpt2`, `Qwen2.5-0.5B-Instruct`,
`Phi-3-mini-4k-instruct`, entre outros.

## Rodando com GPU

Se você tiver uma GPU NVIDIA com CUDA instalado, o AI Runner detecta
automaticamente (`--device auto`, que é o padrão). Para forçar CPU:
```bash
ai-runner run gpt2 --prompt "teste" --device cpu
```

## Criando suas próprias ferramentas de agente

Edite `ai_runner/agent.py` e registre novas ferramentas com o decorator:

```python
from ai_runner.agent import tool

@tool("minha_ferramenta")
def minha_ferramenta(entrada: str) -> str:
    return f"processei: {entrada}"
```

O modelo poderá então chamar `Action: minha_ferramenta[algo]` durante o
raciocínio do agente.

## Estrutura do projeto

```
ai-model-runner/
├── ai_runner/
│   ├── cli.py         # comandos da CLI
│   ├── downloader.py  # baixa modelos do Hugging Face
│   ├── runner.py       # carrega e roda inferência
│   ├── server.py       # API HTTP (FastAPI)
│   ├── agent.py        # loop de agente ReAct + ferramentas
│   ├── registry.py     # registro local dos modelos baixados
│   └── config.py        # caminhos e configuração
├── examples/models.yaml # lista de modelos sugeridos
├── install.sh
├── requirements.txt
├── setup.py
└── README.md
```

## Limitações e próximos passos

- Focado em modelos de **texto** (causal LM) via `transformers`. Para
  imagem/áudio (ex: modelos de difusão, GANs como o Synapse-GAN-Tiny),
  o `pull`/`list`/`remove` funcionam normalmente, mas `run`/`serve`
  precisam de um loader específico — dá pra adaptar `runner.py` seguindo
  o mesmo padrão.
- Para rodar modelos em formato **GGUF** (quantizados, ótimos para CPU),
  é possível trocar o backend do `runner.py` para `llama-cpp-python`.
- Sem autenticação/multiusuário — pensado para uso local, em uma máquina.
