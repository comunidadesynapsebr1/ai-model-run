"""
Agente de IA simples, estilo ReAct (Reason + Act), usando um modelo
local carregado pelo AI Runner. O modelo decide qual "ferramenta"
usar escrevendo linhas no formato:

    Thought: <raciocínio>
    Action: <nome_da_ferramenta>[<entrada>]

O agente executa a ferramenta, devolve o resultado como "Observation"
e repete até o modelo responder com "Final Answer: <resposta>".

Ferramentas incluídas por padrão: calculator. Novas ferramentas podem
ser registradas com @tool ou passadas no dicionário `tools`.
"""
import re
from typing import Callable, Dict, Optional

from .runner import ModelRunner

TOOL_REGISTRY: Dict[str, Callable[[str], str]] = {}


def tool(name: str):
    """Decorator para registrar uma função como ferramenta do agente."""
    def wrapper(fn):
        TOOL_REGISTRY[name] = fn
        return fn
    return wrapper


@tool("calculator")
def _calculator(expression: str) -> str:
    """Avalia uma expressão matemática simples de forma seca."""
    allowed = "0123456789+-*/(). "
    if not all(c in allowed for c in expression):
        return "Erro: expressão contém caracteres não permitidos."
    try:
        # eval restrito, só aritmética básica
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Erro ao calcular: {e}"


SYSTEM_PROMPT_TEMPLATE = """Você é um agente de IA que resolve tarefas passo a passo.
Você tem acesso às seguintes ferramentas: {tool_names}.

Use exatamente este formato:
Thought: seu raciocínio sobre o que fazer
Action: nome_da_ferramenta[entrada da ferramenta]
Observation: (preenchido automaticamente com o resultado)
... (repita Thought/Action/Observation quantas vezes precisar)
Thought: agora eu sei a resposta final
Final Answer: a resposta final para o usuário

Tarefa: {task}
"""


class Agent:
    def __init__(self, model_id: str, tools: Optional[Dict[str, Callable]] = None,
                 device: str = "auto", max_steps: int = 6):
        self.runner = ModelRunner(model_id, device=device)
        self.tools = tools or dict(TOOL_REGISTRY)
        self.max_steps = max_steps

    def run(self, task: str, verbose: bool = True) -> str:
        tool_names = ", ".join(self.tools.keys())
        prompt = SYSTEM_PROMPT_TEMPLATE.format(tool_names=tool_names, task=task)

        for step in range(self.max_steps):
            output = self.runner.generate(prompt, max_new_tokens=150, temperature=0.3)
            new_text = output[len(prompt):]

            if verbose:
                print(new_text)

            final_match = re.search(r"Final Answer:\s*(.+)", new_text, re.DOTALL)
            if final_match:
                return final_match.group(1).strip()

            action_match = re.search(r"Action:\s*(\w+)\[(.*?)\]", new_text)
            if action_match:
                tool_name, tool_input = action_match.groups()
                if tool_name in self.tools:
                    observation = self.tools[tool_name](tool_input)
                else:
                    observation = f"Erro: ferramenta '{tool_name}' não existe."

                prompt += new_text.split("Action:")[0]
                prompt += f"Action: {tool_name}[{tool_input}]\n"
                prompt += f"Observation: {observation}\n"
            else:
                # Modelo não seguiu o formato — encerra com o que tiver
                return new_text.strip()

        return "Número máximo de passos atingido sem resposta final."
