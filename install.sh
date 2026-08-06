#!/usr/bin/env bash
# Instala o AI Runner e suas dependências.
set -e

echo "Criando ambiente virtual (venv)..."
python3 -m venv .venv
source .venv/bin/activate

echo "Instalando dependências..."
pip install --upgrade pip
pip install -e .

echo ""
echo "Instalação concluída!"
echo "Ative o ambiente com: source .venv/bin/activate"
echo "Depois use: ai-runner --help"
