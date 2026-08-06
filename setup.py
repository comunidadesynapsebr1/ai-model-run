from setuptools import setup, find_packages

setup(
    name="ai-runner",
    version="0.1.0",
    description="Baixe e rode modelos de IA (ou agentes) com facilidade.",
    packages=find_packages(),
    install_requires=[
        "huggingface_hub>=0.24.0",
        "transformers>=4.40.0",
        "torch>=2.2.0",
        "fastapi>=0.110.0",
        "uvicorn>=0.29.0",
        "pydantic>=2.6.0",
        "diffusers>=0.27.0",
        "accelerate>=0.30.0",
        "pillow>=10.0.0",
        "soundfile>=0.12.0",
        "sentencepiece>=0.2.0",
    ],
    entry_points={
        "console_scripts": [
            "ai-runner=ai_runner.cli:main",
        ],
    },
    python_requires=">=3.9",
)
