# Embrace AI Agent

A real-time voice chat application powered by ollama models. This project allows you to have voice conversations with AI models like mistral-nemo:12b running locally on your machine.
The Agent is tailored for Emotional therapy conversation.

## Features

- Real-time speech-to-text conversion
- Local LLM inference using Ollama
- Text-to-speech response generation
- Web interface for interaction

## Prerequisites

On your machine please install.
- [Ollama](https://ollama.ai/) - Run LLMs locally
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver

## Installation

### 1. Install prerequisites with Homebrew

- Install ollama and uv using pip

```bash
pip install ollama
pip install uv
```

### 2. Clone the repository

```bash
git clone https://github.com/jesuscopado/local-voice-ai-agent.git
cd local-voice-ai-agent
```

### 3. Set up Python environment and install dependencies

```bash
uv venv
source .venv/bin/activate 
uv sync
```

for windows 
```bash
uv venv
cd venv/Scripts/activate 
uv sync
```

### 4. Download required models in Ollama

```bash
ollama pull mistral-nemo:12b
```

## Usage


#### Web UI (default)
```bash
python local_voice_chat_advanced.py
```

## How it works

The application uses:
- `FastRTC` for WebRTC communication
- `Moonshine` for local speech-to-text conversion
- `Kokoro` for text-to-speech synthesis
- `Ollama` for running local LLM inference with `mistral-nemo:12b` model

When you speak, your audio is:
1. Transcribed to text using Moonshine
2. Sent to a local LLM via Ollama for processing
3. The LLM response is converted back to speech with Kokoro
4. The audio response is streamed back to you via FastRTC
