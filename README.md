# Hybrid Knowledge Platform KI-4-KMU

This project is part of the [KI-4-KMU initiative](https://www.ki-zentrum.ch/2025/07/03/fhnw-ki-praxisleitfaden-und-ki-canvas-ki-4-kmu-methode/) developed at FHNW, which provides SMEs with practical AI tools and methodologies.

It acts as a **proxy server** that exposes OpenAI-compatible endpoints, automatically injecting retrieved context from the [KI4KMU-IngestionLayer](https://www.google.com/search?q=https://github.com/Heron4gf/KI4KMU-IngestionLayer) into your prompts. This enables document-grounded question answering (RAG) using domain-specific knowledge. The system also launches an integrated [Open WebUI]() container, providing a professional chat interface accessible at `http://localhost:3000` immediately after startup.

## Features

* **OpenAI-compatible API**: Implements `/v1/chat/completions` and `/v1/models` endpoints.
* **Context Injection**: Automatically queries the Ingestion Layer and injects results as a `context` role message.
* **Integrated UI**: Starts a pre-configured Open WebUI instance for a turnkey chat experience.
* **Local Model Support**: Fully compatible with [LM Studio]() for secure, local inference.
* **Streaming support**: Pass-through streaming for real-time, low-latency responses.

## Demo

[https://github.com/user-attachments/assets/demo-video-placeholder]()

---

## Architecture

## Environment Variables

Create a `.env` file based on `.env.example`:

```env
BASE_URL="https://openrouter.ai/api/v1"
API_KEY="your-api-key-here"
RETRIEVAL_URL="http://localhost:8001/v1"

```

| Variable | Description | Default |
| --- | --- | --- |
| `BASE_URL` | Upstream OpenAI-compatible API endpoint | `[https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)` |
| `API_KEY` | API key for authentication | (empty) |
| `RETRIEVAL_URL` | Retrieval engine API endpoint (Ingestion Layer) | `http://localhost:8001/v1` |

## Installation & Running

Due to specific container image requirements and the multi-service architecture (Proxy + Open WebUI), this project must be managed exclusively via **Docker Compose**.

### Local Models (LM Studio)

To use local models, ensure LM Studio is running its Local Server. Update your `.env` to point to the Docker host gateway:
`BASE_URL="[http://host.docker.internal:1234/v1](http://host.docker.internal:1234/v1)"`

### Start the Stack

```bash
docker-compose up --build

```

Once the containers are running:

* **API Proxy**: `http://localhost:8000`
* **Open WebUI**: `http://localhost:3000` (Access this in your browser to start chatting)

---

## API Endpoints

### POST /v1/chat/completions

Create a chat completion with retrieval-augmented context.

**Request:**

```json
{
  "model": "openai/gpt-4o",
  "messages": [
    {"role": "user", "content": "What is the compression ratio of the engine?"}
  ]
}

```

### GET /v1/models

List available models from the upstream API.

## Project Structure

```
app/
├── main.py                 # FastAPI entry point
├── api/
│   ├── chat.py             # Chat completions & Context Injection
│   └── models.py           # Models listing router
├── core/
│   ├── config.py           # Configuration management
│   └── client.py           # API Client
├── models/
│   ├── schemas.py          # API models
│   └── retrieval.py        # Retrieval API models
└── utils/
    └── context_injector.py # Logic for merging context into messages

```

## Testing

```bash
docker-compose run --rm app pytest tests/ -v

```