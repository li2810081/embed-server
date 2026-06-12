# embed-server 🚀

**OpenAI-compatible embedding server** — a drop-in replacement for Ollama's `/v1/embeddings` endpoint using **fastembed** (CPU, lightweight).

Built as a **Hermes Agent plugin** but can also run standalone.

> **🌐 Languages**: [English](README.md) · [中文](README.zh.md)

## ✨ Features

- ✅ OpenAI-compatible `/v1/embeddings` API
- ✅ Ollama-compatible `/api/embed` fallback
- ✅ **768-dimension** embeddings (max compatibility)
- ✅ CPU-only — no GPU required
- ✅ ~135MB model bundled (nomic-embed-text-v1.5 quantized)
- ✅ Hermes Agent plugin: auto-managed lifecycle
- ✅ Standalone mode: `python server.py`
- ✅ **Model alias routing**: `nomic-embed-text*` → `nomic-ai/nomic-embed-text-v1.5`

## 🏗️ Project Structure

```
embed-server/
├── plugin.yaml              # Hermes plugin manifest
├── __init__.py               # Plugin hooks + /embed command
├── process_manager.py        # Subprocess lifecycle management
├── server.py                 # FastAPI embedding server
├── model/                    # Bundled model (fastembed cache)
│   └── models--nomic-ai--nomic-embed-text-v1.5/
│       └── snapshots/
│           └── e9b67630.../
│               ├── config.json
│               ├── tokenizer.json
│               ├── tokenizer_config.json
│               ├── special_tokens_map.json
│               └── onnx/
│                   └── model.onnx          (131MB, LFS)
└── README.md
└── README.zh.md              # 中文文档 (Chinese)
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install fastembed fastapi uvicorn
```

### Standalone (no Hermes)

```bash
python server.py --port 11434
```

### As Hermes Plugin

```bash
hermes plugins install https://github.com/li2810081/embed-server.git
hermes plugins enable embed-server
hermes gateway restart
```

## 📡 API

### POST /v1/embeddings

```bash
curl http://localhost:11434/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello world","model":"nomic-ai/nomic-embed-text-v1.5"}'
```

Response:
```json
{
  "object": "list",
  "data": [{"object": "embedding", "index": 0, "embedding": [...]}],
  "model": "nomic-ai/nomic-embed-text-v1.5",
  "usage": {"prompt_tokens": 2, "total_tokens": 2}
}
```

### GET /v1/models

```bash
curl http://localhost:11434/v1/models
```

### GET /health

```bash
curl http://localhost:11434/health
```

## 🔧 Hermes Slash Commands

| Command | Description |
|---------|-------------|
| `/embed status` | Show server status |
| `/embed restart` | Restart the server |
| `/embed stop` | Stop the server |
| `/embed logs [N]` | Show last N log lines |

## 🧠 Model

Bundled model: [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- **Dimensions**: 768
- **Quantized**: ONNX `model_quantized` → `model.onnx` (131MB)
- **Cache path**: `FASTEMBED_CACHE_PATH` env var

### Model Alias Routing

Any model name starting with `nomic-embed-text` is automatically routed to
`nomic-ai/nomic-embed-text-v1.5`. This means you can use any of these names:

```
nomic-embed-text              → nomic-ai/nomic-embed-text-v1.5
nomic-embed-text-v1.5         → nomic-ai/nomic-embed-text-v1.5
nomic-embed-text-v1           → nomic-ai/nomic-embed-text-v1.5
```

This ensures backward compatibility with Ollama configs and scripts that
refer to the model by its short name.

To use a different model, set `EMBED_SERVER_MODEL`:
```bash
EMBED_SERVER_MODEL=BAAI/bge-small-en-v1.5 python server.py
```

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_SERVER_PORT` | 11434 | Server port |
| `EMBED_SERVER_HOST` | 0.0.0.0 | Bind address |
| `EMBED_SERVER_MODEL` | nomic-ai/nomic-embed-text-v1.5 | Model name |
| `FASTEMBED_CACHE_PATH` | ./model/ | Model cache directory |
| `HF_ENDPOINT` | https://hf-mirror.com | HuggingFace mirror |

## 📦 Dependencies

- [fastembed](https://github.com/qdrant/fastembed) — CPU-optimized embedding engine
- [FastAPI](https://fastapi.tiangolo.com/) + [uvicorn](https://www.uvicorn.org/) — HTTP server
- [onnxruntime](https://onnxruntime.ai/) — ONNX inference
