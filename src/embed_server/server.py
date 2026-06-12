"""
FastEmbed OpenAI-compatible embedding server
Drop-in replacement for Ollama embedding API on port 11434
"""
import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import List, Union

# Set HF mirror for model downloads
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# Set fastembed cache path
if not os.environ.get("FASTEMBED_CACHE_PATH"):
    os.environ["FASTEMBED_CACHE_PATH"] = str(Path.home() / "AppData" / "Local" / "Temp" / "fastembed_cache")

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
LOG_DIR = Path(os.environ.get("EMBED_LOG_DIR", str(Path.home() / ".embed-server")))
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "server.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("embed-server")

# ---------------------------------------------------------------------------
# Pydantic models for OpenAI-compatible API
# ---------------------------------------------------------------------------

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = "nomic-ai/nomic-embed-text-v1.5"
    encoding_format: str = "float"

class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]

class UsageInfo(BaseModel):
    prompt_tokens: int
    total_tokens: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: UsageInfo

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "fastembed"

class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(title="FastEmbed Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded model
_model = None
_model_name = None
_loaded_model_spec = None

# Canonical model name for nomic-embed-text variants
_CANONICAL_MODEL = "nomic-ai/nomic-embed-text-v1.5"


def resolve_model(model_name: str) -> str:
    """Route model names to canonical names.

    Any model starting with ``nomic-embed-text`` maps to
    ``nomic-ai/nomic-embed-text-v1.5`` for backward compatibility.
    """
    if model_name.startswith("nomic-embed-text"):
        resolved = _CANONICAL_MODEL
        if model_name != resolved:
            logger.info("Model routing: %s -> %s", model_name, resolved)
        return resolved
    return model_name


def get_model(model_name: str):
    """Lazy-load the embedding model on first request."""
    global _model, _model_name, _loaded_model_spec
    canonical = resolve_model(model_name)
    if _loaded_model_spec != canonical:
        from fastembed import TextEmbedding
        logger.info(f"Loading model: {canonical}")
        t0 = time.time()
        try:
            _model = TextEmbedding(model_name=canonical, local_files_only=False)
        except Exception as e:
            logger.warning(f"Failed with local_files_only=False: {e}")
            _model = TextEmbedding(model_name=canonical, local_files_only=True)
        t1 = time.time()
        _model_name = canonical
        _loaded_model_spec = canonical
        logger.info(f"Model loaded in {t1-t0:.2f}s, dimension={_model.length}")
    return _model


def count_tokens(text: str) -> int:
    """Rough token count."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "fastembed-server",
        "model": _loaded_model_spec or "not loaded",
    }


@app.get("/v1/models")
async def list_models():
    now = int(time.time())
    return ModelsResponse(data=[
        ModelInfo(id=_loaded_model_spec or "nomic-ai/nomic-embed-text-v1.5", created=now),
    ])


@app.post("/v1/embeddings")
async def create_embeddings(req: EmbeddingRequest):
    """OpenAI-compatible embeddings endpoint."""
    resolved = resolve_model(req.model)
    model = get_model(resolved)

    if isinstance(req.input, str):
        texts = [req.input]
    else:
        texts = req.input

    if not texts:
        raise HTTPException(status_code=400, detail="input cannot be empty")
    if len(texts) > 128:
        raise HTTPException(status_code=400, detail="max 128 inputs per request")

    logger.info(f"Embedding {len(texts)} text(s) with {req.model}")
    t0 = time.time()
    results = list(model.embed(texts))
    t1 = time.time()

    embeddings_list = [r.tolist() for r in results]
    total_tokens = sum(count_tokens(t) for t in texts)

    logger.info(f"Done in {t1-t0:.3f}s, {len(texts)} texts, dim={len(embeddings_list[0]) if embeddings_list else 0}")

    return EmbeddingResponse(
        data=[
            EmbeddingData(index=i, embedding=emb)
            for i, emb in enumerate(embeddings_list)
        ],
        model=req.model,
        usage=UsageInfo(prompt_tokens=total_tokens, total_tokens=total_tokens),
    )


# Ollama-compatible /api/embed endpoint
@app.post("/api/embed")
@app.post("/api/embeddings")
async def ollama_compat_embed(request: Request):
    body = await request.json()
    raw_model = body.get("model", "nomic-ai/nomic-embed-text-v1.5")
    model_name = resolve_model(raw_model)
    prompt = body.get("input") or body.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="input is required")

    emb_model = get_model(model_name)

    if isinstance(prompt, str):
        texts = [prompt]
    else:
        texts = prompt

    t0 = time.time()
    results = list(emb_model.embed(texts))
    t1 = time.time()

    logger.info(f"Ollama-compat embed: {len(texts)} texts in {t1-t0:.3f}s")

    return {
        "model": model_name,
        "embeddings": [r.tolist() for r in results],
        "total_duration": int((t1 - t0) * 1e9),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=None, help="Bind address")
    parser.add_argument("--port", type=int, default=None, help="Port")
    parser.add_argument("--model", default=None, help="FastEmbed model name")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    host = args.host or os.environ.get("EMBED_SERVER_HOST", "0.0.0.0")
    port = args.port or int(os.environ.get("EMBED_SERVER_PORT", "11434"))
    model = args.model or os.environ.get("EMBED_SERVER_MODEL", "nomic-ai/nomic-embed-text-v1.5")

    # Pre-load model
    logger.info(f"Pre-loading model: {model}")
    try:
        get_model(model)
    except Exception as e:
        logger.error(f"Failed to load model '{model}': {e}")
        fallback = "nomic-ai/nomic-embed-text-v1.5"
        logger.info(f"Falling back to: {fallback}")
        get_model(fallback)

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, workers=args.workers, log_level="info")


if __name__ == "__main__":
    main()
