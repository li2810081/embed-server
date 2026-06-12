# embed-server 🚀

**OpenAI 兼容的 Embedding 服务** — 用 **fastembed** 替代 Ollama，CPU 运行，极轻量。

作为 **Hermes Agent 插件** 使用，也支持独立运行。

---

## ✨ 特性

- ✅ **OpenAI 兼容** — `POST /v1/embeddings` 完全对齐 OpenAI API
- ✅ **Ollama 兼容** — `POST /api/embed` 可作为替代
- ✅ **768 维向量** — 最大兼容性，支持主流 RAG 框架
- ✅ **CPU 运行** — 无需 GPU，ONNX Runtime 优化
- ✅ **模型内置** — nomic-embed-text-v1.5 量化版 ~135MB
- ✅ **Hermes 插件** — 自动管理生命周期（启动/停止/监控）
- ✅ **独立运行** — `python server.py` 即可
- ✅ **镜像加速** — 国内用户自动使用 hf-mirror.com

## 📦 安装

### 作为 Hermes 插件

```bash
hermes plugins install https://github.com/li2810081/embed-server.git
hermes plugins enable embed-server
hermes gateway restart
```

### 独立运行

```bash
pip install fastembed fastapi uvicorn
python server.py --port 11434
```

## 🏗️ 项目结构

```
embed-server/
├── pyproject.toml              # Python 包配置
├── src/embed_server/
│   ├── __init__.py
│   ├── cli.py                  # CLI 入口 (install/doctor/init/start)
│   ├── server.py               # FastAPI Embedding 服务
│   ├── process_manager.py      # 进程生命周期管理
│   └── plugin/                 # Hermes 插件目录
│       ├── plugin.yaml         # Hermes 插件清单
│       ├── __init__.py         # 插件钩子 + /embed 命令
│       ├── process_manager.py  # 子进程管理
│       ├── server.py           # 插件内服务
│       └── model/              # 内置模型 (~135MB, LFS)
├── README.md                   # 英文文档
├── README.zh.md                # 中文文档 (本文件)
└── .gitattributes              # Git LFS 配置
```

## 🚀 快速安装

```bash
# 1. 安装包
pip install embed-server

# 2. 激活到 Hermes（自动 symlink + 装依赖）
embed-server install

# 3. 体检
embed-server doctor

# 4. 重启网关
hermes gateway restart
```

## 🔧 CLI 命令

| 命令 | 说明 |
|------|------|
| `embed-server install` | 激活插件到 Hermes |
| `embed-server doctor`  | 完整诊断（依赖/模型/端口/环境） |
| `embed-server init`    | 交互式安装向导 |
| `embed-server start`   | 独立运行服务 |

## 📡 API 接口

### POST /v1/embeddings

```bash
curl http://localhost:11434/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input":"你好世界","model":"nomic-ai/nomic-embed-text-v1.5"}'
```

响应:
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

## 🔧 Hermes 命令

| 命令 | 说明 |
|------|------|
| `/embed status` | 查看服务状态 |
| `/embed restart` | 重启服务 |
| `/embed stop` | 停止服务 |
| `/embed logs [N]` | 查看最近 N 行日志 |

## 🧠 模型说明

内置模型: [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- **向量维度**: 768
- **模型大小**: ~135MB (ONNX 量化版)
- **缓存路径**: 由 `FASTEMBED_CACHE_PATH` 指定，默认使用项目内 `model/` 目录

### 切换模型

```bash
EMBED_SERVER_MODEL=BAAI/bge-small-en-v1.5 python server.py
```

支持的模型列表:
```
nomic-ai/nomic-embed-text-v1.5    (768维, 推荐)
BAAI/bge-small-en-v1.5            (384维, 最快)
BAAI/bge-base-en-v1.5             (768维)
BAAI/bge-large-en-v1.5            (1024维)
jinaai/jina-embeddings-v2-base-zh (768维, 中文优化)
```

## ⚙️ 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `EMBED_SERVER_PORT` | 11434 | 服务端口 |
| `EMBED_SERVER_HOST` | 0.0.0.0 | 绑定地址 |
| `EMBED_SERVER_MODEL` | nomic-ai/nomic-embed-text-v1.5 | 模型名称 |
| `FASTEMBED_CACHE_PATH` | ./model/ | 模型缓存目录 |
| `HF_ENDPOINT` | https://hf-mirror.com | HuggingFace 镜像 |

## 📦 依赖

- [fastembed](https://github.com/qdrant/fastembed) — CPU 优化的 Embedding 引擎
- [FastAPI](https://fastapi.tiangolo.com/) + [uvicorn](https://www.uvicorn.org/) — HTTP 服务
- [onnxruntime](https://onnxruntime.ai/) — ONNX 推理

## 📝 许可证

MIT
