"""
embed-server Hermes Plugin — fastembed as an OpenAI-compatible API server.

Registers hooks and slash commands so Hermes manages the embedding server's
lifecycle:
- ``on_session_start`` — starts the server if not running
- ``on_session_end`` — no-op (server persists across sessions)
- ``/embed`` — slash command for status, restart, logs, init, doctor, update
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-import the process manager (avoid circular deps at plugin scan time)
# ---------------------------------------------------------------------------

def _pm():
    from . import process_manager as pm
    return pm


# ---------------------------------------------------------------------------
# Requirements check (doctor / init helpers)
# ---------------------------------------------------------------------------

_REQUIRED_PACKAGES: List[str] = [
    "fastembed",
    "fastapi",
    "uvicorn",
    "onnxruntime",
]

def _plugin_dir() -> Path:
    return Path(__file__).parent.resolve()

def _model_cache_dir() -> Path:
    return _plugin_dir() / "model"

def _check_model_files() -> Dict[str, Any]:
    """Check that bundled model files exist and report sizes."""
    snapshot = (
        _model_cache_dir()
        / "models--nomic-ai--nomic-embed-text-v1.5"
        / "snapshots"
        / "e9b6763023c676ca8431644204f50c2b100d9aab"
    )
    expected = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "onnx/model.onnx",
    ]
    results = []
    all_ok = True
    for rel in expected:
        p = snapshot / rel
        if p.exists():
            results.append(f"  ✅ {rel} ({p.stat().st_size / 1024 / 1024:.1f}MB)")
        else:
            results.append(f"  ❌ {rel} — missing")
            all_ok = False
    return {"ok": all_ok, "path": str(snapshot), "files": results}


def _check_python_packages() -> Dict[str, Any]:
    """Check required Python packages are importable."""
    results = []
    all_ok = True
    for pkg in _REQUIRED_PACKAGES:
        try:
            mod = __import__(pkg.replace("-", "_"))
            ver = getattr(mod, "__version__", "unknown")
            results.append(f"  ✅ {pkg} == {ver}")
        except ImportError:
            results.append(f"  ❌ {pkg} — not installed")
            all_ok = False
    return {"ok": all_ok, "packages": results}


def _check_port(port: int = 11434) -> Dict[str, Any]:
    """Check if the server port is available."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("0.0.0.0", port))
        s.close()
        return {"ok": True, "port": port, "in_use": False}
    except OSError:
        return {"ok": False, "port": port, "in_use": True}


def _check_hermes_env() -> Dict[str, Any]:
    """Check Hermes environment compatibility."""
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    py_ver = sys.version
    return {
        "ok": True,
        "hermes_home": str(home),
        "python": py_ver,
    }


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

def _on_session_start(**kwargs) -> None:
    """Start embedding server on first session if not already running."""
    pm = _pm()
    status = pm.status()
    if status.get("ok") and status.get("alive"):
        logger.info("embed-server: already running (pid=%s)", status.get("pid"))
        return

    port = int(os.environ.get("EMBED_SERVER_PORT", "11434"))
    model = os.environ.get("EMBED_SERVER_MODEL", "nomic-ai/nomic-embed-text-v1.5")

    logger.info("embed-server: starting on port %s with model %s", port, model)
    result = pm.start(port=port, model=model)

    if result.get("ok"):
        logger.info("embed-server: started (pid=%s)", result.get("pid"))
    else:
        logger.warning("embed-server: start failed: %s", result.get("error"))


def _on_session_end(**kwargs) -> None:
    """No-op: embedding server is persistent across sessions.

    Override with EMBED_SERVER_STOP_ON_END=1 if you want per-session lifecycle.
    """
    if os.environ.get("EMBED_SERVER_STOP_ON_END") == "1":
        pm = _pm()
        pm.stop(reason="session_end")
        logger.info("embed-server: stopped on session end")


# ---------------------------------------------------------------------------
# Slash command
# ---------------------------------------------------------------------------

_HELP_TEXT = """\
/embed — embedding server management

Subcommands:
  status              Show server status (running/stopped, port, uptime)
  restart             Restart the server
  stop                Stop the server
  logs [N]            Show last N log lines (default: 20)

  init                Initialize / verify the plugin setup
  doctor              Full diagnostics report
  update              Update plugin and model files

The server runs on port 11434 by default and provides:
  POST /v1/embeddings   OpenAI-compatible embeddings
  GET  /v1/models       List available models
  GET  /health          Health check
"""


def _handle_doctor() -> str:
    """Run full diagnostics and return a formatted report."""
    lines = ["[embed-server] 🏥 Doctor report"]
    lines.append("")

    # Python packages
    lines.append("📦 Python packages:")
    pkgs = _check_python_packages()
    for line in pkgs.get("packages", []):
        lines.append(line)
    if not pkgs["ok"]:
        lines.append("  ⚠ Run `/embed init` to fix missing packages")
    lines.append("")

    # Model files
    lines.append("📁 Model files:")
    model = _check_model_files()
    for line in model.get("files", []):
        lines.append(line)
    if not model["ok"]:
        lines.append("  ⚠ Model files missing; reinstall or run `/embed init`")
    lines.append("")

    # Port
    lines.append("🔌 Port 11434:")
    port = _check_port()
    if port["in_use"]:
        lines.append("  ⚠ Port 11434 is already in use (maybe the server is running?)")
    else:
        lines.append("  ✅ Available")
    lines.append("")

    # Hermes env
    lines.append("⚙ Hermes environment:")
    env = _check_hermes_env()
    lines.append(f"  Hermes home: {env['hermes_home']}")
    lines.append(f"  Python:      {env['python']}")
    lines.append("")

    # Server status if running
    pm = _pm()
    s = pm.status()
    if s.get("ok") and s.get("alive"):
        lines.append("🟢 Server is RUNNING")
        lines.append(f"  PID:     {s.get('pid')}")
        lines.append(f"  Port:    {s.get('port')}")
        lines.append(f"  Uptime:  {s.get('uptime')}s")
    else:
        lines.append("🔴 Server is STOPPED")

    return "\n".join(lines)


def _handle_init() -> str:
    """Initialize the plugin: check deps, model files, test embedding."""
    lines = ["[embed-server] 🔧 Initializing..."]

    # 1. Check packages
    lines.append("")
    lines.append("Step 1/4: Checking Python packages...")
    pkgs = _check_python_packages()
    for line in pkgs.get("packages", []):
        lines.append(line)
    if not pkgs["ok"]:
        lines.append("  ❌ Missing packages. Install them:")
        lines.append("     pip install fastembed fastapi uvicorn onnxruntime")
        return "\n".join(lines)

    # 2. Check model files
    lines.append("")
    lines.append("Step 2/4: Checking model files...")
    model = _check_model_files()
    for line in model.get("files", []):
        lines.append(line)
    if not model["ok"]:
        lines.append("  ❌ Model files incomplete. Reinstall plugin:")
        lines.append("     hermes plugins update embed-server")
        return "\n".join(lines)

    # 3. Try loading the model
    lines.append("")
    lines.append("Step 3/4: Testing model load...")
    try:
        from fastembed import TextEmbedding
        m = TextEmbedding(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            local_files_only=True,
        )
        dim = len(list(m.embed(["test"]))[0])
        lines.append(f"  ✅ Model loaded, dim={dim}")
    except Exception as e:
        lines.append(f"  ❌ Model load failed: {e}")
        return "\n".join(lines)

    # 4. Test API
    lines.append("")
    lines.append("Step 4/4: Starting server...")
    pm = _pm()
    result = pm.start(port=11434, model="nomic-ai/nomic-embed-text-v1.5")
    if result.get("ok"):
        lines.append(f"  ✅ Server started (pid={result.get('pid')}, port={result.get('port')})")
    else:
        lines.append(f"  ⚠ {result.get('status', result.get('error', 'unknown'))}")

    lines.append("")
    lines.append("✅ Initialization complete!")
    return "\n".join(lines)


def _handle_update() -> str:
    """Update the plugin and model."""
    lines = ["[embed-server] 🔄 Updating..."]

    # Pull latest plugin code
    lines.append("")
    lines.append("Step 1/2: Updating plugin...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "hermes_cli.plugins_cmd", "update", "embed-server"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            lines.append("  ✅ Plugin updated")
        else:
            # Fallback: git pull directly
            plugin_git = _plugin_dir() / ".git"
            if plugin_git.exists():
                r = subprocess.run(
                    ["git", "pull", "--ff-only"],
                    capture_output=True, text=True, timeout=30,
                    cwd=str(_plugin_dir()),
                )
                if r.returncode == 0:
                    lines.append(f"  ✅ {r.stdout.strip() or 'Up to date'}")
                else:
                    lines.append(f"  ⚠ {r.stderr.strip()}")
            else:
                lines.append("  ⚠ Not a git checkout; run:")
                lines.append("     hermes plugins update embed-server")
    except Exception as e:
        lines.append(f"  ⚠ Update skipped: {e}")
        lines.append("  Run manually: hermes plugins update embed-server")

    # Restart server with new code
    lines.append("")
    lines.append("Step 2/2: Restarting server...")
    pm = _pm()
    pm.stop(reason="update")
    result = pm.start(port=11434, model="nomic-ai/nomic-embed-text-v1.5")
    if result.get("ok"):
        lines.append(f"  ✅ Server restarted (pid={result.get('pid')})")
    else:
        status = pm.status()
        if status.get("ok") and status.get("alive"):
            lines.append("  ✅ Server already running (restart not needed)")
        else:
            lines.append(f"  ⚠ Server restart: {result.get('error', 'unknown')}")

    lines.append("")
    lines.append("✅ Update complete!")
    return "\n".join(lines)


def _handle_slash(raw_args: str) -> Optional[str]:
    argv = raw_args.strip().split()
    if not argv or argv[0] in {"help", "-h", "--help"}:
        return _HELP_TEXT

    pm = _pm()
    sub = argv[0]

    # ---- Lifecycle ----
    if sub == "status":
        s = pm.status()
        if not s.get("ok"):
            return f"[embed-server] ⛔ {s.get('reason', 'unknown')}"
        alive = s.get("alive", False)
        icon = "🟢" if alive else "🔴"
        uptime = s.get("uptime", 0)
        return (
            f"[embed-server] {icon} {'Running' if alive else 'Stopped'}\n"
            f"  PID:     {s.get('pid')}\n"
            f"  Port:    {s.get('port')}\n"
            f"  Host:    {s.get('host')}\n"
            f"  Model:   {s.get('model')}\n"
            f"  Uptime:  {uptime}s\n"
            f"  Log:     {s.get('logPath', 'N/A')}"
        )

    if sub == "restart":
        model = argv[1] if len(argv) > 1 else None
        result = pm.restart(model=model)
        if result.get("ok"):
            return f"[embed-server] 🔄 Restarted (pid={result.get('pid')}, port={result.get('port')})"
        return f"[embed-server] ❌ Restart failed: {result.get('error', 'unknown')}"

    if sub == "stop":
        result = pm.stop(reason="user_command")
        if result.get("ok"):
            return "[embed-server] ⏹ Stopped"
        return f"[embed-server] ❌ {result.get('reason', 'stop failed')}"

    if sub in ("logs", "log"):
        n = int(argv[1]) if len(argv) > 1 else 20
        result = pm.tail_log(n_lines=n)
        if not result.get("ok"):
            return f"[embed-server] ❌ {result.get('reason', 'log unavailable')}"
        lines = result.get("lines", [])
        if not lines:
            return "[embed-server] 📄 (empty log)"
        return f"[embed-server] 📄 Last {len(lines)}/{result['total']} lines:\n" + "\n".join(lines)

    # ---- Management ----
    if sub == "init":
        return _handle_init()

    if sub == "doctor":
        return _handle_doctor()

    if sub == "update":
        return _handle_update()

    return f"Unknown subcommand: {sub}\n\n{_HELP_TEXT}"


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_command(
        "embed",
        handler=_handle_slash,
        description="Manage the embedding server (init, doctor, update, status, restart, stop, logs).",
    )
