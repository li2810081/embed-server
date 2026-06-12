"""Subprocess lifecycle manager for the fastembed server.

Starts/stops/monitors a uvicorn server process that provides an OpenAI-compatible
embedding API on port 11434. Persistent across sessions — once started it keeps
running until explicitly stopped or Hermes shuts down.

State is tracked in a JSON file under ``$HERMES_HOME/workspace/embed-server/``.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home


def _root() -> Path:
    return Path(get_hermes_home()) / "workspace" / "embed-server"


def _active_file() -> Path:
    return _root() / ".active.json"


def _read_active() -> Optional[Dict[str, Any]]:
    p = _active_file()
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_active(data: Dict[str, Any]) -> None:
    p = _active_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(p)


def _clear_active() -> None:
    try:
        _active_file().unlink()
    except FileNotFoundError:
        pass


def _pid_alive(pid: int) -> bool:
    """Check if a PID exists on the system (cross-platform)."""
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x400000, False, pid)  # PROCESS_QUERY_INFORMATION
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            pass
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _plugin_dir() -> Path:
    """Return the directory where this plugin's source files live."""
    return Path(__file__).parent.resolve()


def start(
    *,
    host: str = "0.0.0.0",
    port: int = 11434,
    model: str = "nomic-ai/nomic-embed-text-v1.5",
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Spawn the embedding server subprocess.

    Returns a dict with status info, or an error if already running.
    """
    existing = _read_active()
    if existing:
        pid = int(existing.get("pid", 0))
        if pid and _pid_alive(pid):
            return {
                "ok": True,
                "status": "already running",
                "pid": pid,
                "port": existing.get("port", port),
            }
        # Stale entry — clean it up
        _clear_active()

    out = _root() / "logs"
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "server.log"

    server_script = _plugin_dir() / "server.py"
    if not server_script.exists():
        return {"ok": False, "error": f"server.py not found at {server_script}"}

    # Point fastembed cache to bundled model directory
    model_dir = str(_plugin_dir() / "model")

    env = os.environ.copy()
    env["EMBED_SERVER_HOST"] = host
    env["EMBED_SERVER_PORT"] = str(port)
    env["EMBED_SERVER_MODEL"] = model
    env["FASTEMBED_CACHE_PATH"] = model_dir
    env["HF_ENDPOINT"] = "https://hf-mirror.com"

    log_fh = open(log_path, "ab", buffering=0)
    try:
        proc = subprocess.Popen(
            [sys.executable, str(server_script)],
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log_fh.close()

    # Wait briefly, then check if still alive
    time.sleep(2)
    if not _pid_alive(proc.pid):
        error_log = ""
        try:
            error_log = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        except Exception:
            pass
        return {
            "ok": False,
            "error": "Server exited immediately after start",
            "log_tail": error_log,
        }

    record = {
        "pid": proc.pid,
        "host": host,
        "port": port,
        "model": model,
        "started_at": time.time(),
        "session_id": session_id,
        "log_path": str(log_path),
    }
    _write_active(record)

    return {
        "ok": True,
        "status": "started",
        "pid": proc.pid,
        "port": port,
        "host": host,
        "model": model,
    }


def status() -> Dict[str, Any]:
    """Return current server state."""
    active = _read_active()
    if not active:
        return {"ok": False, "reason": "server not started"}

    pid = int(active.get("pid", 0))
    alive = _pid_alive(pid) if pid else False

    return {
        "ok": True,
        "alive": alive,
        "pid": pid,
        "host": active.get("host", "0.0.0.0"),
        "port": active.get("port", 11434),
        "model": active.get("model", "nomic-embed-text-v1.5"),
        "startedAt": active.get("started_at"),
        "logPath": active.get("log_path"),
        "uptime": round(time.time() - active.get("started_at", time.time()), 1) if alive else 0,
    }


def stop(*, reason: str = "shutdown") -> Dict[str, Any]:
    """Stop the embedding server process."""
    active = _read_active()
    if not active:
        return {"ok": False, "reason": "server not running"}

    pid = int(active.get("pid", 0))
    if pid and _pid_alive(pid):
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                               capture_output=True, timeout=5)
            else:
                os.kill(pid, signal.SIGTERM)
                for _ in range(10):
                    if not _pid_alive(pid):
                        break
                    time.sleep(0.5)
                if _pid_alive(pid):
                    os.kill(pid, signal.SIGKILL)
        except Exception:
            pass

    _clear_active()
    return {"ok": True, "reason": reason}


def restart(*, model: Optional[str] = None) -> Dict[str, Any]:
    """Restart the server with optional model change."""
    active = _read_active()
    port = int(active.get("port", 11434)) if active else 11434
    host = active.get("host", "0.0.0.0") if active else "0.0.0.0"
    current_model = active.get("model", "nomic-embed-text-v1.5") if active else "nomic-embed-text-v1.5"

    stop(reason="restart")
    time.sleep(1)
    return start(
        host=host,
        port=port,
        model=model or current_model,
    )


def tail_log(n_lines: int = 20) -> Dict[str, Any]:
    """Return the last N lines of the server log."""
    active = _read_active()
    if not active or not active.get("log_path"):
        return {"ok": False, "reason": "no log file found"}

    log_path = Path(active["log_path"])
    if not log_path.exists():
        return {"ok": False, "reason": "log file does not exist"}

    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = lines[-n_lines:]
        return {"ok": True, "lines": tail, "total": len(lines)}
    except Exception as e:
        return {"ok": False, "error": str(e)}
