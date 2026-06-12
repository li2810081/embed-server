"""embed-server CLI — install, doctor, init, start, status."""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Ensure UTF-8 output on Windows (emoji-friendly)
if sys.stdout.encoding and sys.stdout.encoding.upper() != "UTF-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def _package_root() -> Path:
    return Path(__file__).parent.resolve()

def _hermes_plugins_dir() -> Path:
    """~/.hermes/plugins/"""
    return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")) / "plugins"

def _hermes_venv_python() -> Optional[str]:
    """Find Hermes' venv Python."""
    candidates = [
        # Standard install locations
        Path.home() / "AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe",
        Path.home() / ".local/share/hermes/hermes-agent/venv/bin/python",
        Path("/opt/hermes-agent/venv/bin/python"),
    ]
    for p in candidates:
        if p.exists():
            return str(p.resolve())
    # Try PATH
    for name in ["hermes", "hermes-agent"]:
        which = shutil.which(name)
        if which:
            # Resolve symlink to find venv
            p = Path(which).resolve()
            # Walk up to find venv
            for parent in [p.parent, p.parent.parent, p.parent.parent.parent]:
                venv_python = parent / "python"
                if not venv_python.exists():
                    venv_python = parent / "python.exe"
                if venv_python.exists():
                    return str(venv_python)
    return None


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------

def cmd_install(args):
    """Symlink plugin into ~/.hermes/plugins/ + ensure deps in Hermes venv."""
    plugins_dir = _hermes_plugins_dir()
    plugins_dir.mkdir(parents=True, exist_ok=True)

    target = plugins_dir / "embed-server"
    source = _package_root() / "plugin"

    if not source.exists():
        print(f"❌ Plugin source not found at {source}")
        return 1

    # Remove existing
    if target.exists():
        if target.is_symlink():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()

    # Symlink (copy fallback on Windows without Admin)
    try:
        target.symlink_to(source, target_is_directory=True)
        print(f"✅ Symlinked: {target} → {source}")
    except (OSError, NotImplementedError):
        # Fallback: copy
        shutil.copytree(source, target, dirs_exist_ok=True)
        print(f"✅ Copied:  {source} → {target}")

    # Install deps into Hermes venv
    hermes_python = args.hermes_python or _hermes_venv_python()
    if hermes_python:
        print(f"\n📦 Installing deps into Hermes venv ({hermes_python})...")
        deps = ["fastembed", "fastapi", "uvicorn", "onnxruntime"]
        for dep in deps:
            r = subprocess.run(
                [hermes_python, "-m", "pip", "install", dep, "-q"],
                capture_output=True, text=True,
            )
            if r.returncode == 0:
                print(f"  ✅ {dep}")
            else:
                print(f"  ⚠ {dep}: {r.stderr.strip() or 'already installed'}")
    else:
        print("  ⚠ Hermes venv not found, skipping dependency install")

    # Enable the plugin
    print("\n🔌 Enabling plugin...")
    r = subprocess.run(
        [sys.executable, "-m", "hermes_cli.plugins_cmd", "enable", "embed-server"],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        print("  ✅ Plugin enabled")
    else:
        print("  ⚠ Run manually: hermes plugins enable embed-server")

    print("\n✅ Install complete!")
    print("   Next steps:")
    print("     1. Edit ~/.hermes/.env  (set EMBED_SERVER_PORT, EMBED_SERVER_MODEL)")
    print("     2. hermes gateway restart")
    return 0


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

def cmd_doctor(args):
    """Health check — read-only diagnostic."""
    errors = 0

    print("🏥 embed-server doctor\n")

    # 1. Python deps
    print("📦 Python packages:")
    for pkg in ["fastembed", "fastapi", "uvicorn", "onnxruntime"]:
        try:
            mod = __import__(pkg.replace("-", "_"))
            ver = getattr(mod, "__version__", "unknown")
            print(f"  ✅ {pkg} == {ver}")
        except ImportError:
            print(f"  ❌ {pkg} — not installed")
            errors += 1
    print()

    # 2. Hermes plugin
    print("🔌 Hermes plugin:")
    plugin_target = _hermes_plugins_dir() / "embed-server"
    if plugin_target.exists():
        print(f"  ✅ Installed at {plugin_target}")
    else:
        print(f"  ❌ Not installed — run: embed-server install")
        errors += 1

    # 3. Model files
    print("\n📁 Model files:")
    snapshot = (
        _package_root() / "plugin" / "model"
        / "models--nomic-ai--nomic-embed-text-v1.5"
        / "snapshots" / "e9b6763023c676ca8431644204f50c2b100d9aab"
    )
    expected = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "onnx/model.onnx",
    ]
    for rel in expected:
        p = snapshot / rel
        if p.exists():
            print(f"  ✅ {rel}  ({p.stat().st_size / 1024 / 1024:.1f}MB)")
        else:
            print(f"  ❌ {rel} — missing")
            errors += 1

    # 4. Port
    print("\n🔌 Port 11434:")
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("0.0.0.0", 11434))
        s.close()
        print("  ✅ Available")
    except OSError:
        print("  ⚠ In use (server may be running)")
    print()

    # 5. Hermes venv
    hermes_py = _hermes_venv_python()
    print("⚙ Hermes:")
    if hermes_py:
        print(f"  ✅ Venv: {hermes_py}")
    else:
        print("  ⚠ Not found (plugin may still work standalone)")

    # Summary
    print()
    if errors == 0:
        print("✅ All checks passed!")
    else:
        print(f"⚠ {errors} issue(s) found")
    return errors


# ---------------------------------------------------------------------------
# init (interactive setup)
# ---------------------------------------------------------------------------

def cmd_init(args):
    """Interactive setup wizard."""
    print("🔧 embed-server init\n")

    # Step 1: Install deps
    print("Step 1/3: Installing dependencies...")
    for dep in ["fastembed", "fastapi", "uvicorn"]:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", dep, "-q"],
            capture_output=True, text=True,
        )
        print(f"  {'✅' if r.returncode == 0 else '⚠'} {dep}")

    # Step 2: Activate plugin
    print("\nStep 2/3: Activating Hermes plugin...")
    cmd_install(args)

    # Step 3: Write .env template
    print("\nStep 3/3: .env template")
    env_path = Path.home() / ".hermes" / ".env"
    if env_path.exists():
        existing = env_path.read_text()
        if "EMBED_SERVER_PORT" in existing:
            print(f"  ✅ EMBED_SERVER_* already in {env_path}")
        else:
            with open(env_path, "a") as f:
                f.write("\n# embed-server\n")
                f.write("EMBED_SERVER_PORT=11434\n")
                f.write("EMBED_SERVER_MODEL=nomic-ai/nomic-embed-text-v1.5\n")
            print(f"  ✅ Appended to {env_path}")
    else:
        env_path.parent.mkdir(parents=True, exist_ok=True)
        with open(env_path, "w") as f:
            f.write("# embed-server\n")
            f.write("EMBED_SERVER_PORT=11434\n")
            f.write("EMBED_SERVER_MODEL=nomic-ai/nomic-embed-text-v1.5\n")
        print(f"  ✅ Created {env_path}")

    print("\n✅ Init complete! Run: hermes gateway restart")
    return 0


# ---------------------------------------------------------------------------
# start (standalone)
# ---------------------------------------------------------------------------

def cmd_start(args):
    """Start the server standalone (no Hermes)."""
    from embed_server.server import main
    sys.argv = ["embed-server", "--port", str(args.port), "--model", args.model]
    main()


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="embed-server",
        description="OpenAI-compatible embedding server — CPU, lightweight, Hermes plugin",
    )
    sub = parser.add_subparsers(dest="command")

    # install
    p_install = sub.add_parser("install", help="Activate plugin in Hermes")
    p_install.add_argument("--hermes-python", help="Path to Hermes venv Python")

    # doctor
    sub.add_parser("doctor", help="Health check (read-only diagnostic)")

    # init
    p_init = sub.add_parser("init", help="Interactive setup wizard")
    p_init.add_argument("--hermes-python", help="Path to Hermes venv Python")

    # start (standalone)
    p_start = sub.add_parser("start", help="Run server standalone")
    p_start.add_argument("--port", type=int, default=11434)
    p_start.add_argument("--model", default="nomic-ai/nomic-embed-text-v1.5")

    args = parser.parse_args()

    if args.command == "install":
        return cmd_install(args)
    elif args.command == "doctor":
        return cmd_doctor(args)
    elif args.command == "init":
        return cmd_init(args)
    elif args.command == "start":
        return cmd_start(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
