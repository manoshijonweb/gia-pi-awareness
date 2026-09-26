from __future__ import annotations
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import socket
import sys
import tempfile

class FeatureError(RuntimeError):
    """A feature is unavailable or has invalid input; never fabricate a result."""

class Cancelled(FeatureError):
    pass

def configure_logging(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if root.handlers:
        return
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    for handler in (logging.StreamHandler(), RotatingFileHandler(
        folder / "app.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8"
    )):
        handler.setFormatter(formatter)
        root.addHandler(handler)

def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

_guard_installed = False

def install_offline_guard() -> None:
    """Block Python Internet sockets, not local UNIX audio/camera IPC.
    Optional systemd service also uses a private network namespace, which covers
    native libraries. This hook alone is not a native-code security sandbox.
    """
    global _guard_installed
    if _guard_installed:
        return
    def audit(event, args):
        if event in ("socket.connect", "socket.sendto", "socket.bind"):
            sock = args[0]
            if sock.family in (socket.AF_INET, socket.AF_INET6):
                raise PermissionError("Internet sockets disabled in offline runtime")
        if event == "socket.getaddrinfo":
            raise PermissionError("DNS disabled in offline runtime")
    sys.addaudithook(audit)
    _guard_installed = True
