"""Downloads with retry and resume, manifest.json with hashes, atomic writes, git state."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from . import config

CODE_DIRS = ("src/", "notebooks/", "submission/", "tests/")


# ---------------------------------------------------------------- atomic writes
def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def read_json(path: Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------- hashes and manifest
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def manifest_record(path: Path, source_url: str, **extra) -> dict:
    """Register a raw file in manifest.json (download time, URL, hash)."""
    path = Path(path)
    man = read_json(config.MANIFEST, default={"files": {}, "events": []})
    rel = str(path.resolve().relative_to(config.ROOT)) if path.resolve().is_relative_to(config.ROOT) else str(path)
    rec = {"url": source_url, "sha256": sha256_file(path), "bytes": path.stat().st_size,
           "downloaded_at": now_utc(), **extra}
    man["files"][rel] = rec
    atomic_write_json(config.MANIFEST, man)
    return rec


def manifest_event(kind: str, **data) -> None:
    man = read_json(config.MANIFEST, default={"files": {}, "events": []})
    man["events"].append({"kind": kind, "at": now_utc(), **data})
    atomic_write_json(config.MANIFEST, man)


def verify_manifest() -> list[str]:
    """Raw files whose content changed after download (validate.assert_ready stops on these)."""
    man = read_json(config.MANIFEST, default={"files": {}})
    changed = []
    for rel, rec in man["files"].items():
        p = config.ROOT / rel
        if not p.exists() or sha256_file(p) != rec["sha256"]:
            changed.append(rel)
    return changed


# ---------------------------------------------------------------- downloads
def download(url: str, dest: Path, *, retries: int = 6, timeout: int = 60,
             headers: dict | None = None, session=None) -> Path:
    """Resumable download with exponential back-off. Registers the file in manifest.json."""
    import requests
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    sess = session or requests.Session()
    for attempt in range(retries):
        try:
            h = dict(headers or {})
            pos = part.stat().st_size if part.exists() else 0
            if pos:
                h["Range"] = f"bytes={pos}-"
            with sess.get(url, headers=h, stream=True, timeout=timeout) as r:
                if r.status_code == 404:
                    raise FileNotFoundError(url)
                if r.status_code == 416:            # already complete
                    break
                r.raise_for_status()
                mode = "ab" if pos and r.status_code == 206 else "wb"
                with open(part, mode) as f:
                    for block in r.iter_content(1 << 20):
                        f.write(block)
            break
        except FileNotFoundError:
            raise
        except Exception as exc:                    # network, 429, 5xx
            if attempt == retries - 1:
                raise RuntimeError(f"download failed after {retries} tries: {url}") from exc
            time.sleep(min(2 ** attempt * 5, 300))
    os.replace(part, dest)
    manifest_record(dest, url)
    return dest


def get_json(url: str, params: dict | None = None, *, retries: int = 6, timeout: int = 60,
             headers: dict | None = None, session=None, pause: float = 0.0) -> dict:
    import requests
    sess = session or requests.Session()
    for attempt in range(retries):
        try:
            r = sess.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code in (403, 429) or r.status_code >= 500:
                raise RuntimeError(f"HTTP {r.status_code}")
            r.raise_for_status()
            if pause:
                time.sleep(pause)
            return r.json()
        except Exception as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"GET failed after {retries} tries: {url} {params}") from exc
            time.sleep(min(2 ** attempt * 6, 300))
    raise AssertionError("unreachable")


# ---------------------------------------------------------------- git and environment
def _git(*args) -> str:
    r = subprocess.run(["git", "-C", str(config.ROOT), *args],
                       capture_output=True, text=True, check=True)
    return r.stdout.strip()


def git_state() -> dict:
    """dirty = uncommitted changes in code only; outputs/ and data/ are ignored by design."""
    try:
        return {"commit": _git("rev-parse", "HEAD"),
                "dirty": bool(_git("status", "--porcelain", "--", *CODE_DIRS))}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": True}       # no git = never "clean"


def env_lock(path: Path | None = None) -> dict:
    """pip freeze + GPU + CUDA, once per Colab session."""
    info = {"at": now_utc(), "python": sys.version, "platform": platform.platform()}
    try:
        info["pip_freeze"] = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                            capture_output=True, text=True).stdout.splitlines()
    except Exception:
        info["pip_freeze"] = []
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        info["torch"] = None
    path = path or (config.OUTPUTS / "env" / f"env_{info['at'][:19].replace(':', '')}.json")
    atomic_write_json(path, info)
    return info


def append_lock_history(record: dict) -> None:
    config.LOCK_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with config.LOCK_HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": now_utc(), **record}, ensure_ascii=False, default=str) + "\n")
