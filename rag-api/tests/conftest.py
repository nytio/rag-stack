from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest

RAG_API_DIR = Path(__file__).resolve().parents[1]
if str(RAG_API_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_API_DIR))

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
TEST_ENV_PATH = Path(__file__).resolve().parents[2] / ".env.test"


def parse_env_file(path: Path = ENV_PATH) -> dict[str, str]:
    if not path.exists():
        return {}

    data: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        data[key] = value
    return data


def load_env_file(path: Path = ENV_PATH, *, override: bool = False) -> dict[str, str]:
    data = parse_env_file(path)
    for key, value in data.items():
        if override:
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)
    return data


_ENV_DATA: dict[str, str] = {}
_ENV_DATA.update(load_env_file(ENV_PATH))
_ENV_DATA.update(load_env_file(TEST_ENV_PATH, override=True))


def require_modules(*names: str) -> None:
    for name in names:
        pytest.importorskip(name)


def require_url_resolvable(url: str, *, var_name: str) -> None:
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not host:
        pytest.skip(f"{var_name} is not a valid URL: {url}")
    try:
        socket.getaddrinfo(host, port)
    except OSError:
        pytest.skip(f"{var_name} host not resolvable from test env: {host}")


def require_env(*names: str) -> None:
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        pytest.skip(f"Missing required env vars for integration test: {', '.join(missing)}")
