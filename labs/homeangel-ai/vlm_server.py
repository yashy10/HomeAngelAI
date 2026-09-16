#!/usr/bin/env python3
"""Start HomeAngel's local-only VLM server from config.yaml."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml


DEFAULT_CONFIG = Path("/workspace/labs/homeangel-ai/config.yaml")
DEFAULT_MODELS = "Gemma-4-E4B-it,Qwen3-VL-4B-Instruct-GPTQ-a16w4"


def split_csv(value: object) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_model_paths(value: object) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for item in split_csv(value):
        name, sep, path = item.partition("=")
        if not sep:
            print(f"ignoring malformed vlm.model_paths entry: {item}", file=sys.stderr)
            continue
        name = name.strip()
        path = path.strip()
        if name and path:
            mapping[name] = Path(path)
    return mapping


def is_loopback(host: str) -> bool:
    return host.strip().lower() in {"127.0.0.1", "localhost", "::1", "[::1]"}


def load_vlm_config(config_path: Path) -> dict[str, object]:
    if not config_path.is_file():
        raise FileNotFoundError(f"config does not exist: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    vlm = raw.get("vlm", {}) or {}
    host = str(vlm.get("host", "127.0.0.1") or "127.0.0.1")
    port = int(vlm.get("port", 9998))
    models = split_csv(vlm.get("models", DEFAULT_MODELS))
    model_paths = parse_model_paths(vlm.get("model_paths", ""))
    if not is_loopback(host):
        raise ValueError("vlm.host must be loopback because image crops are sent to this server")
    if not models:
        raise ValueError("vlm.models must list at least one model")
    return {
        "host": host,
        "port": port,
        "models": models,
        "model_paths": model_paths,
    }


def discover_served_models(cfg: dict[str, object]) -> list[tuple[str, Path]]:
    model_paths = cfg["model_paths"]
    assert isinstance(model_paths, dict)
    served: list[tuple[str, Path]] = []
    for model in cfg["models"]:
        path = model_paths.get(model)
        if path is None:
            print(f"missing vlm.model_paths entry for {model}", file=sys.stderr)
            continue
        if not path.is_dir():
            print(f"model directory does not exist for {model}: {path}", file=sys.stderr)
            continue
        served.append((str(model), path))
    return served


def start_server(host: str, port: int, served: list[tuple[str, Path]]):
    import pyneat

    server_options = pyneat.GenAIServerOptions()
    server_options.host = host
    server_options.port = port

    genai_server = pyneat.GenAIServer(server_options)
    for model_name, model_path in served:
        actual_name = genai_server.add_model(model_path, model_name)
        print(f"added model: {actual_name} -> {model_path}", flush=True)
    print(f"serving HomeAngel VLM API on http://{host}:{port}", flush=True)
    genai_server.start()
    return genai_server


def selected_models_for_memory(available: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    # Modalix memory is tight for VLMs; serve the first available route entry.
    return available[:1]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check-config", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    try:
        cfg = load_vlm_config(args.config)
    except Exception as exc:
        print(f"VLM config error: {exc}", file=sys.stderr)
        return 2

    available = discover_served_models(cfg)
    served = selected_models_for_memory(available)
    route = " -> ".join(str(model) for model in cfg["models"])
    print(f"VLM route: {route}", flush=True)
    if args.check_config:
        print(f"configured endpoint: http://{cfg['host']}:{cfg['port']}", flush=True)
        print(f"installed local model directories found: {len(available)}", flush=True)
        print(
            "selected model to serve: "
            + (served[0][0] if served else "none"),
            flush=True,
        )
        return 0

    if not served:
        print(
            "No configured VLM model directories exist. Deploy Gemma-4-E4B-it with LLiMa "
            "or update vlm.model_paths before starting the server.",
            file=sys.stderr,
        )
        return 2

    server = None
    try:
        server = start_server(str(cfg["host"]), int(cfg["port"]), served)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nstopping HomeAngel VLM server...", flush=True)
        return 0
    finally:
        if server is not None:
            server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
