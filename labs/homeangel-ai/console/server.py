#!/usr/bin/env python3
import argparse
import cgi
import json
import os
import re
import shlex
import signal
import socket
import ssl
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    import yaml
except Exception:
    yaml = None


APP_ROOT = Path("/workspace/labs/homeangel-ai")
INSIGHT_BASE = os.environ.get("HOMEANGEL_INSIGHT_URL", "https://127.0.0.1:9900").rstrip("/")
INSIGHT_SOURCE_INDEX = int(os.environ.get("HOMEANGEL_INSIGHT_SOURCE", "1"))
DEVKIT_HOST = os.environ.get("HOMEANGEL_DEVKIT_HOST", "192.168.1.20")
APP_CONFIG = Path(os.environ.get("HOMEANGEL_APP_CONFIG", str(APP_ROOT / "config.devkit.yaml")))
APP_BINARY = Path(os.environ.get("HOMEANGEL_APP_BINARY", str(APP_ROOT / "build/homeangel-ai")))
ENV_FILE = Path(os.environ.get("HOMEANGEL_ENV_FILE", str(APP_ROOT / ".env.local")))
APP_LOG = APP_ROOT / "console_app.log"
OUTPUT_FILES = [
    APP_ROOT / "events.log",
    APP_ROOT / "telemetry.json",
    APP_ROOT / "telemetry.json.tmp",
    APP_ROOT / "console_app.log",
]
PRESETS = [
    {
        "id": "fall-alert",
        "title": "Fall Alert",
        "file": "demo.mp4",
        "path": Path("/workspace/assets/videos/fall/demo.mp4"),
        "function": "fall_detection",
        "expected": "fall_detected",
        "summary": "Triggers the emergency alert path.",
    },
    {
        "id": "sit-down",
        "title": "Sit-Down Check",
        "file": "adl_sitdown.mp4",
        "path": Path("/workspace/assets/videos/fall/adl_sitdown.mp4"),
        "function": "sit_no_fall",
        "expected": "safe_sit",
        "summary": "Shows routine sitting without a fall alert.",
    },
    {
        "id": "routine-adl",
        "title": "Routine Activity",
        "file": "routine_adl.mp4",
        "path": Path("/workspace/gmdcsa24/Subject 1/ADL/01.mp4"),
        "function": "adl_no_fall",
        "expected": "safe_activity",
        "summary": "Shows daily movement staying safe.",
    },
]
PRESETS_BY_ID = {preset["id"]: preset for preset in PRESETS}
LOCAL_PRESETS = {preset["file"]: preset["path"] for preset in PRESETS}
TEMP_UPLOAD_PREFIX = "homeangel_upload_"

SSL_CONTEXT = ssl._create_unverified_context()
PROCESS_LOCK = threading.Lock()
APP_PROCESS = None
TELEMETRY_LOCK = threading.Lock()
LAST_TELEMETRY = None
SCENARIO_LOCK = threading.Lock()
CURRENT_SCENARIO = None


def read_text_tail(path, limit=1600):
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return ""
    return data[-limit:].decode("utf-8", errors="replace")


def split_csv(value):
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def env_file_values():
    values = {}
    try:
        lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return values
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            parts = shlex.split(line, comments=True, posix=True)
        except ValueError:
            continue
        if parts and parts[0] == "export":
            parts = parts[1:]
        for part in parts:
            key, sep, value = part.partition("=")
            if sep and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                values[key] = value
    return values


def env_value(name):
    value = os.environ.get(str(name or ""))
    if value:
        return value
    return env_file_values().get(str(name or ""), "")


def load_config_doc():
    if yaml is None:
        raise RuntimeError("PyYAML is required to read Telegram settings")
    if not APP_CONFIG.exists():
        raise RuntimeError(f"app config not found: {APP_CONFIG}")
    return yaml.safe_load(APP_CONFIG.read_text(encoding="utf-8")) or {}


def route_items(routes):
    if not routes:
        return []
    if isinstance(routes, dict):
        return [(str(key), str(value)) for key, value in routes.items()]
    if isinstance(routes, list):
        items = []
        for route in routes:
            if isinstance(route, dict):
                selector = str(route.get("selector", "") or route.get("zone", "") or route.get("device", ""))
                chat_id = str(route.get("chat_id", "") or route.get("chat", ""))
                items.append((selector, chat_id))
            else:
                name, sep, chat_id = str(route).partition("=")
                if sep:
                    items.append((name, chat_id))
        return items
    items = []
    for item in split_csv(routes):
        selector, sep, chat_id = item.partition("=")
        if sep:
            items.append((selector, chat_id))
    return items


def env_reference_name(value):
    text = str(value or "").strip()
    match = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", text)
    if match:
        return match.group(1)
    match = re.fullmatch(r"\$([A-Za-z_][A-Za-z0-9_]*)", text)
    if match:
        return match.group(1)
    return ""


def resolve_env_reference(value):
    env_name = env_reference_name(value)
    if env_name:
        return env_value(env_name).strip()
    return str(value or "").strip()


def telegram_settings():
    try:
        raw = load_config_doc()
    except Exception as exc:
        return {
            "enabled": False,
            "ready": False,
            "token_env": "HOMEANGEL_TELEGRAM_BOT_TOKEN",
            "token_present": False,
            "chat_count": 0,
            "route_count": 0,
            "error": str(exc),
        }

    device_id = str(raw.get("device_id", "modalix-hallway-01") or "modalix-hallway-01")
    zone_label = str(raw.get("zone_label", "bedroom") or "bedroom")
    telegram = ((raw.get("alerts") or {}).get("telegram") or {})
    enabled = bool(telegram.get("enabled", False))
    token_env = str(telegram.get("token_env", "HOMEANGEL_TELEGRAM_BOT_TOKEN") or "HOMEANGEL_TELEGRAM_BOT_TOKEN")
    routes = route_items(telegram.get("routes"))
    chat_ids = []
    chat_envs = []
    missing_chat_envs = []
    for selector, chat in routes:
        if selector.strip() not in {"*", zone_label, device_id}:
            continue
        env_name = env_reference_name(chat)
        if env_name:
            chat_envs.append(env_name)
        resolved_chat = resolve_env_reference(chat)
        if resolved_chat:
            chat_ids.append(resolved_chat)
        elif env_name:
            missing_chat_envs.append(env_name)
    token_present = bool(env_value(token_env))
    return {
        "enabled": enabled,
        "ready": enabled and token_present and bool(chat_ids),
        "token_env": token_env,
        "token_present": token_present,
        "chat_count": len(chat_ids),
        "route_count": len(routes),
        "chat_envs": sorted(set(chat_envs)),
        "missing_chat_envs": sorted(set(missing_chat_envs)),
        "zone_label": zone_label,
        "device_id": device_id,
        "_chat_ids": chat_ids,
    }


def public_telegram_status(settings=None):
    settings = dict(settings or telegram_settings())
    settings.pop("_chat_ids", None)
    return settings


def vlm_settings():
    try:
        raw = load_config_doc()
    except Exception as exc:
        return {
            "enabled": False,
            "active": False,
            "server_ok": False,
            "host": "127.0.0.1",
            "port": 9998,
            "models": [],
            "error": str(exc),
        }

    vlm = raw.get("vlm", {}) or {}
    enabled = bool(vlm.get("enabled", False))
    host = str(vlm.get("host", "127.0.0.1") or "127.0.0.1")
    port = int(vlm.get("port", 9998))
    models = split_csv(vlm.get("models", "Gemma-4-E4B-it,Qwen3-VL-4B-Instruct-GPTQ-a16w4"))
    loopback = host.strip().lower() in {"127.0.0.1", "localhost", "::1", "[::1]"}
    server_ok = False
    served_models = []
    error = ""
    if enabled:
        if not loopback:
            error = "vlm.host must be loopback"
        else:
            try:
                with urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=0.8) as response:
                    data = response.read().decode("utf-8", errors="replace")
                    server_ok = 200 <= response.status < 300
                    try:
                        payload = json.loads(data or "{}")
                        served_models = [
                            str(item.get("id") or item.get("name"))
                            for item in payload.get("data", [])
                            if isinstance(item, dict) and (item.get("id") or item.get("name"))
                        ]
                    except Exception:
                        served_models = []
            except Exception as exc:
                error = str(exc)
    return {
        "enabled": enabled,
        "active": enabled and server_ok,
        "server_ok": server_ok,
        "host": host,
        "port": port,
        "models": models,
        "served_models": served_models,
        "error": error,
    }


def send_telegram_text(message):
    settings = telegram_settings()
    if not settings.get("enabled"):
        raise RuntimeError("Telegram is disabled in config.devkit.yaml")
    token_env = settings.get("token_env") or "HOMEANGEL_TELEGRAM_BOT_TOKEN"
    token = env_value(token_env)
    if not token:
        raise RuntimeError(f"Telegram token missing: export {token_env} before starting the console")
    chat_ids = settings.get("_chat_ids") or []
    if not chat_ids:
        raise RuntimeError("No Telegram chat_id route matches this zone_label/device_id")

    text = str(message or "").strip()
    if not text:
        raise RuntimeError("message is empty")
    if len(text) > 900:
        text = text[:897].rstrip() + "..."

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    sent = 0
    for chat_id in chat_ids:
        body = json.dumps({"chat_id": chat_id, "text": text}, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=12) as response:
                response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Telegram sendMessage failed with HTTP {exc.code}: {body[:300]}") from exc
        sent += 1
    return {"sent": sent, "telegram": public_telegram_status(settings)}


def clear_outputs():
    for path in OUTPUT_FILES:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def public_preset(preset):
    return {
        "id": preset["id"],
        "title": preset["title"],
        "file": preset["file"],
        "function": preset["function"],
        "expected": preset["expected"],
        "summary": preset["summary"],
        "available": preset["path"].exists(),
    }


def set_current_scenario(scenario):
    global CURRENT_SCENARIO
    with SCENARIO_LOCK:
        CURRENT_SCENARIO = dict(scenario)


def current_scenario():
    with SCENARIO_LOCK:
        return dict(CURRENT_SCENARIO) if CURRENT_SCENARIO else None


def devkit_ssh_open(timeout=1.0):
    try:
        with socket.create_connection((DEVKIT_HOST, 22), timeout=timeout):
            return True
    except OSError:
        return False


def app_state():
    global APP_PROCESS
    with PROCESS_LOCK:
        proc = APP_PROCESS
        if proc is None:
            return {"running": False, "returncode": None, "log_tail": read_text_tail(APP_LOG)}
        rc = proc.poll()
        if rc is None:
            return {"running": True, "pid": proc.pid, "returncode": None, "log_tail": read_text_tail(APP_LOG)}
        APP_PROCESS = None
        return {"running": False, "returncode": rc, "log_tail": read_text_tail(APP_LOG)}


def telemetry_state():
    global LAST_TELEMETRY
    path = APP_ROOT / "telemetry.json"
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        with TELEMETRY_LOCK:
            LAST_TELEMETRY = data
        return data, 200
    except FileNotFoundError:
        return {"error": "telemetry not available"}, 404
    except Exception:
        with TELEMETRY_LOCK:
            cached = LAST_TELEMETRY
        if cached is not None:
            stale = dict(cached)
            stale["stale_read"] = True
            return stale, 200
        return {"error": "telemetry not readable yet"}, 503


def insight_request(method, path, payload=None, raw_data=None, headers=None, timeout=20):
    request_headers = dict(headers or {})
    data = raw_data
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    url = f"{INSIGHT_BASE}{path}"
    request = urllib.request.Request(url, data=data, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(request, context=SSL_CONTEXT, timeout=timeout) as response:
            body = response.read()
            content_type = response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Insight {path} failed with HTTP {exc.code}: {body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Insight {path} unavailable: {exc}") from exc

    text = body.decode("utf-8", errors="replace")
    if "application/json" in content_type:
        return json.loads(text or "{}")
    return text


def multipart_body(field_name, filename, data):
    boundary = f"----homeangel-{uuid.uuid4().hex}"
    parts = [
        f"--{boundary}\r\n".encode("utf-8"),
        (
            f'Content-Disposition: form-data; name="{field_name}"; '
            f'filename="{filename}"\r\n'
        ).encode("utf-8"),
        b"Content-Type: application/octet-stream\r\n\r\n",
        data,
        b"\r\n",
        f"--{boundary}--\r\n".encode("utf-8"),
    ]
    return boundary, b"".join(parts)


def upload_media_bytes(filename, data):
    safe_name = Path(filename).name
    if not safe_name:
        raise RuntimeError("uploaded file has no filename")
    boundary, body = multipart_body("file", safe_name, data)
    return insight_request(
        "POST",
        "/api/upload/media",
        raw_data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
        timeout=120,
    )


def upload_media_path(path, upload_name=None):
    safe_name = Path(upload_name or path.name).name
    upload_media_bytes(safe_name, path.read_bytes())
    return safe_name


def media_url(path):
    safe_path = str(path or "").strip()
    if not safe_path:
        return ""
    return f"/api/media/{urllib.parse.quote(safe_path, safe='/')}"


def validate_media_path(path):
    media_path = urllib.parse.unquote(str(path or "")).strip()
    if not media_path or media_path.startswith("/") or ".." in Path(media_path).parts:
        raise RuntimeError("unsafe media path")
    return media_path


def insight_videos():
    videos = insight_request("GET", "/api/mediasrc/videos", timeout=10)
    return videos if isinstance(videos, list) else []


def media_file_paths(items):
    paths = []
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                if item.get("type") == "file" and item.get("path"):
                    paths.append(str(item["path"]))
                if item.get("children"):
                    paths.extend(media_file_paths(item["children"]))
            elif isinstance(item, str):
                paths.append(item)
    return paths


def delete_media(path):
    try:
        insight_request("POST", "/api/delete-media", {"path": path}, timeout=15)
    except RuntimeError:
        pass


def cleanup_temp_uploads():
    try:
        files = insight_request("GET", "/api/media-files", timeout=10)
    except RuntimeError:
        return
    for path in media_file_paths(files):
        if Path(path).name.startswith(TEMP_UPLOAD_PREFIX):
            delete_media(path)


def stop_insight_source():
    try:
        insight_request("POST", "/api/mediasrc/stop", {"index": INSIGHT_SOURCE_INDEX}, timeout=10)
    except RuntimeError:
        pass


def ensure_media(filename, local_path=None):
    safe_name = Path(filename).name
    if not safe_name:
        raise RuntimeError("missing video filename")
    if safe_name in insight_videos():
        return safe_name
    source_path = local_path or LOCAL_PRESETS.get(safe_name)
    if source_path and source_path.exists():
        upload_media_path(source_path, safe_name)
        return safe_name
    raise RuntimeError(f"{safe_name} is not loaded in Insight")


def assign_insight_source(filename, local_path=None):
    safe_name = ensure_media(filename, local_path)
    stop_insight_source()
    insight_request(
        "POST",
        "/api/mediasrc/assign",
        {"index": INSIGHT_SOURCE_INDEX, "file": safe_name, "transport": "rtsp"},
        timeout=15,
    )
    return safe_name


def start_insight_stream(filename, local_path=None):
    safe_name = assign_insight_source(filename, local_path)
    try:
        insight_request("POST", "/api/mediasrc/start", {"index": INSIGHT_SOURCE_INDEX}, timeout=15)
    except RuntimeError as exc:
        if "Already running" not in str(exc):
            raise
    return safe_name


def source_status():
    sources = insight_request("GET", "/api/mediasrc", timeout=10)
    if isinstance(sources, list):
        for source in sources:
            if source.get("index") == INSIGHT_SOURCE_INDEX:
                return source
    return {"index": INSIGHT_SOURCE_INDEX, "file": "", "state": "unknown"}


def viewer_url():
    result = insight_request("GET", "/api/viewer-url?mode=light&src=0", timeout=10)
    if isinstance(result, dict):
        return result.get("url", "")
    return ""


def start_app():
    global APP_PROCESS
    if not APP_BINARY.exists():
        raise RuntimeError(f"app binary not found: {APP_BINARY}")
    if not APP_CONFIG.exists():
        raise RuntimeError(f"app config not found: {APP_CONFIG}")
    if not devkit_ssh_open():
        raise RuntimeError(f"DevKit SSH is not reachable at {DEVKIT_HOST}:22")

    with PROCESS_LOCK:
        if APP_PROCESS is not None and APP_PROCESS.poll() is None:
            return {"running": True, "pid": APP_PROCESS.pid, "message": "analysis already running"}

        clear_outputs()
        APP_LOG.parent.mkdir(parents=True, exist_ok=True)
        log_file = APP_LOG.open("ab", buffering=0)
        command_parts = ["dk", str(APP_BINARY), "--config", str(APP_CONFIG)]
        command = " ".join(shlex.quote(part) for part in command_parts)
        APP_PROCESS = subprocess.Popen(
            ["bash", "-lic", command],
            cwd=str(APP_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    time.sleep(1.0)
    state = app_state()
    if not state.get("running"):
        raise RuntimeError(f"analysis exited early: {state.get('log_tail', '')[-600:]}")
    return {"running": True, "pid": state.get("pid"), "message": "analysis started"}


def stop_app():
    global APP_PROCESS
    with PROCESS_LOCK:
        proc = APP_PROCESS
        APP_PROCESS = None
    if proc is None or proc.poll() is not None:
        return {"running": False, "message": "analysis was not running"}
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=4)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass
    return {"running": False, "message": "analysis stopped"}


def start_analysis_session(filename, scenario, local_path=None):
    stop_app()
    clear_outputs()
    safe_name = start_insight_stream(filename, local_path)
    scenario_payload = dict(scenario)
    scenario_payload["file"] = safe_name
    set_current_scenario(scenario_payload)
    result = {
        "stream_started": True,
        "analysis_started": False,
        "file": safe_name,
        "media_url": media_url(safe_name),
        "scenario": current_scenario(),
        "source": source_status(),
        "viewer_url": viewer_url(),
    }
    try:
        app = start_app()
        result["analysis_started"] = True
        result["app"] = app
        result["message"] = f"{scenario_payload.get('title', safe_name)} is running"
    except Exception as exc:
        result["analysis_error"] = str(exc)
        result["app"] = app_state()
        result["message"] = f"{safe_name} is streaming; analysis did not start"
    return result


def prepare_analysis_session(filename, scenario, local_path=None):
    stop_app()
    clear_outputs()
    safe_name = assign_insight_source(filename, local_path)
    scenario_payload = dict(scenario)
    scenario_payload["file"] = safe_name
    set_current_scenario(scenario_payload)
    return {
        "prepared": True,
        "stream_started": False,
        "analysis_started": False,
        "file": safe_name,
        "media_url": media_url(safe_name),
        "scenario": current_scenario(),
        "source": source_status(),
        "viewer_url": viewer_url(),
        "message": f"{scenario_payload.get('title', safe_name)} is ready",
    }


def run_prepared_session():
    scenario = current_scenario()
    filename = scenario.get("file") if scenario else ""
    if not filename:
        raise RuntimeError("no prepared video session")
    clear_outputs()
    stop_insight_source()
    time.sleep(0.2)
    try:
        insight_request("POST", "/api/mediasrc/start", {"index": INSIGHT_SOURCE_INDEX}, timeout=15)
    except RuntimeError as exc:
        if "Already running" not in str(exc):
            raise
    result = {
        "prepared": True,
        "stream_started": True,
        "analysis_started": False,
        "file": filename,
        "media_url": media_url(filename),
        "scenario": scenario,
        "source": source_status(),
        "viewer_url": viewer_url(),
    }
    try:
        app = start_app()
        result["analysis_started"] = True
        result["app"] = app
        result["message"] = f"{scenario.get('title', filename)} is running"
    except Exception as exc:
        result["analysis_error"] = str(exc)
        result["app"] = app_state()
        result["message"] = f"{filename} is streaming; analysis did not start"
    return result


def reset_session():
    stop_app()
    stop_insight_source()
    cleanup_temp_uploads()
    clear_outputs()
    set_current_scenario({})
    return {"message": "session reset", "app": app_state(), "source": source_status()}


class HomeAngelHandler(SimpleHTTPRequestHandler):
    server_version = "HomeAngelConsole/1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(APP_ROOT), **kwargs)

    def end_headers(self):
        path = urllib.parse.urlparse(self.path).path
        if (
            path.endswith("telemetry.json")
            or path.endswith("events.log")
            or path.startswith("/api/")
            or path == "/console/"
            or path.startswith("/console/")
        ):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/status":
            self.handle_status()
            return
        if path.startswith("/api/media/"):
            self.handle_media_proxy(path.removeprefix("/api/media/"))
            return
        if path == "/api/presets":
            self.send_json({"presets": [public_preset(preset) for preset in PRESETS]})
            return
        if path == "/api/telemetry":
            payload, status = telemetry_state()
            self.send_json(payload, status=status)
            return
        super().do_GET()

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        try:
            if path == "/api/insight/start":
                self.handle_insight_start()
            elif path == "/api/insight/upload-start":
                self.handle_upload_start()
            elif path == "/api/session/start":
                self.handle_session_start()
            elif path == "/api/session/upload-start":
                self.handle_session_upload_start()
            elif path == "/api/session/prepare":
                self.handle_session_prepare()
            elif path == "/api/session/upload-prepare":
                self.handle_session_upload_prepare()
            elif path == "/api/session/run":
                self.send_json(run_prepared_session())
            elif path == "/api/session/reset":
                self.send_json(reset_session())
            elif path == "/api/app/start":
                self.send_json(start_app())
            elif path == "/api/app/stop":
                self.send_json(stop_app())
            elif path == "/api/telegram/test":
                self.handle_telegram_test()
            else:
                self.send_error(404, "unknown endpoint")
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=503)

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def send_json(self, payload, status=200):
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_media_proxy(self, raw_path):
        media_path = validate_media_path(raw_path)
        headers = {}
        if self.headers.get("Range"):
            headers["Range"] = self.headers["Range"]
        request = urllib.request.Request(
            f"{INSIGHT_BASE}/media/{urllib.parse.quote(media_path, safe='/')}",
            headers=headers,
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, context=SSL_CONTEXT, timeout=120) as response:
                self.send_response(response.status)
                self.send_header("Content-Type", response.headers.get("Content-Type", "video/mp4"))
                for header in ("Content-Length", "Content-Range", "Accept-Ranges", "Last-Modified", "ETag"):
                    value = response.headers.get(header)
                    if value:
                        self.send_header(header, value)
                if not response.headers.get("Accept-Ranges"):
                    self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except urllib.error.HTTPError as exc:
            self.send_error(exc.code, exc.reason)
        except Exception as exc:
            self.send_error(502, str(exc))

    def handle_status(self):
        insight = {"ok": False}
        source = {"index": INSIGHT_SOURCE_INDEX, "file": "", "state": "unknown"}
        view = ""
        try:
            health = insight_request("GET", "/api/health", timeout=5)
            insight = {"ok": True, "health": health}
            source = source_status()
            view = viewer_url()
        except Exception as exc:
            insight = {"ok": False, "error": str(exc)}
        self.send_json(
            {
                "insight": insight,
                "source": source,
                "viewer_url": view,
                "media_url": media_url(source.get("file")) if source.get("file") else "",
                "devkit_host": DEVKIT_HOST,
                "devkit_ssh_open": devkit_ssh_open(),
                "presets": [public_preset(preset) for preset in PRESETS],
                "scenario": current_scenario(),
                "app": app_state(),
                "telegram": public_telegram_status(),
                "vlm": vlm_settings(),
            }
        )

    def handle_telegram_test(self):
        payload = self.read_json()
        self.send_json(send_telegram_text(payload.get("message", "")))

    def handle_insight_start(self):
        payload = self.read_json()
        filename = start_insight_stream(str(payload.get("file", "")))
        clear_outputs()
        self.send_json(
            {
                "message": f"{filename} streaming on Insight source {INSIGHT_SOURCE_INDEX}",
                "source": source_status(),
                "media_url": media_url(filename),
                "viewer_url": viewer_url(),
            }
        )

    def handle_upload_start(self):
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
        )
        if "file" not in form:
            raise RuntimeError("missing file field")
        item = form["file"]
        if isinstance(item, list):
            item = item[0]
        filename = Path(item.filename or "").name
        data = item.file.read()
        if not filename or not data:
            raise RuntimeError("empty upload")
        upload_media_bytes(filename, data)
        filename = start_insight_stream(filename)
        clear_outputs()
        self.send_json(
            {
                "message": f"{filename} uploaded and streaming on Insight source {INSIGHT_SOURCE_INDEX}",
                "source": source_status(),
                "media_url": media_url(filename),
                "viewer_url": viewer_url(),
            }
        )

    def handle_session_start(self):
        payload = self.read_json()
        preset_id = str(payload.get("preset", ""))
        preset = PRESETS_BY_ID.get(preset_id)
        if not preset:
            raise RuntimeError(f"unknown preset: {preset_id}")
        if not preset["path"].exists():
            raise RuntimeError(f"preset video missing: {preset['path']}")
        self.send_json(start_analysis_session(preset["file"], public_preset(preset), preset["path"]))

    def handle_session_prepare(self):
        payload = self.read_json()
        preset_id = str(payload.get("preset", ""))
        preset = PRESETS_BY_ID.get(preset_id)
        if not preset:
            raise RuntimeError(f"unknown preset: {preset_id}")
        if not preset["path"].exists():
            raise RuntimeError(f"preset video missing: {preset['path']}")
        self.send_json(prepare_analysis_session(preset["file"], public_preset(preset), preset["path"]))

    def handle_session_upload_start(self):
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
        )
        if "file" not in form:
            raise RuntimeError("missing file field")
        item = form["file"]
        if isinstance(item, list):
            item = item[0]
        original = Path(item.filename or "").name
        data = item.file.read()
        if not original or not data:
            raise RuntimeError("empty upload")

        stop_app()
        stop_insight_source()
        cleanup_temp_uploads()
        temp_name = f"{TEMP_UPLOAD_PREFIX}{uuid.uuid4().hex[:10]}_{original}"
        upload_media_bytes(temp_name, data)
        scenario = {
            "id": "uploaded-video",
            "title": "Uploaded Video",
            "file": temp_name,
            "function": "custom_analysis",
            "expected": "analysis",
            "summary": "Runs the configured local fall detector on a temporary upload.",
            "available": True,
            "original_file": original,
            "temporary": True,
        }
        self.send_json(start_analysis_session(temp_name, scenario))

    def handle_session_upload_prepare(self):
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
        )
        if "file" not in form:
            raise RuntimeError("missing file field")
        item = form["file"]
        if isinstance(item, list):
            item = item[0]
        original = Path(item.filename or "").name
        data = item.file.read()
        if not original or not data:
            raise RuntimeError("empty upload")

        stop_app()
        stop_insight_source()
        cleanup_temp_uploads()
        temp_name = f"{TEMP_UPLOAD_PREFIX}{uuid.uuid4().hex[:10]}_{original}"
        upload_media_bytes(temp_name, data)
        scenario = {
            "id": "uploaded-video",
            "title": "Uploaded Video",
            "file": temp_name,
            "function": "custom_analysis",
            "expected": "analysis",
            "summary": "Runs the configured local fall detector on a temporary upload.",
            "available": True,
            "original_file": original,
            "temporary": True,
        }
        self.send_json(prepare_analysis_session(temp_name, scenario))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default=os.environ.get("HOMEANGEL_CONSOLE_BIND", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("HOMEANGEL_CONSOLE_PORT", "8765")))
    args = parser.parse_args()

    APP_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.bind, args.port), HomeAngelHandler)
    print(f"Serving HomeAngel demo console at http://{args.bind}:{args.port}/console/", flush=True)
    print(f"Insight API target: {INSIGHT_BASE}", flush=True)
    print(f"DevKit SSH target: {DEVKIT_HOST}:22", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
