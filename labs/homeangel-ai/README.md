# HomeAngel AI

Privacy-preserving fall detection for Modalix DevKit using SiMa.ai Neat, Neat Insight, and YOLO26 pose.

HomeAngel AI keeps video fully local. Frames are decoded and used for inference in memory, and the live view is sent only to local Neat Insight. Confirmed alerts cross the privacy boundary as JSON Lines events; the demo console also reads local diagnostic telemetry from `telemetry.json`. Neither output contains frames.

## What This Extends

- `/neat-resources/apps-src/examples/pose-estimation/multi-stream-pose-estimator`
  - Base RTSP input, YOLO26 pose model, synchronized Insight video, and pose metadata.
- `/neat-resources/apps-src/examples/tracking/multi-stream-people-tracker`
  - Reuses the C++ IoU tracker helper for stable `track_id` values.
- `/neat-resources/apps-src/examples/genai/detection-to-vlm-assistant`
  - Reference for the optional local GenAI server and crop-to-VLM confirmation stage.
    HomeAngel routes to Gemma first when it is deployed, with Qwen3-VL as a fallback.

## Files

- App root: `/workspace/labs/homeangel-ai`
- Binary: `/workspace/labs/homeangel-ai/build/homeangel-ai`
- SDK validation config: `/workspace/labs/homeangel-ai/config.yaml`
- DevKit runtime config: `/workspace/labs/homeangel-ai/config.devkit.yaml`
- DevKit smoke-test config: `/workspace/labs/homeangel-ai/config.devkit.smoke.yaml`
- DevKit finite fall-test config: `/workspace/labs/homeangel-ai/config.devkit.falltest.yaml`
- DevKit finite ADL false-alarm config: `/workspace/labs/homeangel-ai/config.devkit.adltest.yaml`
- Events: `/workspace/labs/homeangel-ai/events.log`
- Local telemetry: `/workspace/labs/homeangel-ai/telemetry.json`
- Demo console: `/workspace/labs/homeangel-ai/console/index.html`
- Demo console server: `/workspace/labs/homeangel-ai/console/server.py`
- Console server helper: `/workspace/labs/homeangel-ai/serve_console.sh`
- Local VLM server helper: `/workspace/labs/homeangel-ai/run_vlm_server.sh`
- Local VLM server source: `/workspace/labs/homeangel-ai/vlm_server.py`
- Dashboard: `/workspace/labs/homeangel-ai/dashboard/index.html`
- Fall clip: `/workspace/assets/videos/fall/demo.mp4`
- ADL clip: `/workspace/assets/videos/fall/adl_sitdown.mp4`

## Build

Run inside the Neat SDK container:

```bash
cd /workspace/labs/homeangel-ai
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

## One-Command Start

For the next session, run this inside the Neat SDK container:

```bash
cd /workspace/labs/homeangel-ai
./AngelStart
```

It checks Insight and DevKit access, builds if needed, validates
`config.devkit.yaml`, starts the optional local VLM server when enabled, and then
serves the demo console. Open:

```text
http://127.0.0.1:8765/console/
```

Use `./AngelStart --check-only` to verify the session without starting the console.

Download the default YOLO26 pose model if it is not present:

```bash
cd /workspace/labs/homeangel-ai
sima-cli download \
  "https://docs.sima.ai/pkg_downloads/SDK2.1.3/models/modalix/yolo26-pose/yolo26m-pose-int8-b1.tar.gz" \
  --dest models
```

Validate the config:

```bash
/workspace/labs/homeangel-ai/run_homeangel.sh \
  --config /workspace/labs/homeangel-ai/config.yaml \
  --validate-config-only
```

The shell wrapper above is for SDK-side validation. The DevKit runner launches
ARM64 binaries directly.

Validate the DevKit runtime config:

```bash
/workspace/labs/homeangel-ai/run_homeangel.sh \
  --config /workspace/labs/homeangel-ai/config.devkit.yaml \
  --validate-config-only
```

## Insight Setup

Open Insight:

```text
https://192.168.1.10:9900
```

Import and stream the fall clip through the UI:

1. Open `Media Sources`.
2. Upload `/workspace/assets/videos/fall/demo.mp4`.
3. Go to `Streaming`.
4. Assign `demo.mp4` to source `1`.
5. Start source `1`.
6. Confirm the RTSP URL is `rtsp://127.0.0.1:8554/src1` inside the SDK container.
   When the app runs on the DevKit, use `/workspace/labs/homeangel-ai/config.devkit.yaml`;
   it points the DevKit back to SDK Insight at `rtsp://192.168.1.10:8554/src1`.
7. Open `Video Viewer`, channel `0`.

Equivalent API commands:

```bash
curl -k -F "file=@/workspace/assets/videos/fall/demo.mp4" \
  https://127.0.0.1:9900/api/upload/media
curl -k -H "Content-Type: application/json" \
  -d '{"index":1,"file":"demo.mp4","transport":"rtsp"}' \
  https://127.0.0.1:9900/api/mediasrc/assign
curl -k -H "Content-Type: application/json" \
  -d '{"index":1}' \
  https://127.0.0.1:9900/api/mediasrc/start
curl -k "https://127.0.0.1:9900/api/viewer-url?mode=light&src=0"
```

## Run

Run inference through `dk` so the process executes on the Modalix DevKit and can
use the MLA. Running the binary directly inside the SDK container can validate
configuration, but full inference will fail with `neat.dispatcher_unavailable`.

```bash
rm -f /workspace/labs/homeangel-ai/events.log /workspace/labs/homeangel-ai/telemetry.json
dk /workspace/labs/homeangel-ai/build/homeangel-ai \
  --config /workspace/labs/homeangel-ai/config.devkit.yaml
```

Stop with `Ctrl-C`. On exit, the app prints average FPS, average latency, p95 latency, total frames, and compute placement.

Equivalent helper:

```bash
/workspace/labs/homeangel-ai/run_on_devkit.sh
```

## Optional Gemma VLM Descriptions

The VLM stage is disabled by default. When enabled, it runs only after
`FALL_CONFIRMED`, sends one in-memory crop to a loopback GenAI server, and appends
the returned text as `description` in the same JSON event. Raw frames are never
written to disk and are never sent to Telegram or webhook endpoints.

The model route is configured in `config.yaml` and `config.devkit.yaml`:

```yaml
vlm:
  enabled: false
  host: 127.0.0.1
  port: 9998
  models: Gemma-4-E4B-it,Qwen3-VL-4B-Instruct-GPTQ-a16w4
  model_paths: Gemma-4-E4B-it=/media/nvme/llima/models/Gemma-4-E4B-it,Qwen3-VL-4B-Instruct-GPTQ-a16w4=/media/nvme/llima/models/Qwen3-VL-4B-Instruct-GPTQ-a16w4
```

Deploy or copy the Gemma 4 E4B LLiMa model directory onto the DevKit, then update
`vlm.model_paths` if the local directory or served name differs. Check what the
launcher can see:

```bash
dk /bin/bash /workspace/labs/homeangel-ai/run_vlm_server.sh \
  --config /workspace/labs/homeangel-ai/config.devkit.yaml \
  --check-config
```

Start the local GenAI server in a separate terminal on the DevKit:

```bash
dk /bin/bash /workspace/labs/homeangel-ai/run_vlm_server.sh \
  --config /workspace/labs/homeangel-ai/config.devkit.yaml
```

Then set `vlm.enabled: true` in `/workspace/labs/homeangel-ai/config.devkit.yaml`
and run HomeAngel normally. If Gemma is not installed but the Qwen fallback path is
installed, the launcher serves Qwen and the app falls back to that model name.

Finite smoke test:

```bash
dk /workspace/labs/homeangel-ai/build/homeangel-ai \
  --config /workspace/labs/homeangel-ai/config.devkit.smoke.yaml
```

Finite full fall-clip test:

```bash
rm -f /workspace/labs/homeangel-ai/events.log /workspace/labs/homeangel-ai/telemetry.json
dk /workspace/labs/homeangel-ai/build/homeangel-ai \
  --config /workspace/labs/homeangel-ai/config.devkit.falltest.yaml
tail -f /workspace/labs/homeangel-ai/events.log
```

Expected result for the staged fall clip is one JSON line similar to:

```json
{"confidence":0.578,"device_id":"modalix-hallway-01","event":"fall_detected","timestamp_s":7.173,"track_id":1,"zone_label":"bedroom"}
```

## Demo Console

The demo console is a local diagnostic view for judges. It reads two local streams:

- Stream A: `telemetry.json`, including state, velocity/angle history, FPS, latency, and compute placement.
- Stream B: `events.log`, the JSON event stream that also feeds webhook or Telegram alerts.

The console live-feed pane shows only the selected video through a same-origin local media proxy. It starts paused, fits inside the pane, and displays a large center play control. The pose/state, risk, graphs, events, and phone mock are rendered outside that video surface from local telemetry and JSON events. Use the direct Insight viewer only when you need low-level overlay debugging.

Serve it locally:

```bash
cd /workspace/labs/homeangel-ai
./serve_console.sh
```

Open:

```text
http://127.0.0.1:8765/console/
```

If port `8765` is already in use:

```bash
cd /workspace/labs/homeangel-ai
HOMEANGEL_CONSOLE_PORT=8766 ./serve_console.sh
```

To view from another trusted machine on the same local demo network, bind to the container interfaces and browse through the SDK host address/port that your environment exposes:

```bash
cd /workspace/labs/homeangel-ai
HOMEANGEL_CONSOLE_BIND=0.0.0.0 ./serve_console.sh
```

Keep this server local to the demo network. It is not part of the production family interface.

The console opens with a source-selection page. Pick one of three cached local presets or upload a local video file:

- `Fall Alert`: `demo.mp4`, expected to emit a `fall_detected` JSON event and move the risk bar to `Risky`.
- `Sit-Down Check`: `adl_sitdown.mp4`, expected to show safe sit-down behavior without an alert.
- `Routine Activity`: `routine_adl.mp4`, pulled from the local GMDCSA cache and expected to stay safe.

`Analyze Selected` uploads/assigns the clip to Insight source `1`, moves to the analysis page, and loads the video paused. Press the large center control in the video pane to start the local Insight stream and launch the HomeAngel app on the DevKit with `dk`. The live-feed toggle can hide the local video preview while the detection pipeline continues.

Uploaded clips are sent to Insight under a temporary `homeangel_upload_*` name and are cleaned up on session reset or the next upload. Presets are cached local demo assets.

For the full fall demo from a terminal, the console buttons perform the same sequence as:

```bash
curl -H "Content-Type: application/json" \
  -d '{"preset":"fall-alert"}' \
  http://127.0.0.1:8765/api/session/prepare
curl -H "Content-Type: application/json" \
  -d '{}' \
  http://127.0.0.1:8765/api/session/run
```

Other preset IDs are `sit-down` and `routine-adl`. Uploaded videos use the browser file picker; the server gives them a temporary local `homeangel_upload_*` name and removes them on reset.

If you want to bypass the console server and use Insight directly:

```bash
curl -k -H "Content-Type: application/json" \
  -d '{"index":1,"file":"demo.mp4","transport":"rtsp"}' \
  https://127.0.0.1:9900/api/mediasrc/assign
curl -k -H "Content-Type: application/json" \
  -d '{"index":1}' \
  https://127.0.0.1:9900/api/mediasrc/start
rm -f /workspace/labs/homeangel-ai/events.log /workspace/labs/homeangel-ai/telemetry.json
dk /workspace/labs/homeangel-ai/build/homeangel-ai \
  --config /workspace/labs/homeangel-ai/config.devkit.yaml
```

Watch the left side for pose/state, velocity and torso-angle threshold crossings, FPS/latency, and the privacy-boundary animation. Watch the right side for the event card, byte counter, and phone-style Telegram message text.

## Minimal Event Dashboard

Serve the project folder locally:

```bash
cd /workspace/labs/homeangel-ai
python3 -m http.server 8765
```

Open:

```text
http://127.0.0.1:8765/dashboard/
```

Terminal fallback:

```bash
tail -f /workspace/labs/homeangel-ai/events.log
```

## ADL False-Alarm Test

Switch Insight source `1` to the ADL clip:

```bash
curl -k -F "file=@/workspace/assets/videos/fall/adl_sitdown.mp4" \
  https://127.0.0.1:9900/api/upload/media
curl -k -H "Content-Type: application/json" \
  -d '{"index":1,"file":"adl_sitdown.mp4","transport":"rtsp"}' \
  https://127.0.0.1:9900/api/mediasrc/assign
curl -k -H "Content-Type: application/json" \
  -d '{"index":1}' \
  https://127.0.0.1:9900/api/mediasrc/start
rm -f /workspace/labs/homeangel-ai/events.log /workspace/labs/homeangel-ai/telemetry.json
dk /workspace/labs/homeangel-ai/build/homeangel-ai \
  --config /workspace/labs/homeangel-ai/config.devkit.adltest.yaml
```

Expected result: no `fall_detected` line. If `events.log` is not created, that is also
passing: no alert was emitted. If it false-fires, raise `confirm_seconds` first,
then raise `fall_velocity_threshold`.

## Telegram Alert Action

Telegram is disabled by default. To enable it, put only the routing in
`/workspace/labs/homeangel-ai/config.devkit.yaml`:

```yaml
alerts:
  telegram:
    enabled: true
    token_env: HOMEANGEL_TELEGRAM_BOT_TOKEN
    routes: bedroom=<chat-id>
```

Then export the bot token in the terminal before starting the console or running
the app. Do not put the token in YAML:

```bash
export HOMEANGEL_TELEGRAM_BOT_TOKEN="<bot-token>"
```

Routes are comma-separated `selector=chat_id` pairs. Selectors may be
`zone_label`, `device_id`, or `*`.

The demo console's family-phone panel has a text box and `Send` button for a
manual text-only Telegram test. The message box stays empty while the local video
is running; when the video ends, the console builds the final family message from
the JSON event stream. If no fall event exists, no family message is generated.
The browser never receives the token; the local console server reads
`HOMEANGEL_TELEGRAM_BOT_TOKEN` from its environment and sends only the message
text to the configured chat IDs. When the detector emits a real fall event, the
HomeAngel app sends the same event metadata plus optional VLM text description.
Raw frames are never sent.

## Local Webhook Demo

Set `webhook_url: http://127.0.0.1:8787/alert` in `config.yaml`, then run a local listener:

```bash
python3 -u -c 'from http.server import BaseHTTPRequestHandler,HTTPServer; exec("class H(BaseHTTPRequestHandler):\n def do_POST(self):\n  n=int(self.headers.get(\"content-length\",0)); print(self.rfile.read(n).decode(), flush=True); self.send_response(204); self.end_headers()"); HTTPServer(("127.0.0.1",8787),H).serve_forever()'
```

## Insight Metrics

Show CPU, memory, and MLA stats from Insight:

```bash
curl -k https://127.0.0.1:9900/api/metrics
```

Debug video/metadata ingest:

```bash
curl -k "https://127.0.0.1:9900/api/ingest/stats?all=1&verbose=1"
```

If inference works but the Insight overlay does not appear, verify whether UDP
from the DevKit reaches the SDK Insight vf receiver:

```bash
bash -lic 'dk shell "python3 -c '\''import socket; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.sendto(b\"probe\", (\"192.168.1.10\", 9100)); print(\"sent\")'\''"'
curl -k "https://127.0.0.1:9900/api/ingest/stats?all=1&verbose=1"
```

If `metadata.messages_received` does not increase, the SDK container's UDP port
mapping is not reachable from the DevKit. The app and MLA path are still healthy;
ask for the SDK container to publish UDP `9000-9003` and `9100-9103`, or run
Insight/vf on a target reachable by the DevKit.

## Privacy Proof

Run this before and during the app. It should show no image or video outputs under the app folder. The expected application outputs are `events.log` and local metadata-only `telemetry.json`.

```bash
find /workspace/labs/homeangel-ai -type f \
  \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mov' -o -iname '*.mkv' \) \
  -print
```

Also confirm the source does not call image/video file writers. `cv::imencode` is
allowed because it keeps the confirmed-fall crop in memory for the loopback VLM
request:

```bash
rg -n "imwrite|VideoWriter|ofstream\\([^)]*\\.(jpg|jpeg|png|mp4|avi|mov|mkv)" \
  /workspace/labs/homeangel-ai/src || true
```

The outbound notifier path receives only the event JSON object. `telemetry.json` and the local video preview are local diagnostic surfaces for the demo console, not remote product outputs.
