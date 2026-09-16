# HomeAngel AI

HomeAngel AI is a privacy-preserving fall-detection proof of concept built for the
SiMa.ai Modalix DevKit. It uses YOLO26 pose inference on the MLA to understand a
person's posture and movement, then emits only a small JSON alert when a fall is
confirmed. Video stays local to the device and local demo console.

The application code lives in [`labs/homeangel-ai`](labs/homeangel-ai/).

## What It Demonstrates

HomeAngel is designed around a simple product promise: families get useful safety
alerts without receiving a video feed from inside the home. The edge device can see
locally because it must process video to detect a fall, but raw frames are never
written to disk and never sent to Telegram, webhook endpoints, or any remote API.

For the demo, judges see a local diagnostic console that shows what happens inside
the unit: video preview, pose-derived state, risk level, velocity and torso-angle
graphs, performance numbers, event timeline, and a phone-style alert panel. This
console is local-only and is not the production user interface. The production
interface is the family message.

## How It Works

1. Neat Insight streams a local MP4 or local camera-like source over RTSP.
2. HomeAngel runs YOLO26 pose on the Modalix MLA.
3. A CPU tracker assigns stable `track_id` values across frames.
4. Per tracked person, the app computes centroid motion, torso angle, vertical
   velocity, angular velocity, and stillness.
5. A state machine moves through `UPRIGHT`, `FALLING`, `DOWN`, and
   `FALL_CONFIRMED`.
6. On confirmation, the privacy-boundary module emits one JSON event to
   `events.log` and optionally sends the same text-only metadata to Telegram or a
   webhook.
7. For the single-fall demo, an optional local Gemma VLM stage can add a short
   scene description after confirmation. The frame remains in memory and only the
   resulting text joins the event.

## Privacy Boundary

The alert module receives event metadata only. It has no access to frame buffers.
The only outbound payload is shaped like:

```json
{
  "event": "fall_detected",
  "device_id": "modalix-hallway-01",
  "zone_label": "Bedroom",
  "timestamp_s": 7.17,
  "track_id": 1,
  "confidence": 0.91,
  "description": "Optional local VLM text description."
}
```

Local diagnostic files:

- `events.log`: JSON Lines event stream, safe to send out.
- `telemetry.json`: local-only demo telemetry for the console.

Video files, decoded frames, and annotated frames are not persisted by the app and
are not sent through the notifier.

## Demo Modes

The console supports these local presets:

- `Fall Alert`: one fall clip. Expected result is one fall event, a risky state,
  and a Telegram-style family message.
- `Multi-Room Scan`: four different local feeds. Bedroom contains the fall clip;
  Hallway, Kitchen, and Living Room contain safe daily-activity clips. The system
  identifies which room became risky and emits an event naming that zone/feed.
- `Routine Activity`: one daily-activity clip. Expected result is no fall alert.

Users can also upload a local MP4 through the console for ad hoc testing. Uploaded
clips are assigned to Insight locally and cleaned up by the console session flow.

## Repository Layout

```text
labs/homeangel-ai/
  src/main.cpp                 C++ inference, tracking, fall logic, telemetry, alerts
  console/index.html           Offline local demo console
  console/server.py            Local console API, Insight control, DevKit runner
  dashboard/index.html         Minimal event dashboard
  config.yaml                  SDK validation config
  config.devkit.yaml           DevKit runtime config
  config.devkit.*.yaml         Smoke, fall, and ADL test configs
  AngelStart                   One-command session starter
  serve_console.sh             Console server helper
  run_homeangel.sh             Local validation wrapper
  run_on_devkit.sh             DevKit execution wrapper
  vlm_server.py                Local VLM server wrapper
  .env.example                 Telegram environment template
```

## Quick Start

Run from inside the SiMa Neat SDK container:

```bash
cd /workspace/labs/homeangel-ai
./AngelStart
```

Open the console:

```text
http://127.0.0.1:8765/console/
```

The start script checks Neat Insight, checks DevKit access, builds if needed,
starts optional local VLM support when configured, and serves the local console.

To validate without starting a full console session:

```bash
cd /workspace/labs/homeangel-ai
./AngelStart --check-only
```

## Build Manually

```bash
cd /workspace/labs/homeangel-ai
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Validate config from the SDK side:

```bash
/workspace/labs/homeangel-ai/run_homeangel.sh \
  --config /workspace/labs/homeangel-ai/config.yaml \
  --validate-config-only
```

Run the app on the DevKit so it can use the MLA:

```bash
/workspace/labs/homeangel-ai/run_on_devkit.sh \
  --config /workspace/labs/homeangel-ai/config.devkit.yaml
```

## Telegram Alerts

Telegram is configured through environment variables, not YAML secrets. Copy the
example and fill in your local values:

```bash
cp /workspace/labs/homeangel-ai/.env.example /workspace/labs/homeangel-ai/.env.local
nano /workspace/labs/homeangel-ai/.env.local
```

Required values:

```bash
export HOMEANGEL_TELEGRAM_BOT_TOKEN="<bot-token>"
export HOMEANGEL_TELEGRAM_CHAT_ID="<chat-id>"
```

The app routes `zone_label` or `device_id` values to configured chat IDs. The bot
token is ignored by Git and is never exposed to the browser.

## Performance Reporting

The app reports:

- Per-frame inference latency in milliseconds.
- Average and p95 latency on exit.
- End-to-end FPS and total frames processed.
- Compute placement such as `pose:MLA`, `tracking:CPU`, and `fall_logic:CPU`.

The console also polls local telemetry for live FPS, latency, state, risk, and
threshold graphs.

## Demo Script

1. Start the project with `./AngelStart`.
2. Open `http://127.0.0.1:8765/console/`.
3. Pick `Fall Alert` to show one fall, VLM description, and text-only alert.
4. Pick `Multi-Room Scan` to show four simultaneous feeds and zone-level triage.
5. Point out the privacy boundary: the video remains on the left/local side, while
   only a small JSON/text event appears on the right/outbound side.
6. Tail the event stream as a terminal proof:

```bash
tail -f /workspace/labs/homeangel-ai/events.log
```

## More Documentation

The full operator README, including Insight setup, ADL false-alarm testing, VLM
configuration, local webhook testing, and privacy proof commands, is here:

[`labs/homeangel-ai/README.md`](labs/homeangel-ai/README.md)
