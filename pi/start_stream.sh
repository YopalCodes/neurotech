#!/usr/bin/env bash
# Streams the USB webcam over Wi-Fi as MJPEG: http://<pi-ip>:8080/stream
# The webcam already compresses to MJPEG, so ustreamer passes frames straight through
# and the Pi's CPU stays nearly idle.
#
# Override any setting:  RESOLUTION=1280x720 FPS=15 bash start_stream.sh
set -euo pipefail

DEVICE="${DEVICE:-/dev/video0}"
RESOLUTION="${RESOLUTION:-640x480}"
FPS="${FPS:-30}"
PORT="${PORT:-8080}"

exec ustreamer \
  --device="$DEVICE" \
  --format=MJPEG \
  --resolution="$RESOLUTION" \
  --desired-fps="$FPS" \
  --host=0.0.0.0 \
  --port="$PORT"
