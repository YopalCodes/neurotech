#!/usr/bin/env bash
# Makes the Pi start the camera stream and the command server by itself every time it turns
# on (as systemd services, restarted within seconds if either ever stops). Run once, on the Pi:
#
#     bash install_autostart.sh            # install, and start both now
#     bash install_autostart.sh --status   # are they running? recent log lines
#     bash install_autostart.sh --restart  # after a git pull, to run the new code
#     bash install_autostart.sh --remove   # back to starting them by hand
#
# Stream settings are remembered from the install, e.g.
#     RESOLUTION=1280x720 FPS=30 bash install_autostart.sh
#
# Live logs:  journalctl -u watchdog-stream -f      journalctl -u watchdog-commands -f
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_NAME="${SUDO_USER:-$(id -un)}"
PYTHON="$(command -v python3)"
SERVICES=(watchdog-stream watchdog-commands)

case "${1:-}" in
  --remove)
    sudo systemctl disable --now "${SERVICES[@]}" 2>/dev/null || true
    for s in "${SERVICES[@]}"; do sudo rm -f "/etc/systemd/system/$s.service"; done
    sudo systemctl daemon-reload
    echo "Removed. Start them by hand again: bash start_stream.sh / python3 command_server.py"
    exit 0 ;;
  --status)
    for s in "${SERVICES[@]}"; do systemctl --no-pager --lines=8 status "$s" || true; echo; done
    exit 0 ;;
  --restart)
    sudo systemctl restart "${SERVICES[@]}"
    sleep 3
    for s in "${SERVICES[@]}"; do echo "$s: $(systemctl is-active "$s")"; done
    exit 0 ;;
  "") ;;
  *) echo "Unknown option '$1'. Use --status, --restart or --remove."; exit 1 ;;
esac

if ! command -v ustreamer >/dev/null; then
  echo "ustreamer isn't installed. Run:  sudo apt install -y ustreamer   then this again."
  exit 1
fi

# Only list groups that exist on this Pi, or systemd refuses to start the service.
groups_that_exist() {
  local found=()
  for g in "$@"; do getent group "$g" >/dev/null && found+=("$g"); done
  echo "${found[*]:-}"
}
STREAM_GROUPS="$(groups_that_exist video)"
COMMAND_GROUPS="$(groups_that_exist gpio video)"

ENV_LINES=""
for v in DEVICE RESOLUTION FPS PORT; do
  if [ -n "${!v:-}" ]; then ENV_LINES+="Environment=$v=${!v}"$'\n'; fi
done

sudo tee /etc/systemd/system/watchdog-stream.service >/dev/null <<EOF
[Unit]
Description=Watchdog camera stream (ustreamer, port 8080)
After=network.target
StartLimitIntervalSec=0

[Service]
User=$USER_NAME
${STREAM_GROUPS:+SupplementaryGroups=$STREAM_GROUPS}
${ENV_LINES}ExecStart=/bin/bash "$DIR/start_stream.sh"
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo tee /etc/systemd/system/watchdog-commands.service >/dev/null <<EOF
[Unit]
Description=Watchdog command server (motor control, port 5005)
After=network.target
StartLimitIntervalSec=0

[Service]
User=$USER_NAME
${COMMAND_GROUPS:+SupplementaryGroups=$COMMAND_GROUPS}
WorkingDirectory=$DIR
ExecStart="$PYTHON" -u "$DIR/command_server.py"
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "${SERVICES[@]}" >/dev/null
sudo systemctl restart "${SERVICES[@]}"
sleep 3

echo
for s in "${SERVICES[@]}"; do echo "$s: $(systemctl is-active "$s")"; done
IP="$(hostname -I | awk '{print $1}')"
echo
echo "Both now start on every boot. Video: http://$IP:8080/stream   Commands: port 5005"
echo "If either says 'activating', a copy started by hand is probably still holding its port:"
echo "press Ctrl+C in that terminal and the service takes over within a few seconds."
echo "Check any time with:  bash install_autostart.sh --status"
