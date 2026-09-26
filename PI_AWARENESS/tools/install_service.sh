#!/usr/bin/env bash
# Install and enable the always-listening offline service.
# By default it starts at the next boot. Pass --now only after hardware is already active.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
cd "$ROOT"
if [[ $EUID -eq 0 ]]; then echo 'Run as the normal Pi user, not sudo.' >&2; exit 1; fi
if [[ ! "$ROOT" =~ ^/[a-zA-Z0-9_./-]+$ ]]; then
  echo 'Move the repo to a path containing only letters, digits, /, _, . or -.' >&2; exit 1
fi
[[ -x "$ROOT/.venv/bin/python" ]] || { echo 'Run install.sh first.' >&2; exit 1; }
START_NOW=0
[[ "${1:-}" == "--now" ]] && START_NOW=1
mkdir -p state logs
GROUPS_LINE=''
for group in audio video render i2c gpio; do
  if getent group "$group" >/dev/null; then GROUPS_LINE+=" $group"; fi
done
TEMP="$(mktemp)"
trap 'rm -f "$TEMP"' EXIT
cat > "$TEMP" <<UNIT
[Unit]
Description=PI AWARENESS offline voice-first assistive service
After=local-fs.target sound.target
StartLimitIntervalSec=180
StartLimitBurst=8

[Service]
Type=simple
User=$USER
WorkingDirectory=$ROOT
ExecStart=$ROOT/.venv/bin/python -u $ROOT/server.py
SupplementaryGroups=$GROUPS_LINE
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONDONTWRITEBYTECODE=1
Environment=OMP_NUM_THREADS=2
Environment=OPENBLAS_NUM_THREADS=1
Environment=NUMEXPR_NUM_THREADS=1
Restart=on-failure
RestartSec=5
TimeoutStopSec=15
PrivateNetwork=true
RestrictAddressFamilies=AF_UNIX AF_NETLINK
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=$ROOT/state $ROOT/logs
UMask=0077

[Install]
WantedBy=multi-user.target
UNIT
sudo install -m 644 "$TEMP" /etc/systemd/system/pi-awareness.service
sudo systemctl daemon-reload
sudo systemctl enable pi-awareness.service
if (( START_NOW )); then
  sudo systemctl restart pi-awareness.service
  echo 'Service started now.'
else
  echo 'Service enabled. It will start automatically on the next boot.'
fi
printf 'Status: sudo systemctl status pi-awareness\nLogs:   journalctl -u pi-awareness -f\nStop:   sudo systemctl disable --now pi-awareness\n'
