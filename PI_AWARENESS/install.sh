#!/usr/bin/env bash
# One-time Raspberry Pi 5 provisioning. Run as the normal Pi user: bash install.sh
# Internet is needed only while installing packages/models. Runtime is offline.
set -Eeuo pipefail
trap 'echo "Installation stopped at line $LINENO. Correct the error above and rerun." >&2' ERR
cd "$(dirname "$(readlink -f "$0")")"
if [[ $EUID -eq 0 ]]; then
  echo 'Run bash install.sh as your normal Pi user, not sudo.' >&2
  exit 1
fi
[[ "$(uname -m)" == "aarch64" ]] || { echo 'This build requires Raspberry Pi OS 64-bit (aarch64).' >&2; exit 1; }
MODEL="$(tr -d '\0' </proc/device-tree/model 2>/dev/null || true)"
if [[ "$MODEL" != *"Raspberry Pi 5"* ]]; then
  echo "This hardware profile targets Raspberry Pi 5. Detected: ${MODEL:-unknown}" >&2
  exit 1
fi
python3 - <<'CHECK'
import sys
if not (3,11) <= sys.version_info[:2] <= (3,13):
    raise SystemExit('Supported baseline: Python 3.11–3.13 on Raspberry Pi OS 64-bit.')
CHECK
command -v raspi-config >/dev/null || { echo 'Raspberry Pi OS with raspi-config is required.' >&2; exit 1; }

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  python3-venv python3-pip python3-numpy python3-opencv python3-picamera2 \
  python3-cffi python3-libgpiod libportaudio2 libasound2-dev libatomic1 \
  espeak-ng alsa-utils i2c-tools v4l-utils ca-certificates unzip

# Configure the actual assembled buses. The googlevoicehat overlay exposes one
# full-duplex I2S card for the INMP441 capture and MAX98357A playback wiring.
CONFIG=/boot/firmware/config.txt
[[ -f "$CONFIG" ]] || CONFIG=/boot/config.txt
[[ -f "$CONFIG" ]] || { echo 'Could not find Raspberry Pi config.txt' >&2; exit 1; }
ensure_config_line() {
  local line="$1"
  if ! grep -Eq "^[[:space:]]*${line//./\\.}[[:space:]]*(#.*)?$" "$CONFIG"; then
    printf '\n[all]\n# PI_AWARENESS hardware\n%s\n' "$line" | sudo tee -a "$CONFIG" >/dev/null
    REBOOT_REQUIRED=1
  fi
}
REBOOT_REQUIRED=0
ensure_config_line 'dtparam=i2c_arm=on'
ensure_config_line 'dtparam=i2s=on'
ensure_config_line 'dtoverlay=googlevoicehat-soundcard'
ensure_config_line 'camera_auto_detect=1'
sudo raspi-config nonint do_i2c 0

python3 -m venv --system-site-packages .venv
NUMPY_VERSION="$(.venv/bin/python -c 'import numpy; print(numpy.__version__)')"
printf 'numpy==%s\n' "$NUMPY_VERSION" > .numpy-constraint.txt
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --no-cache-dir --only-binary=onnxruntime \
  -c .numpy-constraint.txt -r requirements.txt
.venv/bin/python - <<'PYDEPS'
# PI_AWARENESS direct runtime dependency check.
# Do not use `pip check` here: this venv intentionally exposes Raspberry Pi OS
# system packages, so pip can report unrelated broken optional/type packages.
import importlib
from importlib.metadata import PackageNotFoundError, version

modules = (
    "numpy",
    "cv2",
    "picamera2",
    "onnxruntime",
    "vosk",
    "sounddevice",
    "smbus2",
    "board",
    "busio",
    "adafruit_vl53l0x",
)

failed = []
for name in modules:
    try:
        importlib.import_module(name)
    except Exception as exc:
        failed.append(f"{name}: {type(exc).__name__}: {exc}")

expected = {
    "onnxruntime": "1.30.0",
    "vosk": "0.3.45",
    "sounddevice": "0.5.2",
    "smbus2": "0.5.0",
    "adafruit-circuitpython-vl53l0x": "3.6.19",
}

for package, wanted in expected.items():
    try:
        got = version(package)
    except PackageNotFoundError:
        failed.append(f"{package}: package metadata not found")
        continue
    if got != wanted:
        failed.append(f"{package}: expected {wanted}, found {got}")

if failed:
    print("PI_AWARENESS dependency validation FAILED:")
    for item in failed:
        print("  -", item)
    raise SystemExit(1)

print("PI_AWARENESS direct runtime dependency check: OK")
PYDEPS
mkdir -p state logs models
chmod 700 state logs
for group in audio video render i2c gpio; do
  getent group "$group" >/dev/null && sudo usermod -aG "$group" "$USER" || true
done

# One-time connected model provisioning. Runtime never downloads replacements.
.venv/bin/python tools/download_models.py
.venv/bin/python -m pip freeze > state/installed-packages.txt

# Software/model check can run before a reboot. I2S hardware may only appear after reboot.
.venv/bin/python server.py --doctor --deep

# This build is a service by default. Enable it now; do not start it before the
# overlay/group changes have definitely taken effect.
bash tools/install_service.sh

cat <<DONE

PI_AWARENESS provisioning complete.
Hardware profile:
  Camera: Raspberry Pi Camera Rev 1.3 / Picamera2
  Mic:    INMP441 I2S (GPIO18/19/20; physical pins 12/35/38)
  Audio:  MAX98357A I2S (GPIO18/19/21; physical pins 12/35/40)
  Range:  VL53L0X on I2C address 0x29

The pi-awareness service is enabled and will start automatically after reboot.
On a successful service start it says exactly: "I am ready"
Then it waits for voice commands with camera and ToF activity disabled until needed.

Reboot now:
  sudo reboot

After reboot, the service starts automatically. Verify it first:
  sudo systemctl status pi-awareness
  journalctl -u pi-awareness -b --no-pager

For an exclusive hardware doctor, temporarily stop the service:
  cd "$PWD"
  sudo systemctl stop pi-awareness
  .venv/bin/python server.py --doctor --deep --hardware
  sudo systemctl start pi-awareness
DONE
