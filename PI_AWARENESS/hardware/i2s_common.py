from __future__ import annotations
from pathlib import Path
import re

CARD_MARKER = "sndrpigooglevoi"
OVERLAY = "googlevoicehat-soundcard"
CONFIG_PATHS = (Path("/boot/firmware/config.txt"), Path("/boot/config.txt"))


def find_i2s_card_number() -> int:
    """Return the ALSA card number for the shared INMP441/MAX98357A I2S card."""
    path = Path("/proc/asound/cards")
    if not path.exists():
        raise RuntimeError("/proc/asound/cards does not exist; ALSA is unavailable")
    for line in path.read_text(errors="replace").splitlines():
        if CARD_MARKER in line:
            match = re.match(r"\s*(\d+)\s+\[", line)
            if match:
                return int(match.group(1))
            parts = line.split()
            if parts and parts[0].isdigit():
                return int(parts[0])
    raise RuntimeError(
        "I2S sound card not found. Expected sndrpigooglevoi. "
        "Check dtoverlay=googlevoicehat-soundcard and reboot."
    )


def find_i2s_device() -> str:
    """Use plughw so ALSA can convert playback formats when necessary."""
    return f"plughw:{find_i2s_card_number()},0"


def active_config_path() -> Path | None:
    return next((p for p in CONFIG_PATHS if p.exists()), None)


def overlay_is_configured() -> bool:
    path = active_config_path()
    if not path:
        return False
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        if line == f"dtoverlay={OVERLAY}" or line.startswith(f"dtoverlay={OVERLAY},"):
            return True
    return False
