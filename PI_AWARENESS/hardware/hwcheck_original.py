#!/usr/bin/env python3
"""Hardware sanity check for the Gia build on a Raspberry Pi.

Verifies what is actually live right now:
  * the I2C bus comes up and is scannable
  * a device answers at 0x29, and which chip it is
  * whether any ALSA playback/capture cards exist

At this stage the audio lines are *expected* to say "no card" -- the device
tree overlay has not been written yet. That is not a wiring fault.

Usage:
    python3 hwcheck.py            # bus 1 (the 40-pin header)
    python3 hwcheck.py --bus 0    # bus 0, if you are on the CAM/ID pins

Requires:
    pip install smbus2 --break-system-packages
"""

import argparse
import os
import re
import subprocess
import sys

I2C_TARGET = 0x29

# Addresses i2cdetect probes with a read rather than a write-quick, because a
# write to these can latch state on EEPROMs and a few real-time clocks.
READ_PROBE_RANGES = ((0x30, 0x37), (0x50, 0x5F))

CONFIG_PATHS = ("/boot/firmware/config.txt", "/boot/config.txt")


# ---------------------------------------------------------------- output


class Out:
    """Minimal status printer. Colours only when stdout is a terminal."""

    _tty = sys.stdout.isatty()

    @classmethod
    def _c(cls, code, text):
        return f"\033[{code}m{text}\033[0m" if cls._tty else text

    @classmethod
    def section(cls, title):
        print()
        print(cls._c("1", title))
        print(cls._c("2", "-" * len(title)))

    @classmethod
    def ok(cls, msg):
        print(f"  [{cls._c('32', ' ok ')}] {msg}")

    @classmethod
    def warn(cls, msg):
        print(f"  [{cls._c('33', 'warn')}] {msg}")

    @classmethod
    def fail(cls, msg):
        print(f"  [{cls._c('31', 'fail')}] {msg}")

    @classmethod
    def info(cls, msg):
        print(f"  [    ] {msg}")

    @classmethod
    def expected(cls, msg):
        print(f"  [{cls._c('36', 'todo')}] {msg}")


def run(cmd):
    """Run a command, returning (rc, stdout+stderr). rc 127 if not installed."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except FileNotFoundError:
        return 127, ""
    except subprocess.TimeoutExpired:
        return 124, ""


# ---------------------------------------------------------------- platform


def check_platform():
    Out.section("Platform")

    model = "unknown"
    try:
        with open("/proc/device-tree/model") as fh:
            model = fh.read().strip("\x00").strip()
    except OSError:
        pass

    if "Raspberry Pi" in model:
        Out.ok(model)
    else:
        Out.warn(f"not a Raspberry Pi? /proc/device-tree/model reports: {model}")

    Out.info(f"python {sys.version.split()[0]}")

    for path in CONFIG_PATHS:
        if not os.path.exists(path):
            continue
        try:
            with open(path) as fh:
                text = fh.read()
        except OSError as exc:
            Out.warn(f"{path}: {exc}")
            break

        live = [
            ln.strip()
            for ln in text.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        if any(re.match(r"dtparam=i2c_arm=on\b", ln) for ln in live):
            Out.ok(f"dtparam=i2c_arm=on present in {path}")
        else:
            Out.fail(f"dtparam=i2c_arm=on NOT active in {path} -- add it and reboot")

        overlays = [ln for ln in live if ln.startswith("dtoverlay=")]
        if overlays:
            Out.info("overlays loaded from config.txt: " + ", ".join(overlays))
        else:
            Out.expected("no dtoverlay= lines yet (expected -- audio overlay comes next)")
        break
    else:
        Out.warn("no config.txt found at either /boot/firmware or /boot")


# ---------------------------------------------------------------- i2c bus


def check_bus_node(bus):
    Out.section(f"I2C bus {bus}")

    node = f"/dev/i2c-{bus}"
    if not os.path.exists(node):
        Out.fail(f"{node} does not exist")
        Out.info("the kernel has no I2C adapter -- causes, most likely first:")
        Out.info("  1. dtparam=i2c_arm=on missing from config.txt (then reboot)")
        Out.info("  2. i2c-dev module not loaded: sudo modprobe i2c-dev")
        return False

    Out.ok(f"{node} present")
    if not os.access(node, os.R_OK | os.W_OK):
        Out.fail(f"no read/write access to {node}")
        Out.info("add yourself to the i2c group: sudo usermod -aG i2c $USER, then log out and back in")
        return False

    Out.ok("read/write access confirmed")
    return True


def scan_bus(smbus, bus):
    """Return the sorted list of addresses that acknowledge on the bus."""
    found = []
    try:
        with smbus.SMBus(bus) as sm:
            for addr in range(0x03, 0x78):
                use_read = any(lo <= addr <= hi for lo, hi in READ_PROBE_RANGES)
                try:
                    if use_read:
                        sm.read_byte(addr)
                    else:
                        sm.write_quick(addr)
                    found.append(addr)
                except OSError:
                    continue
    except (OSError, PermissionError) as exc:
        Out.fail(f"could not open bus {bus}: {exc}")
        return None
    return found


def report_scan(found):
    if found is None:
        return

    if not found:
        Out.fail("bus scanned clean -- no devices acknowledged at any address")
        Out.info("likely causes, in order:")
        Out.info("  1. the fifth pin on the clone board is holding the chip in reset")
        Out.info("     (XSHUT / GPIO0 -- tie it high to 3V3, or drive it high)")
        Out.info("  2. SDA and SCL swapped -- easy on that board, the silkscreen")
        Out.info("     runs SCL before SDA, the reverse of the Pi header order")
        Out.info("  3. dtparam=i2c_arm=on missing from config.txt")
        Out.info("  4. no pull-ups / unpowered board -- check 3V3 and GND first")
        return

    Out.ok(f"{len(found)} device(s) responding: " + ", ".join(f"0x{a:02x}" for a in found))
    if I2C_TARGET in found:
        Out.ok(f"0x{I2C_TARGET:02x} present -- this is the one we want")
    else:
        Out.fail(f"nothing at 0x{I2C_TARGET:02x}")
        Out.info("something is on the bus, so wiring and pull-ups are fine;")
        Out.info("the board is answering on a different address than expected")


# ---------------------------------------------------------------- chip id

# Everything that commonly sits at 0x29, and how to tell it apart.
# Each probe: (name, reader, expected-values, library to install)
#
# VL53L0X  reg 0xC0 (8-bit)    -> 0xEE
# VL53L1X  reg 0x010F (16-bit) -> 0xEA 0xCC
# TSL2591  reg 0x12 | 0xA0     -> 0x50
# TCS34725 reg 0x12 | 0x80     -> 0x44 / 0x4D / 0x10


def _r8(sm, addr, reg):
    return sm.read_byte_data(addr, reg)


def _r16reg(sm, addr, reg16):
    """Read one byte from a 16-bit register address (ST ToF style)."""
    from smbus2 import i2c_msg

    write = i2c_msg.write(addr, [(reg16 >> 8) & 0xFF, reg16 & 0xFF])
    read = i2c_msg.read(addr, 2)
    sm.i2c_rdwr(write, read)
    return list(read)


def identify_chip(smbus, bus):
    Out.section(f"Chip identification at 0x{I2C_TARGET:02x}")

    results = []
    try:
        with smbus.SMBus(bus) as sm:
            # VL53L1X -- 16-bit register space, model id 0xEACC
            try:
                data = _r16reg(sm, I2C_TARGET, 0x010F)
                results.append(("VL53L1X model id 0x010F", f"0x{data[0]:02x}{data[1]:02x}"))
                if data[0] == 0xEA and data[1] == 0xCC:
                    Out.ok("VL53L1X time-of-flight ranger (model id 0xEACC)")
                    Out.info("install: pip install VL53L1X --break-system-packages")
                    return "VL53L1X"
            except OSError:
                pass

            # VL53L0X -- 8-bit register space, model id 0xEE
            try:
                model = _r8(sm, I2C_TARGET, 0xC0)
                rev = _r8(sm, I2C_TARGET, 0xC2)
                results.append(("VL53L0X model id 0xC0", f"0x{model:02x}"))
                if model == 0xEE:
                    Out.ok(f"VL53L0X time-of-flight ranger (model 0xEE, rev 0x{rev:02x})")
                    Out.info("install: pip install adafruit-circuitpython-vl53l0x --break-system-packages")
                    return "VL53L0X"
            except OSError:
                pass

            # TSL2591 -- command bit 0xA0, ID 0x50
            try:
                ident = _r8(sm, I2C_TARGET, 0xA0 | 0x12)
                results.append(("TSL2591 id 0xB2", f"0x{ident:02x}"))
                if ident == 0x50:
                    Out.ok("TSL2591 ambient light sensor (id 0x50)")
                    Out.info("install: pip install adafruit-circuitpython-tsl2591 --break-system-packages")
                    return "TSL2591"
            except OSError:
                pass

            # TCS34725 -- command bit 0x80, ID 0x44 / 0x4D / 0x10
            try:
                ident = _r8(sm, I2C_TARGET, 0x80 | 0x12)
                results.append(("TCS34725 id 0x92", f"0x{ident:02x}"))
                if ident in (0x44, 0x4D, 0x10):
                    Out.ok(f"TCS34725 colour sensor (id 0x{ident:02x})")
                    Out.info("install: pip install adafruit-circuitpython-tcs34725 --break-system-packages")
                    return "TCS34725"
            except OSError:
                pass

    except (OSError, PermissionError) as exc:
        Out.fail(f"could not open bus {bus} for identification: {exc}")
        return None

    Out.warn("device answers, but no known chip signature matched")
    if results:
        Out.info("raw register reads, for the record:")
        for label, value in results:
            Out.info(f"    {label} = {value}")
    else:
        Out.info("every identification register read failed -- the device ACKs its")
        Out.info("address but will not serve registers. Often a chip still held in")
        Out.info("reset, or a clone with a non-standard register map.")
    return None


# ---------------------------------------------------------------- audio


def check_audio():
    Out.section("Audio")

    cards_path = "/proc/asound/cards"
    have_cards = False
    if os.path.exists(cards_path):
        try:
            with open(cards_path) as fh:
                text = fh.read()
            have_cards = "no soundcards" not in text.lower() and text.strip() != ""
        except OSError:
            pass

    for label, cmd in (("playback", ["aplay", "-l"]), ("capture", ["arecord", "-l"])):
        rc, out = run(cmd)
        if rc == 127:
            Out.warn(f"{label}: alsa-utils not installed (sudo apt install alsa-utils)")
            continue

        names = [ln.strip() for ln in out.splitlines() if ln.startswith("card ")]
        # The Pi 5's two HDMI outputs are always present and are not our I2S bus.
        ours = [n for n in names if "vc4hdmi" not in n]

        if not names or "no soundcards" in out.lower():
            Out.expected(f"{label}: no card -- correct for now, the overlay is not written yet")
            continue

        for name in names:
            Out.info(f"    {name}")
        if ours:
            Out.ok(f"{label}: I2S card present -- {len(ours)} of {len(names)}")
        else:
            Out.expected(
                f"{label}: {len(names)} card(s), all HDMI -- no I2S card yet, overlay not written"
            )

    if not have_cards:
        Out.expected("/proc/asound reports no soundcards -- expected at this stage")


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description="Gia hardware sanity check")
    ap.add_argument("--bus", type=int, default=1, help="I2C bus number (default: 1)")
    args = ap.parse_args()

    print("Gia hardware check")
    print("==================")

    check_platform()

    try:
        import smbus2 as smbus
    except ImportError:
        Out.section(f"I2C bus {args.bus}")
        Out.fail("smbus2 not installed")
        Out.info("install it with: pip install smbus2 --break-system-packages")
        check_audio()
        print()
        return 1

    if check_bus_node(args.bus):
        found = scan_bus(smbus, args.bus)
        report_scan(found)
        if found and I2C_TARGET in found:
            identify_chip(smbus, args.bus)

    check_audio()

    print()
    print("Next: post this output and we will write the device tree overlay.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
