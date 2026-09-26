#!/usr/bin/env python3
"""Distance sensor (VL53L0X).

    from vl53l0x import Ranger

    ranger = Ranger()
    mm = ranger.distance()          # millimetres, or None if nothing is there
    ranger.wait_until_closer(600)   # blocks until something comes within 600mm

Run this file directly to watch live readings.

Wiring:  SDA -> pin 3,  SCL -> pin 5,  VIN -> 3V3,  GND -> GND
         XSHUT must float or be high. Pulled low, the chip stays in reset and
         the sensor vanishes from the I2C bus entirely.

Needs:   pip install adafruit-circuitpython-vl53l0x --break-system-packages
"""

import time

import board
import busio
import adafruit_vl53l0x

# What the chip reports when nothing is in front of it.
NOTHING_THERE = 8190

# How long the sensor averages each reading, in microseconds. Higher is
# steadier but slower. The chip default is 33000.
TIMING_BUDGET = 200000


class Ranger:
    """A VL53L0X on the I2C bus at address 0x29."""

    def __init__(self):
        i2c = busio.I2C(board.SCL, board.SDA)
        self.sensor = adafruit_vl53l0x.VL53L0X(i2c)
        self.sensor.measurement_timing_budget = TIMING_BUDGET

    def distance(self):
        """Distance in millimetres, or None when nothing is in range.

        Useful range is roughly 30mm to 1200mm.
        """
        mm = self.sensor.range
        if mm >= NOTHING_THERE:
            return None
        return mm

    def wait_until_closer(self, millimetres, check_every=0.1):
        """Block until something is nearer than `millimetres`. Returns how near.

        This is the "someone walked up" trigger.
        """
        while True:
            mm = self.distance()
            if mm is not None and mm < millimetres:
                return mm
            time.sleep(check_every)

    def wait_until_clear(self, millimetres, check_every=0.1):
        """Block until nothing is nearer than `millimetres`.

        The counterpart to wait_until_closer -- use it to avoid retriggering
        while the same person is still standing there.
        """
        while True:
            mm = self.distance()
            if mm is None or mm >= millimetres:
                return
            time.sleep(check_every)


if __name__ == "__main__":
    ranger = Ranger()
    print("Distance sensor. Ctrl-C to stop.")
    try:
        while True:
            mm = ranger.distance()
            if mm is None:
                print("  nothing in range")
            else:
                print(f"  {mm} mm")
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("stopped")
