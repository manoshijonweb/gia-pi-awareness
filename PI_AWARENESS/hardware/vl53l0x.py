#!/usr/bin/env python3
"""VL53L0X distance sensor adapter based on the supplied working helper.

Wiring:
  SDA -> physical pin 3
  SCL -> physical pin 5
  VIN -> 3V3
  GND -> GND
  XSHUT -> floating/high
"""
from __future__ import annotations
import time

NOTHING_THERE = 8190


class Ranger:
    def __init__(self, timing_budget_us: int = 200000):
        import board
        import busio
        import adafruit_vl53l0x
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.sensor = adafruit_vl53l0x.VL53L0X(self.i2c)
        self.sensor.measurement_timing_budget = int(timing_budget_us)

    def distance(self):
        """Millimetres, or None when the sensor reports no usable return."""
        mm = int(self.sensor.range)
        if mm >= NOTHING_THERE:
            return None
        return mm

    def wait_until_closer(self, millimetres, check_every=0.1):
        while True:
            mm = self.distance()
            if mm is not None and mm < millimetres:
                return mm
            time.sleep(check_every)

    def wait_until_clear(self, millimetres, check_every=0.1):
        while True:
            mm = self.distance()
            if mm is None or mm >= millimetres:
                return
            time.sleep(check_every)

    def close(self):
        try:
            deinit = getattr(self.i2c, "deinit", None)
            if deinit:
                deinit()
        finally:
            self.sensor = None


if __name__ == "__main__":
    ranger = Ranger()
    try:
        while True:
            print(ranger.distance())
            time.sleep(0.2)
    finally:
        ranger.close()
