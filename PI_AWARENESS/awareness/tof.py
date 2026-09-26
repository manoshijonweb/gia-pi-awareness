"""On-demand VL53L0X adapter for the assembled hardware.

Unlike the earlier VL53L1X build, this module does not run a background polling
thread.  A distance command opens the supplied VL53L0X helper, takes the
configured reading(s), and releases the bus.  That keeps idle CPU/I2C activity at
zero while preserving fail-closed range semantics.
"""
from __future__ import annotations
import logging
import statistics
import threading
import time
from .distance import RangeReading


class VL53L0X:
    def __init__(self, cfg):
        if getattr(cfg, "driver", "vl53l0x") != "vl53l0x":
            raise RuntimeError(f"Unsupported distance driver: {cfg.driver}")
        from hardware.vl53l0x import Ranger
        self.cfg = cfg
        self.ranger = Ranger(timing_budget_us=cfg.timing_budget_us)

    def sample(self) -> RangeReading:
        at = time.monotonic()
        mm = self.ranger.distance()
        if mm is None:
            return RangeReading(None, at, False, "VL53L0X supplied no usable return")
        valid = self.cfg.min_distance_mm <= mm <= self.cfg.max_distance_mm
        return RangeReading(float(mm) if valid else None, at, valid,
                            "ok" if valid else f"out of configured range: {mm} mm")

    def close(self):
        self.ranger.close()


class DistanceSensor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.latest = RangeReading(None, 0.0, False, "not measured")

    def start(self):
        # Deliberately no worker thread.  Idle mode performs no ToF polling.
        return self

    def _store(self, value):
        with self.lock:
            self.latest = value

    def get(self) -> RangeReading:
        if not self.cfg.enabled:
            return RangeReading(None, time.monotonic(), False, "disabled in settings.py")
        driver = None
        try:
            driver = VL53L0X(self.cfg)
            readings = []
            for _ in range(max(1, int(self.cfg.samples_per_query))):
                reading = driver.sample()
                if reading.valid:
                    readings.append(reading)
            if not readings:
                result = RangeReading(None, time.monotonic(), False, "no valid VL53L0X measurement")
            else:
                # If multiple samples are requested, never make an approaching obstacle
                # look farther away than the nearest valid sample.
                mm = min(r.mm for r in readings if r.mm is not None)
                result = RangeReading(mm, readings[-1].at, True, "ok")
            self._store(result)
            return result
        except Exception as exc:
            logging.getLogger(__name__).warning("Distance sensor unavailable: %s", exc)
            result = RangeReading(None, time.monotonic(), False, str(exc))
            self._store(result)
            return result
        finally:
            if driver:
                try:
                    driver.close()
                except Exception:
                    logging.getLogger(__name__).exception("VL53L0X shutdown failed")

    def close(self):
        pass
