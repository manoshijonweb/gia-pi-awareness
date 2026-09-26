# Changelog

## 1.2.0 — 22 September 2026

- Rebuilt around the assembled Pi 5 hardware: INMP441, MAX98357A, VL53L0X and Raspberry Pi Camera Rev 1.3.
- Added persistent INMP441 S32_LE/48 kHz I2S capture with DC removal, 48→16 kHz conversion and configurable gain.
- Added automatic shared-I2S ALSA card discovery for both capture and MAX98357A playback.
- Replaced the previous VL53L1X background poller with on-demand VL53L0X measurements.
- Changed CSI camera operation to on-demand capture rather than continuous streaming while idle.
- Added service-first install: `install.sh` configures buses/overlay and enables `pi-awareness.service` automatically.
- Boot/service announcement is exactly `I am ready.` before microphone capture begins.
- `scan` now announces per-class object counts, e.g. `2 people and 1 chair`.
- YOLO26m is optionally warmed once at boot for scan latency; OCR remains lazy by default.
- Added integrated hardware check and INMP441-specific microphone tuning tool.
- Updated tests/docs for the actual hardware profile.

## 1.1.0 — 16 September 2026

- YOLO26m 640×640 became the default detector with YOLO26s/n alternatives.

## 1.0.0

- Initial offline bilingual voice/color/OCR/object/distance prototype.
