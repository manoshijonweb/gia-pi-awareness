from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock,patch
import numpy as np
from settings import Settings
from awareness.app import Application
from awareness.commands import Command
from awareness.distance import RangeReading
from awareness.objects import Detection
from awareness.ocr import OCRResult,TextLine

class ApplicationTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();cfg=Settings();cfg.preload_models=False;cfg.distance.enabled=False
        cfg.distance.calibration_file=Path(self.directory.name)/'calibration.json'
        self.speaker=Mock()
        with patch('awareness.app.Speaker',return_value=self.speaker):self.app=Application(cfg,no_speech=True)
        self.app.camera=Mock(snapshot=lambda:np.full((100,100,3),(255,0,0),np.uint8))
    def tearDown(self):self.app.close();self.directory.cleanup()
    def test_color_end_to_end_actual_opencv(self):
        response=self.app.process(Command('color','en'));self.assertIn('blue',response);self.speaker.say.assert_called()
    def test_hindi_auto_locale(self):self.assertIn('नीला',self.app.process(Command('color','hi')))
    def test_stub_scan_response(self):
        self.app.detector=Mock(detect=lambda _: [Detection(39,.95,(0,0,50,50))])
        self.assertIn('bottle',self.app.process(Command('scan','en')))
    def test_stub_ocr_response(self):
        self.app.reader=Mock(read=lambda *_:OCRResult([TextLine('hello किताब',.95,())],False,1,0))
        self.assertIn('hello किताब',self.app.process(Command('read','hi')))
    def test_sensor_unavailable_no_clear(self):self.assertIn('unavailable',self.app.process(Command('distance','en')))

    def test_live_camera_is_on_demand(self):
        self.app.camera=None
        frame=np.full((100,100,3),(255,0,0),np.uint8)
        with patch('awareness.app.capture_once',return_value=frame) as capture:
            response=self.app.process(Command('color','en'))
        self.assertIn('blue',response);capture.assert_called_once_with(self.app.cfg.camera)

    def test_voice_start_announces_exact_ready_then_starts_listener(self):
        cfg=Settings();cfg.power.preload_detector=False;cfg.power.preload_ocr=False;cfg.preload_models=False
        cfg.distance.enabled=False;cfg.distance.calibration_file=Path(self.directory.name)/'boot-calibration.json'
        speaker=Mock();listener=Mock();listener.start.return_value=listener
        with patch('awareness.app.Speaker',return_value=speaker), patch('awareness.voice.CommandListener',return_value=listener):
            app=Application(cfg,no_speech=False)
            try:
                app.start(voice=True)
            finally:
                app.close()
        speaker.say.assert_any_call('I am ready','en')
        speaker.wait_idle.assert_called_once_with(timeout=20)
        listener.start.assert_called_once()

    def test_voice_calibration_requires_confirm(self):
        self.app.process(Command('calibrate'))
        self.app.process(Command('set_step','en',650));self.assertEqual(self.app.calibration.step_mm,750)
        self.app.process(Command('confirm'));self.assertEqual(self.app.calibration.step_mm,650)
        self.assertTrue(self.app.calibration.path.exists())
    def test_calibration_requires_start(self):
        self.assertIn('calibrate first',self.app.process(Command('set_step','en',650)))
    def test_calibration_cancel(self):
        self.app.process(Command('calibrate'));self.app.process(Command('set_step','en',650))
        self.app.process(Command('cancel'));self.app.process(Command('confirm'))
        self.assertEqual(self.app.calibration.step_mm,750)
    def test_repeat(self):
        first=self.app.process(Command('color'));self.assertEqual(self.app.process(Command('repeat')),first)
    def test_stop_drains_queued_commands(self):
        self.app.submit(Command('read'));self.app.submit(Command('scan'));self.app.submit(Command('stop'))
        self.assertTrue(self.app.commands.empty());self.assertTrue(self.app.cancel.is_set())
    def test_camera_failure_spoken_not_faked(self):
        self.app.camera_error='camera failed';self.assertIn('unavailable',self.app.process(Command('color')))
    def test_missing_model_spoken_not_faked(self):
        self.app.feature_errors['scan']='missing weights';self.assertIn('unavailable',self.app.process(Command('scan')))
    def test_ocr_cancel_drops_late_result(self):
        def read(*_):
            self.app.cancel.set();return OCRResult([TextLine('late result',.9,())],False,1,0)
        self.app.reader=Mock(read=read);self.assertEqual(self.app.process(Command('read')),'')
        self.speaker.say.assert_not_called()
    def test_language_switch(self):
        self.app.process(Command('language_hi'));self.assertIn('नीला',self.app.process(Command('color','en')))
