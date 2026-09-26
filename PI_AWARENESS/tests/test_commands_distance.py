import json
from pathlib import Path
import tempfile
import time
from unittest.mock import Mock, patch
import unittest
from settings import DistanceSettings
from awareness.commands import ALIASES, Command, parse_command, normalize, grammar_phrases, select_candidate
from awareness.distance import Calibration, RangeReading, describe_distance
from awareness.tof import DistanceSensor

class Commands(unittest.TestCase):
    def test_all_aliases(self):
        for language,intents in ALIASES.items():
            for intent,aliases in intents.items():
                for alias in aliases:
                    with self.subTest(language=language,alias=alias):
                        self.assertEqual(parse_command(alias,language),Command(intent,language))
    def test_punctuation(self):
        self.assertEqual(parse_command('  WHAT is THIS?! ').intent,'scan')
    def test_hindi_marks_preserved(self):
        self.assertEqual(normalize('पढ़ो।'),'पढ़ो')
        self.assertEqual(parse_command('दूरी बताओ।').intent,'distance')
    def test_no_substring_trigger(self):
        for text in ('please do not scan','the color is red','read something unknown','scanner','पढ़ो मत'):
            with self.subTest(text=text): self.assertIsNone(parse_command(text))
    def test_wake_prefix(self):
        self.assertIsNone(parse_command('scan','en',True))
        self.assertEqual(parse_command('hello scan','en',True).intent,'scan')
        self.assertEqual(parse_command('सुनो पढ़ो','hi',True).intent,'read')
    def test_calibration_english(self):
        self.assertEqual(parse_command('step length seventy five').value,750)
        self.assertEqual(parse_command('step length 75').value,750)
    def test_calibration_hindi(self):
        self.assertEqual(parse_command('कदम की लंबाई पचहत्तर').value,750)
    def test_grammar_unk_and_no_digits(self):
        for lang in ('en','hi'):
            phrases=grammar_phrases(lang)
            self.assertIn('[unk]',phrases)
            self.assertFalse(any(c.isdigit() for p in phrases for c in p))
    def test_low_confidence_rejected(self):
        self.assertIsNone(select_candidate([('scan','en',.3)],.58,.1))
    def test_ambiguous_bilingual_rejected(self):
        self.assertIsNone(select_candidate([('scan','en',.8),('रंग','hi',.77)],.58,.1))
    def test_matching_bilingual_intents(self):
        self.assertEqual(select_candidate([('read','en',.8),('पढ़ो','hi',.79)],.58,.1).intent,'read')
    def test_clear_bilingual_winner(self):
        self.assertEqual(select_candidate([('read','en',.95),('रंग','hi',.6)],.58,.1).intent,'read')

class StepDistance(unittest.TestCase):
    def test_default_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'calibration.json';c=Calibration(p)
            self.assertEqual(c.step_mm,750);self.assertFalse(c.calibrated)
            c.save(620);loaded=Calibration(p)
            self.assertEqual(loaded.step_mm,620);self.assertTrue(loaded.calibrated)
    def test_invalid_values(self):
        for mm in (0,199,1201,float('nan'),float('inf'),True,None,'bad'):
            with self.subTest(mm=mm), self.assertRaises((ValueError,TypeError)):
                Calibration.validate(mm)
    def test_corrupt_file_falls_back(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'calibration.json';p.write_text('{broken')
            self.assertFalse(Calibration(p).calibrated)
    def test_measured_walk(self):
        with tempfile.TemporaryDirectory() as td:
            c=Calibration(Path(td)/'step.json');c.from_walk(6000,10)
            self.assertEqual(c.step_mm,600)
    def test_walk_bad_counts(self):
        with tempfile.TemporaryDirectory() as td:
            c=Calibration(Path(td)/'step.json')
            for steps in (0,1,2.5,True):
                with self.subTest(steps=steps),self.assertRaises(ValueError):c.from_walk(3000,steps)
    def test_three_steps(self):
        self.assertIn('3 steps',describe_distance(RangeReading(2250,10,True),750,now=10.1))
    def test_conservative_half_steps(self):
        self.assertIn('2.5 steps',describe_distance(RangeReading(2200,10,True),750,now=10.1))
    def test_close_warning(self):
        self.assertIn('less than one step',describe_distance(RangeReading(250,10,True),750,now=10.1))
    def test_invalid_never_clear(self):
        readings=(RangeReading(None,10,False),RangeReading(1000,8,True),RangeReading(0,10,True),
                  RangeReading(float('nan'),10,True),RangeReading(1000,12,True),
                  RangeReading(5000,10,True),RangeReading(1000,float('nan'),True))
        for r in readings:
            with self.subTest(reading=r):
                self.assertIn('unavailable',describe_distance(r,750,now=10.1))
    def test_maximum_cannot_claim_six_steps(self):
        result=describe_distance(RangeReading(4000,10,True),750,now=10.1)
        self.assertIn('5 steps',result);self.assertNotIn('6 steps',result)
    def test_on_demand_sensor_does_not_reuse_old_cache(self):
        sensor=DistanceSensor(DistanceSettings())
        sensor._store(RangeReading(900,time.monotonic()-2,True))
        fake=Mock();fake.sample.return_value=RangeReading(600,time.monotonic(),True,'ok')
        with patch('awareness.tof.VL53L0X',return_value=fake):
            result=sensor.get()
        self.assertTrue(result.valid);self.assertEqual(result.mm,600);fake.sample.assert_called_once();fake.close.assert_called_once()
    def test_disabled_sensor(self):
        sensor=DistanceSensor(DistanceSettings(enabled=False));self.assertFalse(sensor.get().valid)
