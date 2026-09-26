import hashlib
import io
import json
from pathlib import Path
import queue
import stat
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch,Mock
import zipfile
import numpy as np
from settings import CameraSettings,DistanceSettings,VoiceSettings
from awareness.camera import Camera,ImageCamera,capture_once
from awareness.common import FeatureError
from awareness.tof import VL53L0X,DistanceSensor
from awareness.speech import language_runs,text_chunks,resolve_voices,resolve_output_device,Speaker
from awareness.voice import choose_input,vocabulary_grammar,CommandListener
from hardware.inmp441 import convert_i2s_block
from tools.download_models import safe_extract,is_valid,sha256
from tools.model_manifest import Asset,ASSETS

class CameraTests(unittest.TestCase):
    def test_fresh_copy(self):
        c=Camera(CameraSettings());c.latest=np.ones((10,10,3),np.uint8);c.at=time.monotonic()
        image=c.snapshot(.01);image[0]=0;self.assertTrue(c.latest[0].all())
    def test_stale_rejected(self):
        c=Camera(CameraSettings());c.latest=np.ones((10,10,3),np.uint8);c.at=time.monotonic()-5
        with self.assertRaises(FeatureError):c.snapshot(.02)
    def test_capture_once_closes_camera(self):
        cfg=CameraSettings(warmup_s=0)
        fake=Mock();fake.read.return_value=np.ones((10,12,3),np.uint8)
        with patch('awareness.camera.PicameraBackend',return_value=fake):
            image=capture_once(cfg)
        self.assertEqual(image.shape,(10,12,3));fake.close.assert_called_once()
    def test_missing_image_never_falls_back(self):
        with self.assertRaises(FeatureError):ImageCamera('/nonexistent/noimage.png')

class ToFTests(unittest.TestCase):
    def sensor(self,mm=600):
        s=VL53L0X.__new__(VL53L0X);s.cfg=DistanceSettings();s.ranger=Mock();s.ranger.distance.return_value=mm
        return s
    def test_valid_mm_preserved(self):
        r=self.sensor(600).sample();self.assertTrue(r.valid);self.assertEqual(r.mm,600)
    def test_no_return_invalid(self):
        self.assertFalse(self.sensor(None).sample().valid)
    def test_out_of_range_invalid(self):
        for mm in (0,20,1300,8190):
            with self.subTest(mm=mm):self.assertFalse(self.sensor(mm).sample().valid)
    def test_on_demand_sensor_uses_nearest_requested_sample(self):
        cfg=DistanceSettings(samples_per_query=2)
        fake=Mock();fake.sample.side_effect=[
            __import__('awareness.distance',fromlist=['RangeReading']).RangeReading(700,time.monotonic(),True,'ok'),
            __import__('awareness.distance',fromlist=['RangeReading']).RangeReading(500,time.monotonic(),True,'ok')]
        with patch('awareness.tof.VL53L0X',return_value=fake):
            r=DistanceSensor(cfg).get()
        self.assertTrue(r.valid);self.assertEqual(r.mm,500);fake.close.assert_called_once()
    def test_sensor_failure_is_fail_closed(self):
        with patch('awareness.tof.VL53L0X',side_effect=RuntimeError('i2c fail')):
            r=DistanceSensor(DistanceSettings()).get()
        self.assertFalse(r.valid);self.assertIsNone(r.mm)

class AudioTests(unittest.TestCase):
    def test_multilingual_runs(self):
        self.assertEqual(language_runs('लिखा है take two tablets रोज़','hi'),[('लिखा है','hi'),('take two tablets','en'),('रोज़','hi')])
    def test_chunks_preserve_words(self):
        text='one two three four five six';parts=text_chunks(text,10)
        self.assertEqual(' '.join(parts),text);self.assertTrue(all(len(p)<=10 for p in parts))
    def test_tts_english_fallback(self):
        cfg=VoiceSettings(english_voice='en-in');voices='Pty Language Age/Gender VoiceName\n 5 en M English\n 5 hi M Hindi\n'
        with patch('awareness.speech.subprocess.check_output',return_value=voices):
            self.assertEqual(resolve_voices(cfg),{'en':'en','hi':'hi'})
    def test_missing_hindi_fails(self):
        with patch('awareness.speech.subprocess.check_output',return_value='header\n 5 en M English\n'):
            with self.assertRaises(FeatureError):resolve_voices(VoiceSettings())
    def test_native_rate_fallback(self):
        module=SimpleNamespace(query_devices=lambda *_:{'max_input_channels':1,'default_samplerate':48000,'name':'test'},
                               check_input_settings=Mock(side_effect=[ValueError('unsupported'),None]))
        with patch.dict(sys.modules,{'sounddevice':module}):self.assertEqual(choose_input(VoiceSettings(backend='sounddevice',channels=1))[0],48000)
    def test_oov_grammar_filtered(self):
        model=SimpleNamespace(vosk_model_find_word=lambda word:0 if word in ('read','text','hello') else -1)
        phrases=vocabulary_grammar(model,'en',False);self.assertIn('read text',phrases);self.assertNotIn('scan',phrases)
    def test_callback_muted_during_playback(self):
        listener=CommandListener.__new__(CommandListener);listener.cfg=VoiceSettings();listener.speaker=SimpleNamespace(input_blocked=lambda:True)
        listener.discontinuity=threading.Event();listener.queue=queue.Queue()
        listener._capture(bytes(1600),800,None,None)
        self.assertTrue(listener.queue.empty());self.assertTrue(listener.discontinuity.is_set())
    def test_stereo_channel_selection(self):
        listener=CommandListener.__new__(CommandListener);listener.cfg=VoiceSettings(channels=2,channel_index=1)
        listener.speaker=SimpleNamespace(input_blocked=lambda:False);listener.discontinuity=threading.Event();listener.queue=queue.Queue()
        listener._capture(np.array([[1,2],[3,4]],np.int16).tobytes(),2,None,None)
        np.testing.assert_array_equal(np.frombuffer(listener.queue.get()[1],np.int16),[2,4])

    def test_inmp441_s32_stereo_to_16k_mono(self):
        # 30 ms of a left-channel waveform; right channel deliberately different.
        frames=1440;t=np.arange(frames);left=(np.sin(t/10)*2**27).astype('<i4');right=np.full(frames,2**29,dtype='<i4')
        raw=np.column_stack([left,right]).astype('<i4').tobytes()
        pcm=convert_i2s_block(raw,gain=4.0)
        out=np.frombuffer(pcm,dtype='<i2');self.assertEqual(len(out),480);self.assertGreater(np.max(np.abs(out)),0)
    def test_i2s_choose_input_is_16k_after_conversion(self):
        rate,info=choose_input(VoiceSettings());self.assertEqual(rate,16000);self.assertIn('INMP441',info['name'])
    def test_i2s_output_auto_resolution(self):
        with patch('hardware.i2s_common.find_i2s_device',return_value='plughw:2,0'):
            self.assertEqual(resolve_output_device('i2s-auto'),'plughw:2,0')

class DeploymentProfileTests(unittest.TestCase):
    def test_install_configures_exact_pi_hardware_overlay(self):
        text=(Path(__file__).resolve().parents[1]/'install.sh').read_text()
        for setting in ('dtparam=i2c_arm=on','dtparam=i2s=on','dtoverlay=googlevoicehat-soundcard','camera_auto_detect=1'):
            self.assertIn(setting,text)
    def test_service_is_enabled_and_network_isolated(self):
        text=(Path(__file__).resolve().parents[1]/'tools/install_service.sh').read_text()
        self.assertIn('systemctl enable pi-awareness.service',text)
        self.assertIn('PrivateNetwork=true',text)
        self.assertIn('Restart=on-failure',text)

class DownloadSecurity(unittest.TestCase):
    def makezip(self,path,name='model/am/final.mdl',payload=b'data',symlink=False):
        with zipfile.ZipFile(path,'w') as z:
            if symlink:
                item=zipfile.ZipInfo(name);item.create_system=3;item.external_attr=(stat.S_IFLNK|0o777)<<16;z.writestr(item,'/etc/passwd')
            else:z.writestr(name,payload)
    def test_safe_archive(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);self.makezip(p/'a.zip');safe_extract(p/'a.zip',p/'out','model')
            self.assertEqual((p/'out/model/am/final.mdl').read_bytes(),b'data')
    def test_traversal_and_root_rejected(self):
        for name in ('model/../../escape','/model/file','other/file','model\\bad','model/C:bad'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as td:
                p=Path(td);self.makezip(p/'a.zip',name)
                with self.assertRaises(ValueError):safe_extract(p/'a.zip',p/'out','model')
    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);self.makezip(p/'a.zip','model/link',symlink=True)
            with self.assertRaises(ValueError):safe_extract(p/'a.zip',p/'out','model')
    def test_archive_size_cap(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);self.makezip(p/'a.zip',payload=b'0123456789')
            with self.assertRaises(ValueError):safe_extract(p/'a.zip',p/'out','model',maximum_bytes=5)
    def test_local_checksum_verification(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'x.onnx').write_bytes(b'hello')
            asset=Asset('test','https://example.invalid/x','x.onnx',sha256=hashlib.sha256(b'hello').hexdigest())
            self.assertTrue(is_valid(asset,p,{}));(p/'x.onnx').write_bytes(b'changed');self.assertFalse(is_valid(asset,p,{}))
    def test_manifest_pins_and_versions(self):
        self.assertEqual(len(ASSETS),6)
        for asset in ASSETS:
            self.assertTrue(asset.url.startswith('https://'))
            if asset.target.startswith('ocr_'):self.assertEqual(len(asset.sha256),64);self.assertIn('/v3.9.2/',asset.url)
    def test_offline_guard_blocks_internet(self):
        root=Path(__file__).resolve().parents[1]
        script="""from awareness.common import install_offline_guard
import socket
install_offline_guard()
try:
    socket.socket().connect(("127.0.0.1",1))
except PermissionError:
    print("blocked")
else:
    raise RuntimeError("not blocked")
"""
        result=subprocess.run([sys.executable,'-c',script],cwd=root,capture_output=True,text=True,timeout=5)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('blocked',result.stdout)
