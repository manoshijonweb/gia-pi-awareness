"""Synthetic contract/regression tests. These do NOT establish model accuracy or Pi speed."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch, Mock
import numpy as np
from settings import Settings, VisionSettings
from awareness.common import FeatureError
from awareness.objects import (ObjectDetector, decode_yolo26_e2e, decode_yolo26_community,
                               detector_format, describe_objects, letterbox)
from awareness.detector_profiles import PROFILES, apply_profile
from tools.model_manifest import ASSETS, DETECTOR_ASSETS, assets_for
from tools.download_models import provision, is_valid
from tools.benchmark import summary


def outputs(classes=(39,), scores=(.9,), centers=None):
    logits=np.full((1,len(classes),80),-80,np.float32)
    for i,(cls,score) in enumerate(zip(classes,scores)):
        logits[0,i,cls]=np.log(score/(1-score))
    boxes=np.array([centers or [[.5,.5,.25,.5]]*len(classes)],np.float32)
    return logits,boxes


class FakeSession:
    def __init__(self, fmt='community_yolos', result=None, size=640):
        self.fmt=fmt;self.result=result if result is not None else outputs();self.feed=None
        self.input=SimpleNamespace(shape=[1,3,size,size],name='pixel_values',type='tensor(float)')
    def get_inputs(self):return [self.input]
    def get_outputs(self):
        if self.fmt=='community_yolos':
            # Deliberately reverse metadata order: inference must request names, not positions.
            return [SimpleNamespace(name='pred_boxes',shape=[1,300,4]),SimpleNamespace(name='logits',shape=[1,300,80])]
        return [SimpleNamespace(name='output0',shape=[1,300,6])]
    def run(self,names,feed):
        self.feed=feed;self.names=names
        return self.result


class YOLO26DecodeTests(unittest.TestCase):
    def test_community_sigmoid_not_softmax(self):
        logits,box=outputs();b,s,c=decode_yolo26_community(logits,box,480,800)
        np.testing.assert_allclose(b[0],[300,120,500,360]);self.assertAlmostEqual(float(s[0]),.9,places=5)
        self.assertEqual(c.tolist(),[39])
    def test_community_multi_label_scores_do_not_softmax(self):
        logits,box=outputs();logits[0,0,0]=logits[0,0,39]-.01
        _,s,_=decode_yolo26_community(logits,box,640,640)
        self.assertAlmostEqual(float(s[0]),.9,places=5)
    def test_community_empty_candidates(self):
        b,s,c=decode_yolo26_community(np.zeros((1,0,80)),np.zeros((1,0,4)),640,640)
        self.assertEqual(b.shape,(0,4));self.assertEqual(len(s),0)
    def test_community_infinite_logits_are_sigmoid_limits(self):
        logits,box=outputs();logits[:]=-np.inf;logits[0,0,39]=np.inf
        _,scores,ids=decode_yolo26_community(logits,box,640,640)
        self.assertEqual(float(scores[0]),1);self.assertEqual(int(ids[0]),39)
    def test_community_nan_logit_rejects_row(self):
        logits,box=outputs();logits[0,0,4]=np.nan
        self.assertEqual(len(decode_yolo26_community(logits,box,640,640)[1]),0)
    def test_community_infinite_box_rejected(self):
        logits,box=outputs();box[0,0,0]=np.inf
        self.assertEqual(len(decode_yolo26_community(logits,box,640,640)[1]),0)
    def test_community_shapes_rejected(self):
        for logits,boxes in ((np.zeros((1,300,81)),np.zeros((1,300,4))),
                             (np.zeros((2,300,80)),np.zeros((2,300,4))),
                             (np.zeros((1,10,80)),np.zeros((1,9,4))),
                             (np.zeros((1,300,80)),np.zeros((1,300,5)))):
            with self.subTest(shape=logits.shape),self.assertRaises(FeatureError):
                decode_yolo26_community(logits,boxes,640,640)
    def test_e2e_xyxy_not_xywh(self):
        p=np.array([[[10,20,30,50,.9,39]]],np.float32)
        boxes,scores,ids=decode_yolo26_e2e(p)
        np.testing.assert_array_equal(boxes[0],[10,20,30,50]);self.assertEqual(ids[0],39)
    def test_e2e_invalid_classes_filtered_before_cast(self):
        p=np.array([[[1,2,3,4,.9,c] for c in (-1,80,1.5,np.nan,np.inf,79)]],np.float32)
        _,_,ids=decode_yolo26_e2e(p);self.assertEqual(ids.tolist(),[79])
    def test_e2e_invalid_scores(self):
        p=np.array([[[1,2,3,4,s,39] for s in (-.1,1.1,np.nan,np.inf,.9)]],np.float32)
        self.assertEqual(len(decode_yolo26_e2e(p)[1]),1)
    def test_e2e_no_input_mutation(self):
        p=np.array([[[10,20,30,50,.9,39]]],np.float32);before=p.copy()
        b,_,_=decode_yolo26_e2e(p);b[:]=0;np.testing.assert_array_equal(p,before)
    def test_e2e_empty(self):
        self.assertEqual(len(decode_yolo26_e2e(np.zeros((1,0,6)))[1]),0)
    def test_e2e_wrong_contract_rejected(self):
        for shape in ((1,84,8400),(1,300,7),(2,300,6),(300,6)):
            with self.subTest(shape=shape),self.assertRaises(FeatureError):
                decode_yolo26_e2e(np.zeros(shape))


class YOLO26AdapterTests(unittest.TestCase):
    def detector(self,cfg=None,fake=None):
        cfg=cfg or VisionSettings();fake=fake or FakeSession()
        with patch('awareness.objects.session',return_value=fake) as constructor:
            detector=ObjectDetector(cfg)
        return detector,fake,constructor
    def test_default_is_m640(self):
        c=VisionSettings();self.assertEqual(c.detector,'yolo26m');self.assertEqual(c.object_input_size,640)
        self.assertEqual(c.object_model.name,'yolo26m_640.onnx')
    def test_community_preprocess_rgb_unit_range_and_no_padding(self):
        d,fake,_=self.detector();d.detect(np.full((480,800,3),(255,0,0),np.uint8))
        tensor=fake.feed['pixel_values'];self.assertEqual(tensor.shape,(1,3,640,640))
        self.assertTrue(tensor.flags.c_contiguous);self.assertEqual(tensor.dtype,np.float32)
        np.testing.assert_array_equal(tensor[0,:,0,0],[0,0,1]);np.testing.assert_array_equal(tensor[0,:,-1,-1],[0,0,1])
    def test_named_community_outputs(self):
        d,fake,_=self.detector();d.detect(np.zeros((480,800,3),np.uint8))
        self.assertEqual(fake.names,['logits','pred_boxes'])
    def test_community_coordinates_and_hindi(self):
        d,_,_=self.detector();result=d.detect(np.zeros((480,800,3),np.uint8))
        np.testing.assert_allclose(result[0].box,[300,120,500,360]);self.assertIn('बोतल',describe_objects(result,'hi'))
    def test_community_does_not_run_external_nms(self):
        d,_,_=self.detector(fake=FakeSession(result=outputs((39,39),(.9,.8))))
        with patch('awareness.objects.nms',side_effect=AssertionError('NMS should not run')):
            self.assertEqual(len(d.detect(np.zeros((640,640,3),np.uint8))),2)
    def test_community_empty_result(self):
        d,_,_=self.detector(fake=FakeSession(result=outputs(scores=(.1,))))
        self.assertEqual(d.detect(np.zeros((640,640,3),np.uint8)),[])
    def test_community_clips_boxes_to_source(self):
        d,_,_=self.detector(fake=FakeSession(result=outputs(centers=[[.5,.5,2,2]])))
        self.assertEqual(d.detect(np.zeros((480,800,3),np.uint8))[0].box,(0,0,800,480))
    def test_invalid_geometry_discarded(self):
        d,_,_=self.detector(fake=FakeSession(result=outputs(centers=[[.5,.5,-.2,.4]])))
        self.assertEqual(d.detect(np.zeros((640,640,3),np.uint8)),[])
    def test_dedicated_threads(self):
        _,_,construct=self.detector(VisionSettings(object_threads=3,ort_threads=2))
        self.assertEqual(construct.call_args.args[1],3)
    def test_graph_size_mismatch_fails(self):
        with self.assertRaises(FeatureError):self.detector(fake=FakeSession(size=320))
    def test_graph_channel_mismatch_fails(self):
        fake=FakeSession();fake.input.shape=[1,640,640,3]
        with self.assertRaises(FeatureError):self.detector(fake=fake)
    def test_graph_batch_mismatch_fails(self):
        fake=FakeSession();fake.input.shape[0]=2
        with self.assertRaises(FeatureError):self.detector(fake=fake)
    def test_native_graph_as_community_fails(self):
        with self.assertRaises(FeatureError):self.detector(fake=FakeSession(fmt='ultralytics_e2e'))
    def test_community_graph_as_native_fails(self):
        with self.assertRaises(FeatureError):self.detector(VisionSettings(object_format='ultralytics_e2e'))
    def test_e2e_native_centered_letterbox_and_remap(self):
        fake=FakeSession(fmt='ultralytics_e2e')
        d,_,_=self.detector(VisionSettings(object_format='ultralytics_e2e'),fake)
        raw=np.array([[[240,224,400,416,.9,39]]],np.float32)
        with patch('awareness.objects.infer',return_value=raw) as run,patch('awareness.objects.nms',side_effect=AssertionError()):
            result=d.detect(np.full((480,800,3),(255,0,0),np.uint8))
            tensor=run.call_args.args[1]
        np.testing.assert_allclose(result[0].box,[300,120,500,360],atol=1e-4)
        np.testing.assert_allclose(tensor[0,:,0,0],np.full(3,114/255),atol=1e-6)
    def test_native_raw_runs_nms(self):
        d,_,_=self.detector(VisionSettings(object_format='ultralytics_raw'),FakeSession(fmt='ultralytics_raw'))
        raw=np.zeros((1,84,2),np.float32);raw[0,:4,:]=np.array([[320],[320],[80],[80]])
        raw[0,43,:]=[.9,.8]
        with patch('awareness.objects.infer',return_value=raw):
            self.assertEqual(len(d.detect(np.zeros((640,640,3),np.uint8))),1)
    def test_timings_reported(self):
        d,_,_=self.detector();d.warmup()
        self.assertEqual(set(d.last_timings),{'preprocess_s','inference_s','postprocess_s','total_s'})
        self.assertTrue(all(x>=0 for x in d.last_timings.values()))
    def test_bad_image_rejected(self):
        d,_,_=self.detector()
        for image in (None,np.zeros((0,10,3),np.uint8),np.zeros((10,10),np.uint8),np.zeros((10,10,3),np.float32)):
            with self.assertRaises(FeatureError):d.detect(image)
    def test_unknown_detector_or_format_rejected(self):
        for cfg in (VisionSettings(detector='bad'),VisionSettings(object_format='bad')):
            with self.assertRaises(FeatureError):detector_format(cfg)
    def test_letterbox_rounding_matches_native_export(self):
        _,_,left,top=letterbox(np.zeros((333,1000,3),np.uint8),640,640,True)
        self.assertEqual((left,top),(0,213))


class YOLO26ProvisionTests(unittest.TestCase):
    def test_all_three_yolo_assets_pinned(self):
        for name in ('yolo26m','yolo26s','yolo26n'):
            asset=DETECTOR_ASSETS[name]
            self.assertEqual(len(asset.sha256),64);self.assertNotIn('/main/',asset.url)
            self.assertEqual(asset.target,PROFILES[name].filename)
    def test_default_contains_m_not_legacy(self):
        self.assertEqual(len(ASSETS),6);self.assertEqual(ASSETS[0],DETECTOR_ASSETS['yolo26m'])
        self.assertFalse(any('yolox' in a.target for a in ASSETS))
    def test_only_detector_and_shared_selection(self):
        self.assertEqual(len(assets_for('yolo26s',only_detector=True)),1)
        self.assertEqual(len(assets_for('yolo26m',skip_detector=True)),5)
    def test_conflicting_selection_rejected(self):
        with self.assertRaises(ValueError):assets_for('yolo26m',True,True)
    def test_unknown_download_not_silent_fallback(self):
        with self.assertRaises(ValueError):assets_for('yolov8n')
    def test_profile_keeps_other_settings(self):
        cfg=Settings();cfg.camera.rotation=90;cfg.voice.microphone='test';cfg.vision.ocr_det=Path('/test/ocr')
        apply_profile(cfg.vision,'yolo26s')
        self.assertEqual(cfg.vision.detector,'yolo26s');self.assertEqual(cfg.vision.object_model.name,'yolo26s_640.onnx')
        self.assertEqual(cfg.camera.rotation,90);self.assertEqual(cfg.voice.microphone,'test');self.assertEqual(cfg.vision.ocr_det,Path('/test/ocr'))
    def test_profile_never_changes_thread_policy(self):
        cfg=VisionSettings(object_threads=2);apply_profile(cfg,'yolo26n');self.assertEqual(cfg.object_threads,2)
    def test_bad_profile_rejected(self):
        with self.assertRaises(ValueError):apply_profile(VisionSettings(),'unknown')
    def test_missing_verify_never_uses_network(self):
        with tempfile.TemporaryDirectory() as td,patch('tools.download_models.download',side_effect=AssertionError('network used')), patch('builtins.print'):
            self.assertEqual(provision(Path(td),verify=True,assets=assets_for('yolo26m',True)),1)
    def test_benchmark_summary(self):
        s=summary([.1,.2,.3]);self.assertAlmostEqual(s['median_s'],.2);self.assertAlmostEqual(s['p95_s'],.29)


class ExportUpgradeTests(unittest.TestCase):
    def test_export_shape_validation(self):
        from tools.export_yolo26 import validate_contract
        validate_contract([1,3,640,640],[np.zeros((1,300,6))],640)
        with self.assertRaises(ValueError):validate_contract([1,3,640,640],[np.zeros((1,84,8400))],640)
    def test_export_static_input_validation(self):
        from tools.export_yolo26 import validate_contract
        with self.assertRaises(ValueError):validate_contract([1,3,'height','width'],[np.zeros((1,300,6))],640)
    def test_upgrade_preserves_unrelated_fields(self):
        from tools.upgrade_existing import updated_settings
        old='''from dataclasses import dataclass
from pathlib import Path
ROOT=Path('/example')
@dataclass
class VisionSettings:
    detector: str = "yolox_nano"
    object_model: Path = ROOT / "models/yolox_nano.onnx"
    object_input_size: int = 416
    ocr_budget_s: float = 7.5
    ort_threads: int = 2
MICROPHONE="custom I2S device"
STEP_LENGTH=650
'''
        new=updated_settings(old);ns={};exec(new,ns)
        cfg=ns['VisionSettings']()
        self.assertEqual(cfg.detector,'yolo26m');self.assertEqual(cfg.object_threads,3)
        self.assertEqual(cfg.ocr_budget_s,7.5);self.assertEqual(cfg.ort_threads,2)
        self.assertEqual(ns['MICROPHONE'],'custom I2S device');self.assertEqual(ns['STEP_LENGTH'],650)
    def test_upgrade_idempotent(self):
        from tools.upgrade_existing import updated_settings
        from settings import ROOT
        text=(ROOT/'settings.py').read_text();once=updated_settings(text)
        self.assertEqual(updated_settings(once),once)
    def test_upgrade_unknown_schema_refused(self):
        from tools.upgrade_existing import updated_settings
        with self.assertRaises(ValueError):updated_settings('detector="example"\n')

if __name__=='__main__':unittest.main()
