from types import SimpleNamespace
from unittest.mock import patch
import threading
import unittest
import cv2
import numpy as np
from settings import VisionSettings
from awareness.common import FeatureError,Cancelled
from awareness.colors import identify_color,describe_color,hsv_name,correct_illuminant,ColorResult
from awareness.objects import COCO,COCO_HI,Detection,ObjectDetector,decode_yolov8,decode_yolox,letterbox,nms,describe_objects
from awareness.ocr import ctc_decode,db_boxes,order_quad,crop_quad,normalize_line,reading_order,TextReader,TextLine,OCRResult,describe_text

class ObjectTests(unittest.TestCase):
    def test_eighty_classes(self):
        self.assertEqual(len(COCO),80);self.assertEqual(len(COCO_HI),80)
        self.assertEqual(COCO[39],'bottle');self.assertEqual(COCO[73],'book')
    def test_letterbox_distinct(self):
        image=np.zeros((100,200,3),np.uint8)
        x,scale,left,top=letterbox(image,320,320,False)
        self.assertEqual((scale,left,top),(1.6,0,0));self.assertEqual(x[319,0,0],114)
        x,_,left,top=letterbox(image,320,320,True)
        self.assertEqual((left,top),(0,80));self.assertEqual(x[0,0,0],114)
    def test_v8_layouts(self):
        prediction=np.zeros((1,84,10),np.float32)
        prediction[0,:4,0]=[100,150,40,60];prediction[0,43,0]=.9
        for p in (prediction,prediction.transpose(0,2,1)):
            boxes,scores,ids=decode_yolov8(p)
            np.testing.assert_allclose(boxes[0],[80,120,120,180]);self.assertEqual(ids[0],39)
            self.assertAlmostEqual(float(scores[0]),.9,places=5)
    def test_v8_wrong_export_rejected(self):
        with self.assertRaises(FeatureError):decode_yolov8(np.zeros((1,10,6)))
    def test_yolox_grid_and_objectness(self):
        raw=np.zeros((1,21,85),np.float32) # 32x32 -> 16+4+1 cells
        raw[0,0,:4]=[2,2,0,0];raw[0,0,4]=.8;raw[0,0,5+39]=.9
        boxes,scores,ids=decode_yolox(raw,32,32)
        np.testing.assert_allclose(boxes[0],[12,12,20,20]);self.assertAlmostEqual(scores[0],.72,places=5)
        self.assertEqual(ids[0],39)
    def test_yolox_grid_mismatch(self):
        with self.assertRaises(FeatureError):decode_yolox(np.zeros((1,20,85)),32,32)
    def test_class_aware_nms(self):
        boxes=np.array([[0,0,20,20]]*3);scores=np.array([.9,.8,.7]);classes=np.array([0,0,1])
        self.assertEqual(nms(boxes,scores,classes,.45),[0,2])
    def test_empty_nms(self):
        self.assertEqual(nms(np.zeros((0,4)),np.zeros(0),np.zeros(0),.45),[])
    def test_description_not_clear(self):
        self.assertIn('does not mean',describe_objects([],'en'))
    def test_description_counts(self):
        d=[Detection(39,.9,(0,0,1,1)),Detection(39,.8,(0,0,1,1)),Detection(73,.7,(0,0,1,1))]
        self.assertEqual(describe_objects(d,'en'),'I see 2 bottles and 1 book.')
        hindi=describe_objects(d,'hi');self.assertIn('2 बोतल',hindi);self.assertIn('1 किताब',hindi)
    def test_description_person_plural(self):
        d=[Detection(0,.9,(0,0,1,1)),Detection(0,.8,(2,2,3,3)),Detection(56,.7,(0,0,1,1))]
        self.assertEqual(describe_objects(d,'en'),'I see 2 people and 1 chair.')
    def test_v8_detect_mock_onnx_preprocess(self):
        cfg=VisionSettings(detector='yolov8n',object_input_size=320)
        detector=ObjectDetector.__new__(ObjectDetector);detector.cfg=cfg;detector.h=detector.w=320;detector.session=object()
        output=np.zeros((1,84,1),np.float32);output[0,:4,0]=[160,160,80,80];output[0,43,0]=.95
        image=np.full((320,320,3),(255,0,0),np.uint8)
        with patch('awareness.objects.infer',return_value=output) as infer:
            result=detector.detect(image);tensor=infer.call_args.args[1]
        self.assertEqual(result[0].class_id,39)
        np.testing.assert_allclose(tensor[0,:,0,0],[0,0,1])
    def test_yolox_detect_mock_onnx_preprocess(self):
        cfg=VisionSettings(detector="yolox_nano",object_input_size=32)
        detector=ObjectDetector.__new__(ObjectDetector);detector.cfg=cfg;detector.h=detector.w=32;detector.session=object()
        output=np.zeros((1,21,85),np.float32)
        with patch('awareness.objects.infer',return_value=output) as infer:
            detector.detect(np.full((32,32,3),(255,0,0),np.uint8));tensor=infer.call_args.args[1]
        np.testing.assert_allclose(tensor[0,:,0,0],[255,0,0])

class ColorTests(unittest.TestCase):
    def setUp(self):self.cfg=VisionSettings()
    def test_solid_colors(self):
        for bgr,name in (((255,0,0),'blue'),((0,0,255),'red'),((0,255,0),'green'),((255,255,255),'white'),((150,150,150),'gray')):
            with self.subTest(name=name):
                r=identify_color(np.full((100,100,3),bgr,np.uint8),self.cfg)
                self.assertEqual(r.name,name);self.assertGreater(r.dominance,.99)
    def test_darkness_not_false_black(self):
        r=identify_color(np.zeros((100,100,3),np.uint8),self.cfg)
        self.assertIn('Too dark',describe_color(r,'en'))
    def test_central_region(self):
        image=np.full((200,200,3),(0,255,0),np.uint8);image[50:150,50:150]=(255,0,0)
        self.assertEqual(identify_color(image,self.cfg).name,'blue')
    def test_red_wrap(self):
        hsv=np.full((100,100,3),(0,255,255),np.uint8);hsv[:,50:,0]=179
        self.assertEqual(identify_color(cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR),self.cfg).name,'red')
    def test_color_constancy_not_graying_red(self):
        source=np.full((100,100,3),(0,0,255),np.uint8);out,applied=correct_illuminant(source)
        self.assertFalse(applied);np.testing.assert_array_equal(source,out)
    def test_ambiguous_color(self):
        self.assertIn('Several colors',describe_color(ColorResult('red','',.34,True,False),'en'))
    def test_hindi_dark_blue(self):
        self.assertIn('गहरा नीला',describe_color(ColorResult('blue','dark',.95,True,False),'hi'))
    def test_bad_image(self):
        with self.assertRaises(FeatureError):identify_color(None,self.cfg)

class OCRTests(unittest.TestCase):
    def setUp(self):self.cfg=VisionSettings()
    def test_ctc_repeat_blank_unicode(self):
        chars=['blank','क','ि','त','ा','ब',' '];ids=[1,1,0,2,3,4,5,0,5]
        p=np.zeros((1,len(ids),len(chars)),np.float32)
        for index,value in enumerate(ids):p[0,index,value]=.9
        text,confidence=ctc_decode(p,chars)
        self.assertEqual(text,'किताबब');self.assertAlmostEqual(confidence,.9,places=5)
    def test_ctc_empty(self):
        self.assertEqual(ctc_decode(np.ones((1,4,1)),['blank']),('',0.0))
    def test_ctc_dict_mismatch(self):
        with self.assertRaises(FeatureError):ctc_decode(np.ones((1,4,2)),['blank'])
    def test_ctc_nan(self):
        with self.assertRaises(FeatureError):ctc_decode(np.full((1,4,2),np.nan),['blank','a'])
    def test_order_no_duplicate_corners(self):
        points=np.array([[10,0],[20,10],[10,20],[0,10]],np.float32)
        result=order_quad(points)
        self.assertEqual(len(set(map(tuple,result))),4)
    def test_order_regular_quad(self):
        points=np.array([[20,20],[10,10],[10,20],[20,10]],np.float32)
        np.testing.assert_array_equal(order_quad(points),[[10,10],[20,10],[20,20],[10,20]])
    def test_db_boxes(self):
        p=np.zeros((1,1,100,100),np.float32);p[0,0,20:35,20:70]=.95
        boxes=db_boxes(p,(200,200),self.cfg);self.assertEqual(len(boxes),1)
        self.assertTrue(np.all(boxes[0]>=0));self.assertTrue(np.all(boxes[0]<200))
    def test_db_low_probability(self):
        self.assertEqual(db_boxes(np.full((32,32),.1),(32,32),self.cfg),[])
    def test_db_nan(self):
        with self.assertRaises(FeatureError):db_boxes(np.full((32,32),np.nan),(32,32),self.cfg)
    def test_crop_and_normalization(self):
        image=np.full((100,100,3),(255,0,0),np.uint8)
        crop=crop_quad(image,np.array([[10,10],[80,10],[80,30],[10,30]],np.float32))
        self.assertEqual(crop.shape,(20,70,3))
        x=normalize_line(crop,48,320)
        self.assertEqual(x.shape,(1,3,48,320));np.testing.assert_allclose(x[0,:,0,0],[1,-1,-1])
        self.assertTrue(np.all(x[0,:,:,-1]==0))
    def test_vertical_crop_rotates(self):
        crop=crop_quad(np.zeros((100,100,3),np.uint8),np.array([[10,10],[25,10],[25,90],[10,90]],np.float32))
        self.assertGreater(crop.shape[1],crop.shape[0])
    def test_reading_order(self):
        def box(x,y):return np.array([[x,y],[x+20,y],[x+20,y+10],[x,y+10]],np.float32)
        boxes=reading_order([box(0,50),box(40,10),box(0,10)])
        self.assertEqual([(b[0,0],b[0,1]) for b in boxes],[(0,10),(40,10),(0,50)])
    def test_incomplete_description(self):
        r=OCRResult([TextLine('किताब',.9,())],True,2,1)
        self.assertIn('Some text could not be read',describe_text(r,'en'))
    def test_read_cancel(self):
        r=TextReader.__new__(TextReader);r.cfg=self.cfg;r.detect=lambda _: [np.zeros((4,2),np.float32)]
        cancel=threading.Event();cancel.set()
        with self.assertRaises(Cancelled):r.read(np.zeros((100,100,3),np.uint8),cancel)
    def test_read_cooperative_budget(self):
        r=TextReader.__new__(TextReader);r.cfg=VisionSettings(ocr_budget_s=-1)
        r.detect=lambda _:[np.zeros((4,2),np.float32)]
        result=r.read(np.zeros((100,100,3),np.uint8));self.assertTrue(result.incomplete);self.assertEqual(result.lines,[])
