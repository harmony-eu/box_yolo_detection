# !/usr/bin/env python3

import cv2
import numpy as np
import os
import sys
import torch
import torch.backends.cudnn as cudnn

from pathlib import Path

import rospkg
import rospy

from cv_bridge import CvBridge
from rostopic import get_topic_type
from detection_msgs.msg import BoundingBox, BoundingBoxes
from sensor_msgs.msg import Image, CompressedImage
from std_srvs.srv import SetBool, SetBoolResponse


# add yolov5 submodule to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / "yolov5"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative path

# import from yolov5 submodules
from models.common import DetectMultiBackend
from utils.general import (
    check_img_size,
    check_requirements,
    non_max_suppression,
    scale_coords
)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox


@torch.no_grad()
class Yolov5Detector:
    def __init__(self, weights, data_yaml):
        self.conf_thres = 0.75
        self.iou_thres = 0.45
        self.agnostic_nms = True
        self.max_det = 1000
        self.classes = None
        self.line_thickness = 3
        self.view_image = False

        self.inference_size_w = 1280
        self.inference_size_h = 720
        # Initialize weights 
        rospack = rospkg.RosPack()
        box_path = rospack.get_path('box_code')

        self.weights = os.path.join(box_path, weights)
        self.data = os.path.join(box_path, data_yaml)
        self.publish_image = False

        self.input_image_topic = None
        self.output_topic = None
        self.output_image_topic = None
        self.on_off_service_name = None


    def init_model(self):
        # Initialize model
        self.device = select_device(str(0))
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=True, data=self.data)
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = (
            self.model.stride,
            self.model.names,
            self.model.pt,
            self.model.jit,
            self.model.onnx,
            self.model.engine,
        )

        # Setting inference size
        self.img_size = [self.inference_size_w, self.inference_size_h]
        self.img_size = check_img_size(self.img_size, s=self.stride)

        # Half
        self.half = False
        self.half &= (
            self.pt or self.jit or self.onnx or self.engine
        ) and self.device.type != "cpu"  # FP16 supported on limited backends with CUDA
        if self.pt or self.jit:
            self.model.model.half() if self.half else self.model.model.float()
        bs = 1  # batch_size
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.model.warmup()  # warmup        
        
        # Initialize subscriber to Image/CompressedImage topic
        input_image_type, input_image_topic, _ = get_topic_type(self.input_image_topic, blocking = True)
        self.compressed_input = input_image_type == "sensor_msgs/CompressedImage"

        self.on_off_service = rospy.Service(self.on_off_service_name, SetBool, self.on_of_callback)

        # Initialize prediction publisher
        self.pred_pub = rospy.Publisher(
            self.output_topic, BoundingBoxes, queue_size=10
        )

        
        # Initialize image publisher
        if self.publish_image:
            self.image_pub = rospy.Publisher(
                self.output_image_topic, Image, queue_size=10
            )
        
        # Initialize CV_Bridge
        self.bridge = CvBridge()


    def on_of_callback(self, req):
        msg = "yolo turned "
        if req.data is True:
            if self.compressed_input:
                self.image_sub = rospy.Subscriber(
                    self.input_image_topic, CompressedImage, self.input_image_callback, queue_size=1
                )
            else:
                self.image_sub = rospy.Subscriber(
                    self.input_image_topic, Image, self.input_image_callback, queue_size=1
                )
            msg += "on"
        else:
            self.image_sub.unregister()
            self.image_sub = None
            msg += "off"

        return SetBoolResponse(True, msg)



    def input_image_callback(self, data):
        """adapted from yolov5/detect.py"""
        if self.compressed_input:
            im = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
        else:
            im = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        
        im, im0 = self.preprocess(im)

        # Run inference
        im = torch.from_numpy(im).to(self.device) 
        im = im.half() if self.half else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det
        )

        ### To-do move pred to CPU and fill BoundingBox messages
        
        # Process predictions 
        det = pred[0].cpu().numpy()

        bounding_boxes = BoundingBoxes()
        bounding_boxes.header = data.header
        bounding_boxes.image_header = data.header
        
        annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                bounding_box = BoundingBox()
                c = int(cls)
                # Fill in bounding box message
                bounding_box.Class = self.names[c]
                bounding_box.probability = conf 
                bounding_box.xmin = int(xyxy[0])
                bounding_box.ymin = int(xyxy[1])
                bounding_box.xmax = int(xyxy[2])
                bounding_box.ymax = int(xyxy[3])

                bounding_boxes.bounding_boxes.append(bounding_box)

                # Annotate the image
                if self.publish_image or self.view_image:  # Add bbox to image
                      # integer class
                    label = f"{self.names[c]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))       

                
                ### POPULATE THE DETECTION MESSAGE HERE

            # Stream results
            im0 = annotator.result()

        # Publish prediction
        self.pred_pub.publish(bounding_boxes)

        # Publish & visualize images
        if self.view_image:
            cv2.imshow(str(0), im0)
            cv2.waitKey(1)  # 1 millisecond
        if self.publish_image:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(im0, "bgr8"))
        

    def preprocess(self, img):
        """
        Adapted from yolov5/utils/datasets.py LoadStreams class
        """
        img0 = img.copy()
        img = np.array([letterbox(img, self.img_size, stride=self.stride, auto=self.pt)[0]])
        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return img, img0 


if __name__ == "__main__":
    node_name = 'box_detector'

    check_requirements(exclude=("tensorboard", "thop"))
    
    rospy.init_node(node_name, anonymous=False)
    detector = Yolov5Detector(weights='models/best.pt', data_yaml='models/model.yaml')
    # set communication options
    detector.input_image_topic = '/kinect/rgb/image_raw'
    detector.output_topic = node_name + '/detections'
    detector.publish_image = True
    detector.output_image_topic = node_name + '/image_output'
    detector.on_off_service_name = node_name + '/set_on_state'

    # start computation
    detector.init_model()
    
    rospy.spin()
