# !/usr/bin/env python3

import argparse
import cv2
import numpy as np
import os
import sys
import torch
import torch.backends.cudnn as cudnn

from pathlib import Path
from time import sleep

import rclpy

from rclpy.node import Node

from cv_bridge import CvBridge
from detection_msgs.msg import BoundingBox, BoundingBoxes
from detection_msgs.srv import GetRgbDepthAndBbox
from sensor_msgs.msg import Image, CompressedImage

from message_filters import (
    TimeSynchronizer, 
    ApproximateTimeSynchronizer, 
    Subscriber
)

# add yolov5 submodule to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / "yolov5"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative path

# import from yolov5 submodules
from models.common import DetectMultiBackend
from utils.general import (check_img_size, check_requirements,
                           non_max_suppression, scale_coords)
from utils.torch_utils import select_device
from utils.augmentations import letterbox


@torch.no_grad()
class Yolov5Detector(Node):

    def __init__(self, weights, data_yaml):
        super().__init__('box_yolo_detection')
        self.conf_thres = 0.75
        self.iou_thres = 0.45
        self.agnostic_nms = True
        self.max_det = 1000
        self.classes = None
        self.line_thickness = 3
        self.view_image = False

        self.inference_size_w = 1280
        self.inference_size_h = 720
        self._target_object_class = ['box', 'Box']

        # Initialize weights
        this_file = Path(__file__).resolve()
        box_path = this_file.parents[1]

        self.weights = os.path.join(box_path, weights)
        self.data = os.path.join(box_path, data_yaml)

        self.input_rgb_topic = None
        self.input_depth_topic = None
        self.bb_service_name = None

        self.rgb_sub = None
        self._last_rgb_received = None
        self._last_depth_received = None
        self.selected_device = 0  # or 'cpu'

    def init_torch(self):
        # Initialize model
        self.device = select_device(str(self.selected_device))
        self.model = DetectMultiBackend(self.weights,
                                        device=self.device,
                                        dnn=True,
                                        data=self.data)
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
        # - FP16 supported on limited backends with CUDA
        self.half &= (self.pt or self.jit or self.onnx or
                      self.engine) and self.device.type != "cpu"
        if self.pt or self.jit:
            self.model.model.half() if self.half else self.model.model.float()
        bs = 1  # batch_size
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.model.warmup()  # warmup

    def intialize(self):
        self.init_torch()
        while len(self.get_publishers_info_by_topic(self.input_rgb_topic)) < 1:
            self.get_logger().info(
                'waiting for '+ str(self.input_rgb_topic) + " to be published..."
            )
            sleep(1.0)
        while len(self.get_publishers_info_by_topic(self.input_depth_topic)) < 1:
            self.get_logger().info(
                'waiting for '+ str(self.input_depth_topic) + " to be published..."
            )
            sleep(1.0)

        # Initialize subscriber to Image/CompressedImage topic
        ti = self.get_publishers_info_by_topic(self.input_rgb_topic)[0]
        self.compressed_rgb_input = bool(ti.topic_type == "sensor_msgs/msg/CompressedImage")

        # Initialize CV_Bridge
        self.bridge = CvBridge()

        if self.compressed_rgb_input:
            self.rgb_sub = Subscriber(self, CompressedImage, self.input_rgb_topic)
        else:
            self.rgb_sub = Subscriber(self, Image, self.input_rgb_topic)

        self.depth_sub = Subscriber(self, Image, self.input_depth_topic)


        self.ats = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=1, slop=0.05)
        self.ats.registerCallback(self.input_image_callback)
        # self.ts = TimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=1)
        # self.ts.registerCallback(self.input_image_callback)

        self.get_logger().info(
            'Synchronized subscription to '+ str(self.input_rgb_topic) +' and '+ str(self.input_depth_topic)
        )

        self.bb_srv = self.create_service(
            GetRgbDepthAndBbox, 
            self.bb_service_name, 
            self.bb_srv_callback
        )


    def input_image_callback(self, rgb_msg, depth_msg):
        self._last_rgb_received = rgb_msg
        self._last_depth_received = depth_msg


    def bb_srv_callback(
        self, 
        req: GetRgbDepthAndBbox.Request, 
        res: GetRgbDepthAndBbox.Response
    ):
        """adapted from yolov5/detect.py"""
        self.get_logger().info('Received box detection request')
        if self._last_rgb_received is not None:
            if self.compressed_rgb_input:
                im = self.bridge.compressed_imgmsg_to_cv2(
                    self._last_rgb_received,
                    desired_encoding="bgr8"
                )
            else:
                im = self.bridge.imgmsg_to_cv2(
                    self._last_rgb_received,
                    desired_encoding="bgr8"
                )

            im, im0 = self.preprocess(im)
            # Run inference
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.half else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]

            pred = self.model(im, augment=False, visualize=False)
            pred = non_max_suppression(pred,
                                       self.conf_thres,
                                       self.iou_thres,
                                       self.classes,
                                       self.agnostic_nms,
                                       max_det=self.max_det)

            # Process predictions
            det = pred[0].cpu().numpy()

            bounding_boxes = BoundingBoxes()
            bounding_boxes.header = self._last_rgb_received.header
            bounding_boxes.image_header = self._last_rgb_received.header

            if len(det):
                res.success.data = True
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4],
                                          im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    bounding_box = BoundingBox()
                    c = int(cls)
                    # Fill in bounding box message
                    bounding_box.object_class = self.names[c]
                    bounding_box.probability = float(conf)
                    bounding_box.xmin = int(xyxy[0])
                    bounding_box.ymin = int(xyxy[1])
                    bounding_box.xmax = int(xyxy[2])
                    bounding_box.ymax = int(xyxy[3])

                    bounding_boxes.bounding_boxes.append(bounding_box)

                # Select the best bounding box
                res.bbox = self._select_bbox(bounding_boxes)
                res.rgb = self._last_rgb_received
                res.depth = self._last_depth_received

                # Clear last img variable
                self.get_logger().info('Box detected!')

            else:
                res.success.data = False
                self.get_logger().warn('Box not detected!')
        else:
            res.success.data = False
            self.get_logger().error('No image received!')

            
        self._last_rgb_received = None
        self._last_depth_received = None

        return res

    def preprocess(self, img):
        """
        Adapted from yolov5/utils/datasets.py LoadStreams class
        """
        img0 = img.copy()
        img = np.array([
            letterbox(img, self.img_size, stride=self.stride, auto=self.pt)[0]
        ])
        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return img, img0

    def _select_bbox(self, bounding_boxes):
        high_score = -1
        high_box = None
        for bbox in bounding_boxes.bounding_boxes:
            if (bbox.probability > high_score and
                    str(bbox.object_class) in self._target_object_class):
                high_score = bbox.probability
                high_box = bbox

        return high_box


def main(args=None):
    rclpy.init(args=args)
    node_name = 'box_yolo_detection'

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type',
                        choices=[
                            model_name.name.rsplit(".pt", maxsplit=1)[0]
                            for model_name in sorted((FILE.parents[1] /
                                                      "models").rglob("*.pt"))
                        ],
                        required=True,
                        help="Type of model to load.")
    args = parser.parse_args()

    check_requirements(exclude=("tensorboard", "thop"))
    detector = Yolov5Detector(weights=f'models/{args.model_type}.pt',
                              data_yaml='models/model.yaml')
    # set communication options
    detector.input_rgb_topic = '/myumi_005/sensors/top_azure/rgb/image_raw/compressed'
    detector.input_depth_topic = '/myumi_005/sensors/top_azure/depth_to_rgb/image_raw'
    detector.bb_service_name = node_name + '/get_rgbd_and_bbox'

    # start computation
    detector.selected_device = '0'  # or 0 for cuda
    detector.intialize()
    rclpy.spin(detector)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
