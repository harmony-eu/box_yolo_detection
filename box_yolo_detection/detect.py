# !/usr/bin/env python3

import argparse
import cv2
import numpy as np
import os
import sys
import torch
import torch.backends.cudnn as cudnn

from pathlib import Path

import rclpy

from rclpy.node import Node

from cv_bridge import CvBridge
from detection_msgs.msg import BoundingBox, BoundingBoxes
from sensor_msgs.msg import Image, CompressedImage
from std_srvs.srv import SetBool

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
from utils.plots import Annotator, colors
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
        # Initialize weights
        this_file = Path(__file__).resolve()
        box_path = this_file.parents[1]

        self.weights = os.path.join(box_path, weights)
        self.data = os.path.join(box_path, data_yaml)
        self.publish_image = False

        self.input_image_topic = None
        self.output_topic = None
        self.output_image_topic = None
        self.on_off_service_name = None

        self.image_sub = None
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
        # Initialize subscriber to Image/CompressedImage topic
        ti = self.get_publishers_info_by_topic(self.input_image_topic)[0]
        self.compressed_input = bool(ti.topic_type == "sensor_msgs/CompressedImage")

        self.on_off_service = self.create_service(SetBool, self.on_off_service_name, self.on_off_callback)

        # Initialize prediction publisher
        self.pred_pub = self.create_publisher(BoundingBoxes, self.output_topic, 10)

        # Initialize image publisher
        if self.publish_image:
            self.image_pub = self.create_publisher(Image, self.output_image_topic, 10)

        # Initialize CV_Bridge
        self.bridge = CvBridge()

    def on_off_callback(self, req: SetBool.Request, resp: SetBool.Response):
        msg = "yolo turned "
        if req.data is True:
            if self.compressed_input:
                self.image_sub = self.create_subscription(CompressedImage,
                                                          self.input_image_topic,
                                                          self.input_image_callback,
                                                          1)
            else:
                self.image_sub = self.create_subscription(Image,
                                                          self.input_image_topic,
                                                          self.input_image_callback,
                                                          1)
            msg += "on"
        else:
            if self.image_sub is not None:
                self.image_sub.destroy()
                self.image_sub = None
            msg += "off"

        print(msg)
        resp.success = True
        resp.message = msg
        return resp

    def input_image_callback(self, data):
        """adapted from yolov5/detect.py"""
        if self.compressed_input:
            im = self.bridge.compressed_imgmsg_to_cv2(data,
                                                      desired_encoding="bgr8")
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
        pred = non_max_suppression(pred,
                                   self.conf_thres,
                                   self.iou_thres,
                                   self.classes,
                                   self.agnostic_nms,
                                   max_det=self.max_det)

        ### To-do move pred to CPU and fill BoundingBox messages

        # Process predictions
        det = pred[0].cpu().numpy()

        bounding_boxes = BoundingBoxes()
        bounding_boxes.header = data.header
        bounding_boxes.image_header = data.header

        annotator = Annotator(im0,
                              line_width=self.line_thickness,
                              example=str(self.names))
        if len(det):
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
        img = np.array([
            letterbox(img, self.img_size, stride=self.stride, auto=self.pt)[0]
        ])
        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return img, img0


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
    detector.input_image_topic = '/camera/image_raw' #/camera/image_raw/compressed
    detector.output_topic = node_name + '/detections'
    detector.publish_image = True
    detector.output_image_topic = node_name + '/image_output'
    detector.on_off_service_name = node_name + '/set_on_state'

    # start computation
    detector.selected_device = 'cpu'  # or 0 for cuda
    detector.intialize()
    rclpy.spin(detector)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()