# yolo_wrapper.py

import torch
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path, imgsz, conf_thresh, iou_thresh, device, fp16):
        self.model = YOLO(model_path)
        self.model.fuse()
        self.device = device
        self.fp16 = fp16
        # these get applied per-inference
        self.imgsz = imgsz
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    def detect(self, frame):
        # run inference
        results = self.model(
            frame,
            imgsz=self.imgsz,
            device=self.device.index if hasattr(self.device, "index") else self.device,
            conf=self.conf_thresh,
            iou=self.iou_thresh
        )
        boxes = results[0].boxes.xyxy.cpu().numpy()
        dets = []
        for x1, y1, x2, y2 in boxes:
            w, h = int(x2 - x1), int(y2 - y1)
            dets.append((int(x1), int(y1), w, h))
        return dets
