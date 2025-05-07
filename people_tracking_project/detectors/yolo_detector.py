# detectors/yolo_detector.py

from ultralytics import YOLO
import cv2

class YOLODetector:
    def __init__(self, model_name="yolov8n.pt", img_size=640, conf_thresh=0.5):
        self.model = YOLO(model_name)
        self.img_size = img_size
        self.conf_thresh = conf_thresh

    def detect(self, frame):
        results = self.model.predict(source=frame, imgsz=self.img_size, conf=self.conf_thresh, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                if box.conf.item() >= self.conf_thresh:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf)
                    cls = int(box.cls)
                    detections.append([x1, y1, x2, y2, conf, cls])
        return detections
