# blob_detection.py
import torch
from ultralytics import YOLO

# ✅ Load YOLOv8n (Nano) model once
model = YOLO('yolov8n.pt')  # This will download/load YOLOv8 nano model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

def detect_blobs(frame):
    """Detect humans using YOLOv8n."""
    results = model.predict(source=frame, device=device.index if device.type == 'cuda' else 'cpu', verbose=False)

    detections = []
    for r in results:
        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
            if int(cls) == 0:  # Class 0 = 'person' for COCO
                x1, y1, x2, y2 = box
                detections.append((
                    int(x1.item()),
                    int(y1.item()),
                    int((x2 - x1).item()),
                    int((y2 - y1).item())
                ))

    return detections
