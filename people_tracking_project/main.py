# main.py

import cv2
import time
from detectors.yolo_detector import YOLODetector
from trackers.deepsort_wrapper import TrackByDetection
from utils.draw_utils import draw_tracks, draw_metrics
from evaluation.evaluation import compute_mock_metrics

# --- Paths ---
input_path = "people_tracking_project\\sample_videos\\test_video.mp4"
output_path = "people_tracking_project\\results\\output_tracking.mp4"


# --- Config ---
conf_thresh = 0.5
img_size = 640
iou_thresh = 0.5
skip_interval = 1

# --- Init Detector + Tracker ---
detector = YOLODetector(model_name="yolov8n.pt", img_size=img_size, conf_thresh=conf_thresh)
tracker = TrackByDetection(conf_thresh, img_size, iou_thresh, skip_interval)

# --- Init Video Capture ---
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video: {input_path}")

fps_input = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# --- Init Video Writer ---
out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps_input,
    (width, height)
)

# --- Tracking Loop ---
frame_count = 0
total_time = 0
all_tracks = []

print("[INFO] Starting tracking... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()
    detections = detector.detect(frame)
    formatted_dets = [[*d[:4], d[4]] for d in detections if d[5] == 0]
    tracks = tracker.update(formatted_dets, frame)
    end = time.time()

    all_tracks.extend(tracks)
    frame_count += 1
    total_time += (end - start)

    # --- Draw + Show ---
    frame = draw_tracks(frame, tracks)
    fps = frame_count / total_time if total_time > 0 else 0
    mota, idf1 = compute_mock_metrics(all_tracks)
    frame = draw_metrics(frame, mota, idf1, fps)

    out.write(frame)
    cv2.imshow("Live Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"[✓] Tracking complete. Output saved to: {output_path}")
print(f"[INFO] Average FPS: {frame_count / total_time:.2f}")
print(f"[INFO] Total frames processed: {frame_count}")
print(f"[INFO] Total time taken: {total_time:.2f} seconds")
print(f"[INFO] MOTA: {mota:.3f}, IDF1: {idf1:.3f}")
print("[INFO] Done.")