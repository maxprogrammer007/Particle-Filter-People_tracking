# profile_tracker.py
import cv2, torch
from tracker_manager import TrackerManager
from blob_detection  import detect_blobs

VIDEO = "sample_videos/test_video.mp4"
MAX_FRAMES = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tracker = TrackerManager(
    num_particles=75,
    noise=5.0,
    patch_size=20,
    use_deep_features=True,
    device=device
)

cap = cv2.VideoCapture(VIDEO)
count = 0
while count < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.resize(frame, (160,160))
    blobs = detect_blobs(frame)
    tracker.update(frame, blobs)
    count += 1
cap.release()
