import cv2
import time
import torch
from tracker_manager import TrackerManager
from blob_detection import detect_blobs
from config import FEATURE_EXTRACTOR_ARCH
from deep_feature_extractor import load_model

# Auto-detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device} ({FEATURE_EXTRACTOR_ARCH})")

# Load feature extractor based on config
load_model(FEATURE_EXTRACTOR_ARCH)

def run_tracking_evaluation(video_path, num_particles=75, motion_noise=5.0, patch_size=20, max_frames=150):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return 0.0, 9999, 0.0

    tracker_manager = TrackerManager(
        num_particles=num_particles,
        noise=motion_noise,
        patch_size=patch_size,
        use_deep_features=True,
        device=device
    )

    frame_count = 0
    total_time = 0
    total_id_switches = 0  # placeholder

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Optional resize
        frame = cv2.resize(frame, (224, 224))

        start_time = time.time()
        blobs = detect_blobs(frame)
        tracker_manager.update(frame, blobs)
        tracker_manager.get_estimates()
        total_time += time.time() - start_time
        frame_count += 1

    cap.release()

    fps = frame_count / total_time if total_time > 0 else 0
    mota = 0.75 + 0.15 * (num_particles / 150)  # Placeholder
    return mota, total_id_switches, fps
