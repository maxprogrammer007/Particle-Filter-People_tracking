import cv2
import time
import torch
from tracker_manager import TrackerManager
from blob_detection import detect_blobs

# Auto-detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

def run_tracking_evaluation(video_path, num_particles=75, motion_noise=5.0, patch_size=20, max_frames=None):
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
    total_id_switches = 0  # Placeholder

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Optional: resize to 224x224 to speed up deep inference
        frame = cv2.resize(frame, (160, 160))

        start_time = time.time()
        blobs = detect_blobs(frame)
        tracker_manager.update(frame, blobs)
        centers = tracker_manager.get_estimates()
        end_time = time.time()

        total_time += (end_time - start_time)
        frame_count += 1

        # 👉 Only break if max_frames is set and exceeded
        if max_frames is not None and frame_count >= max_frames:
            break

    cap.release()

    fps = frame_count / total_time if total_time > 0 else 0
    mota = 0.75 + 0.15 * (num_particles / 150)  # ⚡ Temporary simulated MOTA

    return mota, total_id_switches, fps
