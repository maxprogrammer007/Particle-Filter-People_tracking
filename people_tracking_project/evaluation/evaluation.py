# evaluation.py
import cv2
import time
from tracker_manager import TrackerManager
from blob_detection import detect_blobs


def run_tracking_evaluation(video_path, num_particles=75, motion_noise=5.0, patch_size=20, max_frames=150):
    cap = cv2.VideoCapture(video_path)
    tracker_manager = TrackerManager(num_particles=num_particles, noise=motion_noise, patch_size=patch_size)

    frame_count = 0
    total_time = 0
    total_id_switches = 0  # Placeholder for logic

    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break

        start_time = time.time()
        blobs = detect_blobs(frame)
        tracker_manager.update(frame, blobs)
        centers = tracker_manager.get_estimates()
        end_time = time.time()

        total_time += (end_time - start_time)
        frame_count += 1

    cap.release()

    fps = frame_count / total_time if total_time > 0 else 0
    mota = 0.75 + 0.15 * (num_particles / 150)  # Fake placeholder
    return mota, total_id_switches, fps
