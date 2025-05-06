# eval_deep_particle_filter.py

import sys
import time
import numpy as np
import torch
import cv2

from tracker_manager import TrackerManager
from blob_detection import detect_blobs

def compute_avg_deep_sim(
    video_path: str,
    num_particles: int = 75,
    motion_noise: float = 5.0,
    patch_size: int = 20
) -> (float, int):
    """
    Runs the deep‐feature particle filter over the entire video
    and returns (average deep‐feature similarity, number of frames processed).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    tracker = TrackerManager(
        num_particles=num_particles,
        noise=motion_noise,
        patch_size=patch_size,
        use_deep_features=True,
        device=device
    )

    sims = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # No global resize: use full-resolution
        blobs = detect_blobs(frame)
        tracker.update(frame, blobs)

        # collect deep‐feature similarity from each PF
        for pf, _ in tracker.trackers:
            if hasattr(pf, "last_deep_sim"):
                sims.append(pf.last_deep_sim)

        frame_count += 1

    cap.release()
    if not sims:
        return 0.0, frame_count
    return float(np.mean(sims)), frame_count

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python eval_deep_particle_filter.py path/to/video.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    start = time.time()
    avg_sim, n_frames = compute_avg_deep_sim(video_path)
    elapsed = time.time() - start

    print(f"\nProcessed {n_frames} frames in {elapsed:.1f}s")
    print(f"Average deep‐feature similarity: {avg_sim*100:.2f}%")

    if avg_sim >= 0.96:
        print("✅ Deep particle filter performance is ≥96%")
    else:
        print("❌ Deep particle filter performance is <96%")
