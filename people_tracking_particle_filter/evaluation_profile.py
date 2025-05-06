# evaluation_profile.py

import cv2
import time
import torch
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
from tracker_manager import TrackerManager
from blob_detection import detect_blobs
from config import VIDEO_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

def profile_run(num_particles, motion_noise, patch_size, max_frames=50):
    cap = cv2.VideoCapture(VIDEO_PATH)
    tracker = TrackerManager(
        num_particles=num_particles,
        noise=motion_noise,
        patch_size=patch_size,
        
        device=device
    )

    frame_count = 0
    all_times = []

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True
    ) as prof:
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            #frame = cv2.resize(frame, (160,160))

            with record_function("detect_and_update"):
                blobs = detect_blobs(frame)
                tracker.update(frame, blobs)
            frame_count += 1

    cap.release()

    # Print top ops by CUDA time
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=10
    ))

    # Export trace for chrome://tracing
    prof.export_chrome_trace("trace.json")

if __name__ == "__main__":
    # profile a small segment
    profile_run(num_particles=100, motion_noise=5.0, patch_size=20, max_frames=50)
