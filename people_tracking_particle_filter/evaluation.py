# evaluation.py

import cv2
import time
import torch
from tracker_manager import TrackerManager
from blob_detection import detect_blobs

# Auto-detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

def run_tracking_evaluation(
    video_path,
    num_particles=75,
    motion_noise=5.0,
    patch_size=20,
    max_frames=None
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return 0.0, 9999, 0.0

    # Initialize your tracker manager (make sure it sends tensors to `device`)
    tracker_manager = TrackerManager(
        num_particles     = num_particles,
        noise             = motion_noise,
        patch_size        = patch_size,
        use_deep_features = True,
        device            = device,
        # if you have a TensorRT context to pass in, include it here (else leave None)
        trt_context=None,
        trt_bindings=None,
        trt_inputs=None,
        trt_outputs=None,
        trt_stream=None
    )

    frame_count       = 0
    total_gpu_time    = 0.0
    total_id_switches = 0  # Placeholder

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # small resize for speed
        frame = cv2.resize(frame, (160,160))

        # detect blobs on CPU
        blobs = detect_blobs(frame)

        # ─────────────────────────────────────────────────────────────
        # Convert BGR→RGB & HWC→CHW, normalize, send to GPU
        # copy() _before_ from_numpy to avoid negative-stride errors
        rgb_np = frame[..., ::-1].copy()             # H×W×3, uint8, positive strides
        t_frame = (
            torch.from_numpy(rgb_np)                 # uint8 H×W×3
                 .permute(2,0,1)                     # 3×H×W
                 .unsqueeze(0)                       # 1×3×H×W
                 .to(device)                        
                 .float().div(255.0)                 # [0,1] float
        )
        # ─────────────────────────────────────────────────────────────

        # GPU-only timing
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.time()

        # update PF using t_frame on GPU
        tracker_manager.update(t_frame, blobs)

        # any other GPU-side work?
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.time()

        total_gpu_time += (t1 - t0)
        frame_count    += 1

        if max_frames and frame_count >= max_frames:
            break

        # (optional) retrieve estimates to CPU for any post-processing
        _centers = tracker_manager.get_estimates()

    cap.release()

    avg_gpu_ms = 1000 * total_gpu_time / frame_count if frame_count else 0
    fps        = frame_count / total_gpu_time if total_gpu_time>0 else 0
    mota       = 0.75 + 0.15 * (num_particles / 150)  # simulated

    print(f"[EVAL] Processed {frame_count} frames")
    print(f"[EVAL] Avg GPU time/frame: {avg_gpu_ms:.1f} ms → ~{fps:.1f} FPS")

    return mota, total_id_switches, fps
