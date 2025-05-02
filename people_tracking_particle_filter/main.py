import os                # <<< for path handling
import cv2
import torch
from blob_detection import detect_blobs
from tracker_manager import TrackerManager
from old_utils import draw_particles, draw_tracking
from config import (
    NUM_PARTICLES,
    MOTION_NOISE,
    PATCH_SIZE,
    USE_DEEP_FEATURES,
    VIDEO_PATH,
    OUTPUT_PATH
)

# ---- NEW TRT imports ----
import tensorrt as trt            # <<< 
from tensorrt_utils import (      # <<<
    load_engine, 
    allocate_buffers, 
    do_inference
)

# point this at wherever you built/saved your .trt
TRT_ENGINE_PATH = r"C:\Users\abhin\OneDrive\Documents\GitHub\Particle-Filter-People_tracking\feat_extractor.trt"  # <<<

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running on: {device}")

    # ---- NEW: build TRT runtime + engine + buffers ----
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime     = trt.Runtime(TRT_LOGGER)
    engine      = load_engine(runtime, TRT_ENGINE_PATH)
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    context     = engine.create_execution_context()
    print(f"[INFO] Loaded TRT engine from {TRT_ENGINE_PATH}")

    cap = cv2.VideoCapture(VIDEO_PATH)

    tracker_manager = TrackerManager(
        num_particles=NUM_PARTICLES,
        noise=MOTION_NOISE,
        patch_size=PATCH_SIZE,
        use_deep_features=USE_DEEP_FEATURES,
        device=device,

        # <<< pass TRT bits into your tracker so it can do feature extraction
        trt_context = context,
        trt_bindings = bindings,
        trt_inputs = inputs,
        trt_outputs = outputs,
        trt_stream = stream,
    )

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*'XVID'),
        20.0,
        (frame_width, frame_height)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        blobs = detect_blobs(frame)
        # tracker_manager should internally call do_inference() when
        # USE_DEEP_FEATURES is True, using the TRT buffers you passed in.
        tracker_manager.update(frame, blobs)
        centers = tracker_manager.get_estimates()

        for pf, _ in tracker_manager.trackers:
            draw_particles(frame, pf.particles)
        draw_tracking(frame, centers)

        cv2.imshow("Tracking", frame)
        out.write(frame)

        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
