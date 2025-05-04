# main.py
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import cv2
import torch
import tensorrt as trt

from old_utils        import extract_batch_features
from blob_detection   import detect_blobs
from drawing_utils    import draw_particles, draw_tracking
from tracker_manager  import TrackerManager
from config           import (
    NUM_PARTICLES,
    MOTION_NOISE,
    PATCH_SIZE,
    USE_DEEP_FEATURES,
    VIDEO_PATH,
    OUTPUT_PATH
)
from tensorrt_utils   import load_engine, allocate_buffers

TRT_ENGINE_PATH = (
    r"C:\\Users\\abhin\\OneDrive\\Documents\\GitHub\\Particle-Filter-People_tracking\\feat_extractor.trt"
)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1) Load TRT engine & buffers
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = load_engine(runtime, TRT_ENGINE_PATH)
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    context = engine.create_execution_context()
    print(f"[INFO] Loaded TensorRT engine.")

    # 2) PF manager
    tracker = TrackerManager(
        num_particles     = NUM_PARTICLES,
        noise             = MOTION_NOISE,
        patch_size        = PATCH_SIZE,
        use_deep_features = USE_DEEP_FEATURES,
        device            = device,
        trt_context       = context,
        trt_bindings      = bindings,
        trt_inputs        = inputs,
        trt_outputs       = outputs,
        trt_stream        = stream,
    )

    cap = cv2.VideoCapture(VIDEO_PATH)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_PATH,
                          cv2.VideoWriter_fourcc(*"XVID"),
                          20, (fw, fh))

    first = True
    target_patch = None

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        blobs = detect_blobs(frame)
        if first and blobs:
            # pick first detected blob as target
            x,y,w,h = blobs[0]
            target_patch = frame[y:y+h, x:x+w]
            first = False

        tracker.update(frame, blobs, target_patch)

        # draw
        for pf in tracker.trackers:
            draw_particles(frame, pf.particles)
        centers = tracker.get_estimates()
        draw_tracking(frame, centers)

        cv2.imshow("Track", frame)
        out.write(frame)
        if cv2.waitKey(1)==27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
