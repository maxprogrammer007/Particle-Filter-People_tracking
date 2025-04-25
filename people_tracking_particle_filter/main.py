import cv2
import torch
from blob_detection import detect_blobs
from tracker_manager import TrackerManager
from utils import draw_particles, draw_tracking
from config import (
    NUM_PARTICLES,
    MOTION_NOISE,
    PATCH_SIZE,
    USE_DEEP_FEATURES,
    VIDEO_PATH,
    OUTPUT_PATH
)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running on: {device}")

    cap = cv2.VideoCapture(VIDEO_PATH)

    tracker_manager = TrackerManager(
        num_particles=NUM_PARTICLES,
        noise=MOTION_NOISE,
        patch_size=PATCH_SIZE,
        use_deep_features=USE_DEEP_FEATURES,
        device=device  # 👈 Forward device
    )

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(OUTPUT_PATH,
                          cv2.VideoWriter_fourcc(*'XVID'),
                          20.0,
                          (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        blobs = detect_blobs(frame)
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