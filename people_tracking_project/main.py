# main.py
import cv2
from blob_detection import detect_blobs
from tracker_manager import TrackerManager
from utils import draw_particles, draw_tracking

# Optional: pass parameters (these can also come from NSGA-II)
NUM_PARTICLES = 75
MOTION_NOISE = 5.0
PATCH_SIZE = 20

def main():
    cap = cv2.VideoCapture("sample_videos/test_video.mp4")

    tracker_manager = TrackerManager(
        num_particles=NUM_PARTICLES,
        noise=MOTION_NOISE,
        patch_size=PATCH_SIZE
    )

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter('output_tracking.avi', 
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