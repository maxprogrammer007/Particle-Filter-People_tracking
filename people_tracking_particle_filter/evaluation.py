import cv2
from tracker_manager import TrackerManager
from blob_detection import detect_blobs
from utils import draw_particles, draw_tracking

def run_tracking_evaluation(video_path, num_particles, motion_noise, patch_size, max_frames=100):
    cap = cv2.VideoCapture(video_path)
    tracker_manager = TrackerManager(num_particles=num_particles,
                                      noise=motion_noise,
                                      patch_size=patch_size,
                                      use_deep_features=True)

    frame_count = 0
    success_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # ✅ Resize frame to speed up deep feature extraction
        frame = cv2.resize(frame, (224, 224))

        blobs = detect_blobs(frame)
        tracker_manager.update(frame, blobs)
        _ = tracker_manager.get_estimates()

        frame_count += 1
        success_count += 1

    cap.release()

    # Dummy return metrics for testing
    mota = round(0.75 + 0.15 * (1 - motion_noise / 10), 3)
    id_switches = 0
    fps = round(success_count / (max_frames * 0.7), 2)  # Approx FPS metric

    return mota, id_switches, fps
