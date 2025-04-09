import cv2
from background_subtraction import create_background_subtractor, get_foreground_mask
from blob_detection import detect_blobs
from tracker_manager import TrackerManager
from utils import draw_particles, draw_tracking

def main():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or replace with video path
    bg_subtractor = create_background_subtractor()
    tracker_manager = TrackerManager()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = get_foreground_mask(bg_subtractor, frame)
        blobs = detect_blobs(fg_mask)

        tracker_manager.update(frame, blobs)
        centers = tracker_manager.get_estimates()

        # Draw tracking results
        for pf, _ in tracker_manager.trackers:
            draw_particles(frame, pf.particles)
        draw_tracking(frame, centers)

        cv2.imshow("Tracking", frame)

        if cv2.waitKey(30) & 0xFF == 27:  # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
