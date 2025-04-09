import cv2
from blob_detection import detect_blobs
from tracker_manager import TrackerManager
from utils import draw_particles, draw_tracking

def main():
    cap = cv2.VideoCapture("C:\\Users\\abhin\\OneDrive\\Documents\\GitHub\\Particle-Filter-People_tracking\\people_tracking_particle_filter\\sample_videos\\test_video.mp4")

    tracker_manager = TrackerManager()

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

        # No background subtraction here anymore
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
