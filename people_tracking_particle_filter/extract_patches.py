# blob_detection.py
import os
import cv2
import torch
from ultralytics import YOLO

# ✅ Load YOLOv8n model
model = YOLO('yolov8n.pt')  # Class 0 = person in COCO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# Path to video
VIDEO_PATH = "C:\\Users\\abhin\\OneDrive\\Documents\\GitHub\\Particle-Filter-People_tracking\\people_tracking_particle_filter\\sample_videos\\test_video.mp4"

# Output folder for patches
OUTPUT_FOLDER = "dataset/patches"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def save_blobs_from_video(video_path=VIDEO_PATH):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    total_patches = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model.predict(
            source=frame,
            device=device.index if device.type == 'cuda' else 'cpu',
            verbose=False
        )

        for r in results:
            for i, (box, cls) in enumerate(zip(r.boxes.xyxy, r.boxes.cls)):
                if int(cls) == 0:  # Only 'person'
                    x1, y1, x2, y2 = map(int, box)
                    person_crop = frame[y1:y2, x1:x2]

                    # Save patch
                    filename = os.path.join(OUTPUT_FOLDER, f"frame{frame_idx}_person{i}.jpg")
                    cv2.imwrite(filename, person_crop)
                    total_patches += 1

        frame_idx += 1

    cap.release()
    print(f"[INFO] Saved {total_patches} patches from {frame_idx} frames to {OUTPUT_FOLDER}")
if __name__ == "__main__":
    save_blobs_from_video()
     