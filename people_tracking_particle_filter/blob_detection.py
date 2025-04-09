import cv2

# Create the HOG person detector once
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detect_blobs(frame):
    # Resize frame to improve detection speed and accuracy
    frame_resized = cv2.resize(frame, (640, 480))
    
    # Detect people
    rects, _ = hog.detectMultiScale(frame_resized, winStride=(8,8))
    
    # Adjust coordinates back to original frame size
    scale_x = frame.shape[1] / 640
    scale_y = frame.shape[0] / 480
    detections = []
    
    for (x, y, w, h) in rects:
        detections.append((
            int(x * scale_x),
            int(y * scale_y),
            int(w * scale_x),
            int(h * scale_y)
        ))
    
    return detections
