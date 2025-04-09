import cv2

def create_background_subtractor():
    return cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

def get_foreground_mask(subtractor, frame):
    fg_mask = subtractor.apply(frame)
    _, fg_mask = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)  # Only strong motion
    fg_mask = cv2.medianBlur(fg_mask, 5)  # Smooth
    fg_mask = cv2.dilate(fg_mask, None, iterations=2)  # Fill holes
    fg_mask = cv2.erode(fg_mask, None, iterations=1)  # Remove noise
    return fg_mask
