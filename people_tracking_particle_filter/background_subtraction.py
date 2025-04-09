import cv2

def create_background_subtractor():
    return cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

def get_foreground_mask(subtractor, frame):
    fg_mask = subtractor.apply(frame)
    return fg_mask
