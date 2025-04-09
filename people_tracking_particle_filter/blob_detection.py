import cv2

def detect_blobs(foreground_mask):
    contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:  # filter small blobs
            x, y, w, h = cv2.boundingRect(cnt)
            blobs.append((x, y, w, h))
    return blobs
