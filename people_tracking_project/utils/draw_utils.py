# utils/draw_utils.py

import cv2

def draw_tracks(frame, tracks, color=(0, 255, 0)):
    for tid, bbox in tracks:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {tid}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

def draw_metrics(frame, mota, idf1, fps):
    text = f"MOTA: {mota:.3f}  IDF1: {idf1:.3f}  FPS: {fps:.2f}"
    cv2.putText(frame, text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame
