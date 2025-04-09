import cv2

def draw_particles(frame, particles):
    for p in particles:
        cv2.circle(frame, (int(p.x), int(p.y)), 2, (0, 255, 0), -1)

def draw_tracking(frame, centers):
    for (x, y) in centers:
        cv2.circle(frame, (x, y), 8, (255, 0, 0), 2)
