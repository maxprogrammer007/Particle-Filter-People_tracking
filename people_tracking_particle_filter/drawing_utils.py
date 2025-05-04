# drawing_utils.py

import cv2

def draw_particles(frame, particles):
    for x, y in particles:
        cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

def draw_tracking(frame, centers):
    """
    centers should be an iterable of (x,y) tuples or lists.
    We filter out any Nones, cast to ints, and then draw.
    """
    for c in centers:
        if c is None:
            continue
        # unpack
        x, y = c
        # make sure they are real numbers
        if x is None or y is None:
            continue
        # cast to int
        xi, yi = int(round(x)), int(round(y))
        cv2.circle(frame, (xi, yi), 20, (255, 0, 0), 2)
