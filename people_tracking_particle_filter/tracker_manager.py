from particle_filter import ParticleFilter
from histogram_model import get_color_histogram

class TrackerManager:
    def __init__(self):
        self.trackers = []

    def update(self, frame, detections):
        # TODO: Add smart association (for now, assume 1-to-1)
        if len(self.trackers) < len(detections):
            for det in detections[len(self.trackers):]:
                x, y, w, h = det
                center = (x + w // 2, y + h // 2)
                patch = frame[y:y+h, x:x+w]
                hist = get_color_histogram(patch)
                pf = ParticleFilter(center)
                self.trackers.append((pf, hist))

        for pf, hist in self.trackers:
            pf.predict()
            pf.update(frame, hist, get_color_histogram)

    def get_estimates(self):
        return [pf.estimate() for pf, _ in self.trackers]
