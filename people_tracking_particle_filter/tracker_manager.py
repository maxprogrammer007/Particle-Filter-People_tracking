# tracker_manager.py
from particle_filter import ParticleFilter
from histogram_model import get_color_histogram

class TrackerManager:
    def __init__(self, num_particles=75, noise=5.0, patch_size=20):
        self.trackers = []
        self.num_particles = num_particles
        self.noise = noise
        self.patch_size = patch_size

    def update(self, frame, detections):
        if len(self.trackers) < len(detections):
            for det in detections[len(self.trackers):]:
                x, y, w, h = det
                center = (x + w // 2, y + h // 2)
                patch = frame[y:y+h, x:x+w]
                hist = get_color_histogram(patch)
                pf = ParticleFilter(center,
                                    num_particles=self.num_particles,
                                    noise=self.noise,
                                    patch_size=self.patch_size)
                self.trackers.append((pf, hist))

        for pf, hist in self.trackers:
            pf.predict()
            pf.update(frame, hist, get_color_histogram)

    def get_estimates(self):
        return [pf.estimate() for pf, _ in self.trackers]