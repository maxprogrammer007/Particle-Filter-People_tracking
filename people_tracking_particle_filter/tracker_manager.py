from particle_filter import ParticleFilter

class TrackerManager:
    def __init__(self, num_particles=75, noise=5.0, patch_size=20, use_deep_features=True):
        self.trackers = []
        self.num_particles = num_particles
        self.noise = noise
        self.patch_size = patch_size
        self.use_deep_features = use_deep_features

    def update(self, frame, detections):
        # Initialize new trackers if needed
        if len(self.trackers) < len(detections):
            for det in detections[len(self.trackers):]:
                x, y, w, h = det
                center = (x + w // 2, y + h // 2)
                patch = frame[y:y + h, x:x + w]

                pf = ParticleFilter(center,
                                    num_particles=self.num_particles,
                                    noise=self.noise,
                                    patch_size=self.patch_size)
                self.trackers.append((pf, patch))

        # Update all trackers
        for pf, target_patch in self.trackers:
            pf.predict()
            pf.update(frame, target_patch, use_deep_features=self.use_deep_features)

    def get_estimates(self):
        return [pf.estimate() for pf, _ in self.trackers]
