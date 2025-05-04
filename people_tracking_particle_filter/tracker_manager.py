# tracker_manager.py

from particle_filter import ParticleFilter

class TrackerManager:
    def __init__(
        self,
        num_particles=75,
        noise=5.0,
        patch_size=20,
        use_deep_features=True,
        device=None
    ):
        self.trackers = []  # List of tuples: (ParticleFilter, last_patch)
        self.num_particles = num_particles
        self.noise = noise
        self.patch_size = patch_size
        self.use_deep_features = use_deep_features
        self.device = device

    def update(self, frame, detections):
        # 1) Add new trackers if we have new detections
        if detections:
            for det in detections[len(self.trackers):]:
                x, y, w, h = det
                center = (x + w // 2, y + h // 2)
                patch = frame[y : y + h, x : x + w]
                pf = ParticleFilter(
                    center,
                    num_particles=self.num_particles,
                    noise=self.noise,
                    patch_size=self.patch_size,
                    device=self.device
                )
                self.trackers.append((pf, patch))

        # 2) Predict step for all trackers
        for pf, _ in self.trackers:
            pf.predict()

        # 3) Update step: only when there are detections this frame
        if not detections:
            return  # skip update on stride frames

        # Call the built-in update() which batches ROI->features internally
        for pf, patch in self.trackers:
            pf.update(frame, patch, use_deep_features=self.use_deep_features)

    def get_estimates(self):
        return [pf.estimate() for pf, _ in self.trackers]
