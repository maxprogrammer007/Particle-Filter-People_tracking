# tracker_manager.py

from particle_filter import ParticleFilter

class TrackerManager:
    def __init__(
        self,
        num_particles=75,
        noise=5.0,
        patch_size=20,
        use_deep_features=True,
        device=None,
        # <<< NEW: accept these five TRT-related objects
        trt_context=None,
        trt_bindings=None,
        trt_inputs=None,
        trt_outputs=None,
        trt_stream=None
    ):
        self.trackers = []
        self.num_particles       = num_particles
        self.noise               = noise
        self.patch_size          = patch_size
        self.use_deep_features   = use_deep_features
        self.device              = device

        # <<< STORE the TRT objects on self for later
        self.trt_context  = trt_context
        self.trt_bindings = trt_bindings
        self.trt_inputs   = trt_inputs
        self.trt_outputs  = trt_outputs
        self.trt_stream   = trt_stream

    def update(self, frame, detections):
        # if more detections than existing trackers, spawn new PFs
        if len(self.trackers) < len(detections):
            for det in detections[len(self.trackers):]:
                x, y, w, h = det
                center = (x + w // 2, y + h // 2)
                patch  = frame[y : y + h, x : x + w]

                # <<< Pass all TRT bits into every new PF
                pf = ParticleFilter(
                    center,
                    num_particles      = self.num_particles,
                    noise              = self.noise,
                    patch_size         = self.patch_size,
                    device             = self.device,
                    use_deep_features  = self.use_deep_features,
                    # TRT args:
                    trt_context  = self.trt_context,
                    trt_bindings = self.trt_bindings,
                    trt_inputs   = self.trt_inputs,
                    trt_outputs  = self.trt_outputs,
                    trt_stream   = self.trt_stream,
                )
                self.trackers.append((pf, patch))

        # then for every existing PF, do predict + update
        for pf, target_patch in self.trackers:
            pf.predict()
            pf.update(frame, target_patch)

    def get_estimates(self):
        return [pf.estimate() for pf, _ in self.trackers]
