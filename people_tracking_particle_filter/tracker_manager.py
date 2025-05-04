# tracker_manager.py

from particle_filter import ParticleFilter

class TrackerManager:
    def __init__(
        self,
        num_particles,
        noise,
        patch_size,
        use_deep_features,
        device,
        trt_context, trt_bindings, trt_inputs, trt_outputs, trt_stream
    ):
        self.trackers = []
        self.params   = dict(
            num_particles=num_particles,
            noise=noise,
            patch_size=patch_size,
            use_deep_features=use_deep_features,
            device=device,
            trt_context=trt_context,
            trt_bindings=trt_bindings,
            trt_inputs=trt_inputs,
            trt_outputs=trt_outputs,
            trt_stream=trt_stream
        )

    def update(self, frame, blobs, target_patch):
        # spawn new PFs if new blobs appear
        while len(self.trackers) < len(blobs):
            pos = blobs[len(self.trackers)][:2]
            pf  = ParticleFilter(initial_pos=pos, **self.params)
            self.trackers.append(pf)

        # update all trackers (all use GPU under the hood)
        for pf in self.trackers:
            pf.predict()
            pf.update(frame, blobs, target_patch)

    def get_estimates(self):
        # return list of (x,y) per tracker
        return [pf.estimate() for pf in self.trackers]
