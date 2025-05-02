# particle_filter.py

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align
import numpy as np
import cv2

# pull in your TensorRT wrapper
from tensorrt_utils import do_inference

def preprocess_for_trt(patch, host_buffer):
    """
    Copy `patch` (H×W×C uint8 BGR) into the TensorRT host_buffer as NCHW float32.
    Adapt this to your model’s exact preprocessing (resize, mean/std, etc.).
    """
    # 1) resize to input size if needed
    h, w = host_buffer.shape[2:]    # expecting host_buffer shape = (batch, C, H, W)
    patch = cv2.resize(patch, (w, h))
    # 2) BGR->RGB, to float32, scale
    img = patch[..., ::-1].astype(np.float32) / 255.0
    # 3) HWC->CHW
    img = np.transpose(img, (2, 0, 1))
    # 4) copy into host_buffer[0]
    host_buffer[0] = img
    return

class ParticleFilter:
    def __init__(
        self,
        initial_pos,
        num_particles=50,
        noise=5.0,
        patch_size=20,
        device=None,
        use_deep_features=False,
        # ← these five get passed in from TrackerManager
        trt_context=None,
        trt_bindings=None,
        trt_inputs=None,
        trt_outputs=None,
        trt_stream=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_particles = num_particles
        self.noise = noise
        self.patch_size = patch_size
        self.use_deep_features = use_deep_features

        # TensorRT bits
        self.trt_context  = trt_context
        self.trt_bindings = trt_bindings
        self.trt_inputs   = trt_inputs
        self.trt_outputs  = trt_outputs
        self.trt_stream   = trt_stream

        # PF state
        self.particles = torch.tensor(
            initial_pos, dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(num_particles, 1)
        self.target_feature = None

    @torch.no_grad()
    def predict(self):
        noise = torch.randn_like(self.particles) * self.noise
        self.particles += noise

    @torch.no_grad()
    def update(self, frame, target_patch):
        """
        frame: HxWx3 uint8 BGR, full frame
        target_patch: the initial patch used to set self.target_feature
        """
        # 1) On first call, extract and stash the target’s deep feature
        if self.use_deep_features and self.target_feature is None:
            # preprocess target_patch into trt_inputs[0]['host']
            preprocess_for_trt(target_patch, self.trt_inputs[0]['host'])
            # run TRT once
            out_arr = do_inference(
                self.trt_context,
                self.trt_bindings,
                self.trt_inputs,
                self.trt_outputs,
                self.trt_stream
            )[0]
            # convert to torch Tensor on device
            self.target_feature = torch.from_numpy(out_arr).to(self.device)

        # 2) Crop N patches around each particle via ROI-Align
        frame_t = (
            torch.from_numpy(frame).permute(2, 0, 1)
            .unsqueeze(0).float().to(self.device)
            / 255.0
        )
        B, C, H, W = frame_t.shape
        half = self.patch_size // 2
        xs = self.particles[:, 0].clamp(0, W - 1)
        ys = self.particles[:, 1].clamp(0, H - 1)
        x1 = (xs - half).clamp(0, W - 1)
        y1 = (ys - half).clamp(0, H - 1)
        x2 = (xs + half).clamp(0, W - 1)
        y2 = (ys + half).clamp(0, H - 1)
        rois = torch.stack([
            torch.zeros_like(x1), x1, y1, x2, y2
        ], dim=1)

        patches = roi_align(
            input=frame_t,
            boxes=rois,
            output_size=(self.patch_size, self.patch_size),
            spatial_scale=1.0,
            aligned=True
        )

        # 3) Get features either via TRT or via your old extractor
        if self.use_deep_features:
            # copy all N patches into the host buffer at once
            # assume trt_inputs[0]['host'] shape = (N, C, H, W)
            host = self.trt_inputs[0]['host']
            for i, p in enumerate(patches):
                img = (p.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                preprocess_for_trt(img, host[i:i+1])
            trt_outs = do_inference(
                self.trt_context,
                self.trt_bindings,
                self.trt_inputs,
                self.trt_outputs,
                self.trt_stream
            )
            batch_features = torch.from_numpy(trt_outs[0]).to(self.device)
        else:
            # your existing torch inference path:
            batch_features = extract_batch_features([
                (p.cpu().permute(1,2,0).numpy()*255).astype(np.uint8)
                for p in patches
            ]).to(self.device)

        # 4) Compute similarity & resample
        sims = F.cosine_similarity(
            batch_features,
            self.target_feature.unsqueeze(0).expand_as(batch_features),
            dim=1
        )
        weights = 1.0 - sims
        self.resample(weights)

    @torch.no_grad()
    def resample(self, weights):
        lik = torch.exp(-weights)
        probs = lik / lik.sum()
        idx = torch.multinomial(probs, self.num_particles, replacement=True)
        self.particles = self.particles[idx]

    @torch.no_grad()
    def estimate(self):
        m = self.particles.mean(dim=0)
        return int(m[0].item()), int(m[1].item())
