# particle_filter.py

import torch
import torch.nn.functional as F
import numpy as np
import cv2

from torchvision.ops import roi_align
from tensorrt_utils import do_inference, preprocess_for_trt
from old_utils      import extract_batch_features

class ParticleFilter:
    def __init__(
        self,
        initial_pos,               # (x,y)
        num_particles=50,
        noise=5.0,
        patch_size=20,
        device=None,
        use_deep_features=False,
        # TensorRT bits (if you have an engine)
        trt_context=None,
        trt_bindings=None,
        trt_inputs=None,
        trt_outputs=None,
        trt_stream=None,
    ):
        # device==CUDA if available
        self.device            = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_particles     = num_particles
        self.noise             = noise
        self.patch_size        = patch_size
        self.use_deep_features = use_deep_features

        # TRT pieces
        self.trt_context  = trt_context
        self.trt_bindings = trt_bindings
        self.trt_inputs   = trt_inputs
        self.trt_outputs  = trt_outputs
        self.trt_stream   = trt_stream

        # disable TRT‐path if missing
        if self.use_deep_features and (not trt_context or not trt_inputs):
            self.use_deep_features = False

        # initialize particles on GPU
        init = torch.tensor(initial_pos, dtype=torch.float32, device=self.device)
        self.particles      = init.unsqueeze(0).repeat(num_particles, 1)  # [N,2]
        self.target_feature = None

    @torch.no_grad()
    def predict(self):
        self.particles += torch.randn_like(self.particles) * self.noise

    @torch.no_grad()
    def update(self, frame, blobs, target_patch):
        """
        frame: H×W×3 uint8 BGR np
        blobs: (unused here)
        target_patch: H0×W0×3 uint8 BGR np
        """
        H, W = frame.shape[:2]
        ps   = self.patch_size
        half = ps // 2

        # 1) set target_feature once
        if self.target_feature is None:
            if self.use_deep_features:
                preprocess_for_trt(target_patch, self.trt_inputs[0]['host'])
                out = do_inference(
                    self.trt_context, self.trt_bindings,
                    self.trt_inputs, self.trt_outputs,
                    self.trt_stream
                )[0]
                self.target_feature = torch.from_numpy(out).to(self.device)
            else:
                feat = extract_batch_features([target_patch], device=self.device)
                self.target_feature = feat  # [1, C]

        # 2) extract features for every particle
        if not self.use_deep_features:
            # CPU‐crop + GPU feature‐extractor
            xs = self.particles[:,0].clamp(0, W-1).cpu().numpy().astype(int)
            ys = self.particles[:,1].clamp(0, H-1).cpu().numpy().astype(int)
            crops = []
            for x,y in zip(xs,ys):
                x1, y1 = max(0,x-half), max(0,y-half)
                x2, y2 = min(W,x+half),   min(H,y+half)
                p = frame[y1:y2, x1:x2]
                p = cv2.resize(p, (ps, ps), interpolation=cv2.INTER_LINEAR)
                crops.append(p)
            batch_features = extract_batch_features(crops, device=self.device)  # [N,C]

        else:
            # fast GPU ROI Align → TRT
            frame_t = (torch.from_numpy(frame)[None]
                            .permute(0,3,1,2)
                            .to(self.device).float() / 255.0)  # [1,3,H,W]

            xs = self.particles[:,0].clamp(0, W).to(self.device)
            ys = self.particles[:,1].clamp(0, H).to(self.device)
            x1 = (xs-half).clamp(0, W); y1 = (ys-half).clamp(0, H)
            x2 = (xs+half).clamp(0, W); y2 = (ys+half).clamp(0, H)

            rois = torch.stack([
                torch.zeros(self.num_particles, device=self.device),
                x1, y1, x2, y2
            ], dim=1)

            patches = roi_align(
                frame_t, rois,
                output_size=(ps, ps),
                spatial_scale=1.0, aligned=True
            )  # [N,3,ps,ps]

            # pack into TRT host buffer
            host = self.trt_inputs[0]['host']
            for i,p in enumerate(patches):
                img = (p.cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
                preprocess_for_trt(img, host[i:i+1])
            out = do_inference(
                self.trt_context, self.trt_bindings,
                self.trt_inputs, self.trt_outputs,
                self.trt_stream
            )[0]  # NumPy on host
            batch_features = torch.from_numpy(out).to(self.device)

        # 3) weight = cosine distance
        sims = F.cosine_similarity(
            batch_features,
            self.target_feature.unsqueeze(0),
            dim=1
        )  # [N]
        weights = (1 - sims).clamp(min=0)

        # 4) resample
        self._resample(weights)

    @torch.no_grad()
    def _resample(self, weights: torch.Tensor):
        # Move weights to CPU, sanitize
        w = weights.detach().cpu()
        w = torch.nan_to_num(w, nan=0.0, posinf=1e6, neginf=1e6)

        # Convert “distance” to likelihood
        lik = torch.exp(-w)
        s = lik.sum().item()

        # Build a proper probability distribution
        if s <= 0 or not np.isfinite(s):
            probs = torch.ones_like(lik) / lik.numel()
        else:
            probs = lik / s

        # Clip and re-normalize
        probs = torch.clamp(probs, min=0.0)
        probs = probs / probs.sum()

        # Draw new indices on CPU
        idx_cpu = torch.multinomial(probs, self.num_particles, replacement=True)

        # As a safety guard, ensure every idx is in [0, num_particles)
        if idx_cpu.min() < 0 or idx_cpu.max() >= self.num_particles:
            idx_cpu = torch.randint(0, self.num_particles,
                                    (self.num_particles,),
                                    dtype=torch.long)

        # Move to the same device as your particles
        idx = idx_cpu.to(self.particles.device)

        # Finally, reorder your particles tensor
        self.particles = self.particles[idx]



    @torch.no_grad()
    def estimate(self):
        m = self.particles.mean(dim=0)
        return int(m[0].item()), int(m[1].item())
