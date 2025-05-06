# fused_particle_filter.py

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align
from deep_feature_extractor import _model as scripted_backbone
from config import PATCH_SIZE, APPEARANCE_EMA_ALPHA, NUM_PARTICLES

class FusedPF(torch.nn.Module):
    def __init__(self, num_particles: int, alpha: float, patch_size: int, noise: float):
        super().__init__()
        self.backbone      = scripted_backbone  # scripted ShuffleNet backbone
        self.num_particles = num_particles
        self.alpha         = alpha
        self.patch_size    = patch_size
        # store noise as tensor
        self.noise_tensor  = torch.tensor(noise)

    @torch.no_grad()
    def forward(self, frame_tensor, particles, template_feat):
        # 1) Add Gaussian noise
        pts = particles + (torch.randn_like(particles) * self.noise_tensor)

        # 2) Build ROIs
        half = self.patch_size // 2
        x1 = (pts[:,0]-half).clamp(0, frame_tensor.size(3)-1)
        y1 = (pts[:,1]-half).clamp(0, frame_tensor.size(2)-1)
        x2 = (pts[:,0]+half).clamp(0, frame_tensor.size(3)-1)
        y2 = (pts[:,1]+half).clamp(0, frame_tensor.size(2)-1)
        rois = torch.stack([torch.zeros_like(x1), x1, y1, x2, y2], dim=1)

        # 3) ROI-Align
        patches = roi_align(
            frame_tensor, rois,
            output_size=(self.patch_size, self.patch_size),
            spatial_scale=1.0,
            aligned=True
        )  # [N,3,ps,ps]

        # 4) Backbone + global avg pool
        fmap   = self.backbone(patches)                               # [N,C,Hf,Wf]
        pooled = F.adaptive_avg_pool2d(fmap, (1,1)).view(fmap.size(0), -1)
        feats  = F.normalize(pooled, p=2, dim=1)                      # [N,C]

        # 5) Cosine similarities & mean
        sims     = torch.sum(feats * template_feat.unsqueeze(0), dim=1)  # [N]
        mean_sim = sims.mean()                                          # scalar

        # 6) Resample particles by similarity
        weights = torch.exp(-(1.0 - sims))
        probs   = weights / weights.sum()
        K = pts.size(0)                # total particles fed in
        idx = torch.multinomial(probs, K, replacement=True)

        new_pts = pts[idx]

        return new_pts, mean_sim

if __name__ == "__main__":
    import os, torch
    from config import NUM_PARTICLES, MOTION_NOISE, PATCH_SIZE, APPEARANCE_EMA_ALPHA

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate fused PF with your default motion noise:
    pf_mod = FusedPF(
        num_particles=NUM_PARTICLES,
        alpha=APPEARANCE_EMA_ALPHA,
        patch_size=PATCH_SIZE,
        noise=MOTION_NOISE         # use MOTION_NOISE from config.py
    ).to(device).eval()

    # Example inputs (adjust H,W to your video resolution)
    H, W = 480, 640
    example_frame     = torch.randn(1, 3, H, W, device=device)
    example_particles = torch.rand(NUM_PARTICLES, 2, device=device) * torch.tensor([W, H], device=device)
    example_template  = torch.randn(1024, device=device)

    # Trace & save
        # 3) Trace the module (disable strict trace checking for nondeterminism)
    traced = torch.jit.trace(
        pf_mod,
        (example_frame, example_particles, example_template),
        check_trace=False
    )

    out_path = "fused_pf.pt"
    traced.save(out_path)
    print(f"[INFO] Traced & saved fused PF to {os.path.abspath(out_path)}")
