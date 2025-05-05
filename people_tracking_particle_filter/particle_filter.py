import torch
import torch.nn.functional as F
from torchvision.ops import roi_align
import numpy as np
from deep_feature_extractor import extract_features
from config import APPEARANCE_EMA_ALPHA

class ParticleFilter:
    def __init__(self, initial_pos, num_particles=50, noise=5.0, patch_size=20, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_particles = num_particles
        self.noise = noise
        self.patch_size = patch_size

        self.particles = torch.full(
            (num_particles, 2),
            fill_value=0.0,
            dtype=torch.float32,
            device=self.device
        )
        self.particles += torch.tensor(initial_pos, device=self.device)
        self.target_feature = None

    @torch.no_grad()
    def predict(self):
        noise = torch.randn_like(self.particles) * self.noise
        self.particles += noise

    @torch.no_grad()
    def update(self, frame, target_patch, use_deep_features=True):
        # 1) Update appearance template via EMA
        if use_deep_features:
            f_new = extract_features([target_patch]).squeeze(0)
            if self.target_feature is None:
                self.target_feature = f_new
            else:
                alpha = APPEARANCE_EMA_ALPHA
                self.target_feature = F.normalize(
                    (1 - alpha) * self.target_feature + alpha * f_new,
                    dim=0
                )

        # 2) Prepare frame tensor for ROI-Align
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        frame_h, frame_w = frame_tensor.shape[2:]

        # 3) Compute ROIs from particle centers
        half = self.patch_size // 2
        centers = self.particles.clamp(min=0)
        x1 = (centers[:, 0] - half).clamp(0, frame_w - 1)
        y1 = (centers[:, 1] - half).clamp(0, frame_h - 1)
        x2 = (centers[:, 0] + half).clamp(0, frame_w - 1)
        y2 = (centers[:, 1] + half).clamp(0, frame_h - 1)
        rois = torch.stack([torch.zeros_like(x1), x1, y1, x2, y2], dim=1)

        # 4) ROI-Align to get patches
        with torch.amp.autocast(device_type="cuda", enabled=True):
            patches = roi_align(
                input=frame_tensor,
                boxes=rois,
                output_size=(self.patch_size, self.patch_size),
                spatial_scale=1.0,
                aligned=True
            )

        # 5) Convert for feature extraction
        np_patches = [
            (p.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            for p in patches
        ]

        # 6) Batch extract features
        batch_feats = extract_features(np_patches)

        # 7) Compute cosine similarity
        sims = F.cosine_similarity(
            batch_feats,
            self.target_feature.unsqueeze(0).expand_as(batch_feats),
            dim=1
        )

        # 8) Resample based on similarity
        weights = torch.exp(- (1.0 - sims))
        probs = weights / weights.sum()
        indices = torch.multinomial(probs, self.num_particles, replacement=True)
        self.particles = self.particles[indices]

    @torch.no_grad()
    def estimate(self):
        pos = self.particles.mean(dim=0)
        return int(pos[0].item()), int(pos[1].item())
