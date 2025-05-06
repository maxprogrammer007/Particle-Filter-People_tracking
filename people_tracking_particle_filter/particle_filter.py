# particle_filter.py

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align
import numpy as np
import cv2
from deep_feature_extractor import extract_features

class ParticleFilter:
    def __init__(self, initial_pos, num_particles=50, noise=5.0, patch_size=20, device=None):
        self.device        = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_particles = num_particles
        self.noise         = noise
        self.patch_size    = patch_size

        # Initialize particles around initial_pos
        self.particles = torch.full(
            (num_particles, 2),
            fill_value=0.0,
            dtype=torch.float32,
            device=self.device
        )
        self.particles += torch.tensor(initial_pos, device=self.device)

        # Appearance template feature
        self.target_feature = None
        # Last‐frame deep‐feature similarity
        self.last_deep_sim = 0.0

    @torch.no_grad()
    def predict(self):
        """Propagate particles with Gaussian noise."""
        noise = torch.randn_like(self.particles) * self.noise
        self.particles += noise

    @torch.no_grad()
    def update(self, frame, target_patch, use_deep_features=True):
        """
        1) Initialize or EMA‐update the template feature
        2) ROI‐Align all particles into patches
        3) Resize patches to 224×224 & batch‐extract deep features
        4) Compute & store average cosine similarity
        5) Resample based on similarity weights
        """
        # 1) Template init / EMA update
        if use_deep_features:
            feat = extract_features([target_patch]).squeeze(0)  # [D]
            if self.target_feature is None:
                self.target_feature = feat.clone()
            else:
                from config import APPEARANCE_EMA_ALPHA
                α = APPEARANCE_EMA_ALPHA
                self.target_feature = F.normalize((1-α)*self.target_feature + α*feat, dim=0)

        # 2) Build ROI‐Align input
        ft = torch.from_numpy(frame).permute(2,0,1).unsqueeze(0).float().to(self.device)/255.0
        _,_,H,W = ft.shape
        half = self.patch_size // 2
        c = self.particles.clamp(min=0)
        x1 = (c[:,0]-half).clamp(0,W-1)
        y1 = (c[:,1]-half).clamp(0,H-1)
        x2 = (c[:,0]+half).clamp(0,W-1)
        y2 = (c[:,1]+half).clamp(0,H-1)
        rois = torch.stack([torch.zeros_like(x1), x1, y1, x2, y2], dim=1)

        # 3) Extract patches
        with torch.amp.autocast(device_type="cuda", enabled=True):
            patches = roi_align(
                ft, rois,
                output_size=(self.patch_size, self.patch_size),
                spatial_scale=1.0,
                aligned=True
            )  # [N,3,ps,ps]

        # 4) To numpy, resize, extract features
        np_patches = [(p.cpu().permute(1,2,0).numpy()*255).astype(np.uint8) for p in patches]
        np_patches = [cv2.resize(p, (224,224), interpolation=cv2.INTER_LINEAR) for p in np_patches]
        batch_feats = extract_features(np_patches)  # [N, D]

        # 5) Cosine similarities
        sims = F.cosine_similarity(
            batch_feats,
            self.target_feature.unsqueeze(0).expand_as(batch_feats),
            dim=1
        )
        self.last_deep_sim = float(sims.mean().item())

        # 6) Resample
        weights = torch.exp(-(1.0 - sims))
        probs   = weights / weights.sum()
        idx     = torch.multinomial(probs, self.num_particles, replacement=True)
        self.particles = self.particles[idx]

    @torch.no_grad()
    def estimate(self):
        """Return the mean particle position as (x, y) ints."""
        pos = self.particles.mean(dim=0)
        return int(pos[0].item()), int(pos[1].item())
