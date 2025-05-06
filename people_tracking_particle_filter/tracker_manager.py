# tracker_manager.py

import torch
import cv2
from blob_detection import detect_blobs
from config import NUM_PARTICLES, MOTION_NOISE, PATCH_SIZE, APPEARANCE_EMA_ALPHA
from deep_feature_extractor import extract_features

class TrackerManager:
    def __init__(
        self,
        num_particles=NUM_PARTICLES,
        noise=MOTION_NOISE,
        patch_size=PATCH_SIZE,
        device=None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_particles = num_particles
        self.noise = noise
        self.patch_size = patch_size

        # Load the fused PF TorchScript once
        self.pf_mod = torch.jit.load("fused_pf.pt").to(self.device).eval()

        # Each tracker: dict with 'particles', 'template', 'last_sim'
        self.trackers = []

    def update(self, frame, detections):
        frame_t = (
            torch.from_numpy(frame)
                 .permute(2,0,1)
                 .unsqueeze(0)
                 .float()
                 .to(self.device)
            / 255.0
        )

        # 1) Add new trackers if needed
        for det in detections[len(self.trackers):]:
            x, y, w, h = det
            center = torch.tensor([[x + w//2, y + h//2]], device=self.device).float()
            # Initialize a cloud around that center
            particles = center.repeat(self.num_particles, 1)
            # Build initial template feature
            patch = frame[y:y+h, x:x+w]
            patch224 = cv2.resize(patch, (224,224), interpolation=cv2.INTER_LINEAR)
            template = extract_features([patch224]).squeeze(0).to(self.device)
            self.trackers.append({
                "particles": particles,
                "template":  template,
                "last_sim":  0.0
            })

        if not self.trackers:
            return

        # 2) Batch all particles & templates across trackers
        all_particles = torch.cat([t["particles"] for t in self.trackers], dim=0)  # [M*N,2]
        all_templates = torch.cat([
            t["template"].unsqueeze(0).repeat(self.num_particles,1)
            for t in self.trackers
        ], dim=0)  # [M*N,C]

        # 3) Single fused PF call
        new_particles, mean_sims = self.pf_mod(frame_t, all_particles, all_templates)
        # new_particles: [M*N,2]
        # mean_sims: scalar if M==1 else [M*N]

        M = len(self.trackers)
        N = self.num_particles  # use manager’s known per-tracker count

        # 4) Split outputs back to per-tracker
        if M == 1:
            self.trackers[0]["particles"] = new_particles
            self.trackers[0]["last_sim"]  = float(mean_sims.item())
        else:
            parts_split = new_particles.view(M, N, 2)
            sims_split  = mean_sims.view(M, N)
            for idx, tracker in enumerate(self.trackers):
                tracker["particles"] = parts_split[idx]
                tracker["last_sim"]  = float(sims_split[idx].mean().item())

        # (Optional) EMA‐update each template here, if desired
        # α = APPEARANCE_EMA_ALPHA
        # for t in self.trackers:
        #     new_feat = ...  # extract from latest best patch
        #     t["template"] = torch.nn.functional.normalize((1-α)*t["template"] + α*new_feat, dim=0)

    def get_estimates(self):
        centers = []
        for t in self.trackers:
            mean_pt = t["particles"].mean(dim=0)
            centers.append((int(mean_pt[0].item()), int(mean_pt[1].item())))
        return centers
