import numpy as np
import torch
import torch.nn.functional as F
from deep_feature_extractor import extract_deep_feature

class Particle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.weight = 1.0

class ParticleFilter:
    def __init__(self, initial_pos, num_particles=50, noise=5.0, patch_size=20, device=None):
        self.num_particles = num_particles
        self.noise = noise
        self.patch_size = patch_size
        self.device = device
        self.particles = [Particle(initial_pos[0], initial_pos[1]) for _ in range(num_particles)]
        self.target_feature = None

    def predict(self):
        for p in self.particles:
            p.x += np.random.normal(0, self.noise)
            p.y += np.random.normal(0, self.noise)

    def update(self, frame, target_patch, use_deep_features=True):
        frame_h, frame_w = frame.shape[:2]
        half_patch = self.patch_size // 2

        if use_deep_features and self.target_feature is None:
            self.target_feature = extract_deep_feature(target_patch, device=self.device)

        for p in self.particles:
            x = int(np.clip(p.x, 0, frame_w - 1))
            y = int(np.clip(p.y, 0, frame_h - 1))

            patch = frame[max(0, y - half_patch):y + half_patch,
                          max(0, x - half_patch):x + half_patch]

            if patch.size > 0:
                if use_deep_features:
                    particle_feat = extract_deep_feature(patch, device=self.device)
                    sim = F.cosine_similarity(
                        torch.tensor(self.target_feature).unsqueeze(0),
                        torch.tensor(particle_feat).unsqueeze(0)
                    ).item()
                    p.weight = 1.0 - sim
                else:
                    p.weight = 1.0
            else:
                p.weight = 1.0

        self.resample()

    def resample(self):
        weights = np.array([p.weight for p in self.particles])
        weights = np.exp(-weights)
        weights /= np.sum(weights)
        indices = np.random.choice(len(self.particles), len(self.particles), p=weights)
        self.particles = [self.particles[i] for i in indices]

    def estimate(self):
        x = np.mean([p.x for p in self.particles])
        y = np.mean([p.y for p in self.particles])
        return int(x), int(y)
