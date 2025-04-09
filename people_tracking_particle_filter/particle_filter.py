import numpy as np
import random
import cv2

class Particle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.weight = 1.0

class ParticleFilter:
    def __init__(self, initial_pos, num_particles=50):
        self.num_particles = num_particles
        self.particles = [Particle(initial_pos[0], initial_pos[1]) for _ in range(num_particles)]

    def predict(self, move_std=5):
        for p in self.particles:
            p.x += np.random.normal(0, move_std)
            p.y += np.random.normal(0, move_std)

    def update(self, frame, target_histogram, histogram_func):
        frame_h, frame_w = frame.shape[:2]
        for p in self.particles:
            x = int(min(max(p.x, 0), frame_w-1))
            y = int(min(max(p.y, 0), frame_h-1))
            patch = frame[max(0,y-10):y+10, max(0,x-10):x+10]
            if patch.size > 0:
                particle_hist = histogram_func(patch)
                p.weight = cv2.compareHist(target_histogram, particle_hist, cv2.HISTCMP_BHATTACHARYYA)
            else:
                p.weight = 1.0

        self.resample()

    def resample(self):
        weights = np.array([p.weight for p in self.particles])
        weights = np.exp(-weights)  # Lower distance = higher weight
        weights /= np.sum(weights)

        indices = np.random.choice(len(self.particles), len(self.particles), p=weights)
        self.particles = [self.particles[i] for i in indices]

    def estimate(self):
        x = np.mean([p.x for p in self.particles])
        y = np.mean([p.y for p in self.particles])
        return int(x), int(y)
