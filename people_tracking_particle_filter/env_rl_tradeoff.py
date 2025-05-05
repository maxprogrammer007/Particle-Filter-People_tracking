import numpy as np
import torch
from gym import Env, spaces
from evaluation import run_tracking_evaluation

class TradeoffEnv(Env):
    """
    Gym environment to tune tracker parameters via RL.
    Actions: [detection_interval, num_particles, patch_size]
    Observation: normalized [FPS, MOTA]
    """
    def __init__(self, max_frames=50, mota_threshold=0.90):
        super().__init__()
        # detection_interval ∈ {1,2,3,4}
        # num_particles ∈ [30,150]
        # patch_size ∈ [10,40]
        self.action_space = spaces.MultiDiscrete([4, 121, 31])
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.max_frames = max_frames
        self.mota_threshold = mota_threshold
        self.state = np.zeros(2, dtype=np.float32)

    def reset(self):
        self.state = np.zeros(2, dtype=np.float32)
        return self.state

    def step(self, action):
        det_interval = int(action[0]) + 1
        num_particles = int(action[1]) + 30
        patch_size = int(action[2]) + 10

        mota, idsw, fps = run_tracking_evaluation(
            video_path="sample_videos/test_video.mp4",
            num_particles=num_particles,
            noise=5.0,
            patch_size=patch_size,
            max_frames=self.max_frames,
            detection_interval=det_interval
        )

        norm_fps = min(fps / 4.0, 1.0)
        norm_mota = min(mota, 1.0)
        reward = norm_fps * norm_mota
        if mota < self.mota_threshold:
            reward -= 1.0

        self.state = np.array([norm_fps, norm_mota], dtype=np.float32)
        done = True  # single-step episode
        return self.state, reward, done, {}

    def render(self, mode='human'):
        print(f"FPS={(self.state[0]*4):.2f}, MOTA={self.state[1]:.3f}")
