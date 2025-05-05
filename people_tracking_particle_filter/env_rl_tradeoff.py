# env_rl_tradeoff.py

import numpy as np
from gym import Env, spaces
from evaluation import run_tracking_evaluation
from config import MOTION_NOISE  # your default noise value

class TradeoffEnv(Env):
    """
    Gym environment to tune tracker parameters via RL.
    Actions:
      - num_particles ∈ [30,150]
      - patch_size    ∈ [10,40]
    Observation:
      - normalized [FPS, MOTA]
    """
    def __init__(self, max_frames=50, mota_threshold=0.90):
        super().__init__()
        # Action: two discrete dims
        self.action_space = spaces.MultiDiscrete([121, 31])  
        # Observation: last-normalized fps & mota
        self.observation_space = spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)
        self.max_frames = max_frames
        self.mota_threshold = mota_threshold
        self.state = np.zeros(2, dtype=np.float32)

    def reset(self):
        self.state[:] = 0.0
        return self.state

    def step(self, action):
        # Decode action
        num_particles = int(action[0]) + 30   # maps 0–120 -> 30–150
        patch_size    = int(action[1]) + 10   # maps 0–30  -> 10–40

        # Run evaluation
        mota, idsw, fps = run_tracking_evaluation(
            video_path     = "C:\\Users\\abhin\\OneDrive\\Documents\\GitHub\\Particle-Filter-People_tracking\\people_tracking_particle_filter\\sample_videos\\test_video.mp4",
            num_particles  = num_particles,
            motion_noise   = MOTION_NOISE,
            patch_size     = patch_size,
            max_frames     = self.max_frames
        )

        # Normalize metrics
        norm_fps  = min(fps / 4.0, 1.0)
        norm_mota = min(mota, 1.0)
        reward    = norm_fps * norm_mota

        # Penalize below-threshold accuracy
        if mota < self.mota_threshold:
            reward -= 1.0

        self.state = np.array([norm_fps, norm_mota], dtype=np.float32)
        done = True  # one-step episode
        return self.state, reward, done, {}

    def render(self, mode='human'):
        fps = self.state[0] * 4.0
        mota = self.state[1]
        print(f"FPS={fps:.2f}, MOTA={mota:.3f}")
