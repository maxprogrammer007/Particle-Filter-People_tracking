# env_rl_tradeoff.py
import gym
from gym import spaces
import numpy as np
import torch
from evaluation import run_tracking_evaluation

class TradeoffEnv(gym.Env):
    """
    Gym environment to tune tracker parameters (detection_interval, num_particles, patch_size)
    with reward combining FPS and MOTA.
    """
    def __init__(self, max_frames=50, mota_threshold=0.90):
        super().__init__()
        # Action space: detection_interval (1-4), num_particles (30-150), patch_size (10-40)
        self.action_space = spaces.MultiDiscrete([4, 121, 31])
        # Observation: last fps and last mota normalized
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.max_frames = max_frames
        self.mota_threshold = mota_threshold
        self.state = np.array([0.0, 0.0], dtype=np.float32)

    def reset(self):
        # initial state dummy
        self.state = np.array([0.0, 0.0], dtype=np.float32)
        return self.state

    def step(self, action):
        # decode action
        det_interval = int(action[0]) + 1             # maps 0-3 -> 1-4
        num_particles = int(action[1]) + 30          # 0-120 -> 30-150
        patch_size = int(action[2]) + 10             # 0-30 -> 10-40

        # run evaluation
        mota, idsw, fps = run_tracking_evaluation(
            "sample_video.mp4",
            num_particles, noise=5.0,
            patch_size=patch_size,
            max_frames=self.max_frames,
            detection_interval=det_interval
        )
        # normalize metrics
        norm_fps = min(fps / 4.0, 1.0)
        norm_mota = min(mota, 1.0)
        # reward = weighted sum
        reward = norm_fps * norm_mota
        # enforce mota threshold
        if mota < self.mota_threshold:
            reward -= 1.0

        self.state = np.array([norm_fps, norm_mota], dtype=np.float32)
        done = True  # one-step episodes
        return self.state, reward, done, {}

    def render(self, mode='human'):
        print(f"State: FPS={self.state[0]*4:.2f}, MOTA={self.state[1]:.3f}")

# Training script stub (train_rl.py)
from stable_baselines3 import PPO
from env_rl_tradeoff import TradeoffEnv

def main():
    env = TradeoffEnv(max_frames=50)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000)
    model.save("tradeoff_agent.zip")

if __name__ == "__main__":
    main()

# Usage:
# 1. Adjust run_tracking_evaluation to accept detection_interval.
# 2. Install: pip install gym stable-baselines3
# 3. python train_rl.py
