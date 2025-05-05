# train_rl.py

from multiprocessing import freeze_support
from stable_baselines3 import PPO
from env_rl_tradeoff import TradeoffEnv

def main():
    freeze_support()  # for Windows multiprocessing
    env = TradeoffEnv(max_frames=50, mota_threshold=0.90)
    model = PPO("MlpPolicy", env, verbose=1, device="cpu")  # use CPU for MLP policy
    model.learn(total_timesteps=10)
    model.save("tradeoff_agent.zip")
    print("[INFO] Saved RL agent to tradeoff_agent.zip")

if __name__ == "__main__":
    main()
