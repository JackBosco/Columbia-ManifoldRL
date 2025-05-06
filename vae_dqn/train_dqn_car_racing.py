import os
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import RecordEpisodeStatistics
from torchvision import transforms

import torch
from torch import nn

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from vae_model import VAE


class LatentPassthroughExtractor(BaseFeaturesExtractor):
    """Pass through the VAE’s latent vector unchanged."""
    def __init__(self, observation_space: spaces.Box):
        super().__init__(observation_space, features_dim=observation_space.shape[0])

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations


class CarRacingLatentWrapper(gym.Wrapper):
    """
    Encodes frames via a pretrained VAE and (optionally) applies mild reward shaping.
    Returns a 5-tuple (obs, reward, terminated, truncated, info) as required by Gymnasium.
    """
    def __init__(self, env, vae_model: VAE):
        super().__init__(env)
        self.vae = vae_model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
        ])
        # Latent dim defines new observation_space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(vae_model.latent_dim,), dtype=np.float32
        )
        self.steps_no_progress = 0
        self.tiles_visited = set()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.steps_no_progress = 0
        self.tiles_visited.clear()
        return self._encode(obs), info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        shaped = reward

        # Reward for visiting new track tiles
        if hasattr(self.env.unwrapped, 'tile_visited_count'):
            current = self.env.unwrapped.tile_visited_count
            new_tiles = current - len(self.tiles_visited)
            if new_tiles > 0:
                shaped += new_tiles * 0.2
                self.steps_no_progress = 0
            else:
                self.steps_no_progress += 1
            self.tiles_visited = set(range(current))

        # Small speed bonus
        speed = np.linalg.norm(self.env.unwrapped.car.hull.linearVelocity)
        shaped += 0.005 * speed

        # Soft stall penalty
        if self.steps_no_progress > 100:
            shaped -= 0.05

        return self._encode(obs), shaped, terminated, truncated, info

    def _encode(self, obs: np.ndarray) -> np.ndarray:
        device = next(self.vae.parameters()).device
        img = self.transform(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            mu, _ = self.vae.encode(img)
        return mu.cpu().numpy().flatten()


def create_env_sac(vae_model: VAE):
    # Use RecordEpisodeStatistics to get info["episode"] at episode end
    base = gym.make('CarRacing-v3', render_mode=None)
    base = RecordEpisodeStatistics(base)
    return CarRacingLatentWrapper(base, vae_model)


def train_sac_agent(vae_model: VAE, timesteps: int = 1_000_000):
    # Vectorized and normalized environments
    venv = DummyVecEnv([lambda: create_env_sac(vae_model)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_reward=10.0)

    eval_venv = DummyVecEnv([lambda: create_env_sac(vae_model)])
    eval_venv = VecNormalize(eval_venv, norm_obs=True, norm_reward=False)

    # Evaluation & checkpoint callbacks
    eval_cb = EvalCallback(
        eval_venv,
        best_model_save_path='./logs/',
        log_path='./logs/',
        eval_freq=20_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    ckpt_cb = CheckpointCallback(
        save_freq=100_000,
        save_path='./logs/',
        name_prefix='sac_car_racing'
    )

    # Build SAC with a simple MLP policy on the latent space
    policy_kwargs = dict(
        features_extractor_class=LatentPassthroughExtractor,
        net_arch=[256, 256]
    )
    model = SAC(
        "MlpPolicy",
        venv,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        #tensorboard_log="./tb_sac/",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    model.learn(
        total_timesteps=timesteps,
        callback=[eval_cb, ckpt_cb]
    )
    return model, venv


def test_sac_agent(model: SAC, env, num_episodes: int = 10):
    rewards = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
        rewards.append(total)
        print(f"Test Episode {ep+1}: Reward = {total:.2f}")
    print(f"Average Test Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    return rewards


def plot_training_progress(log_path: str = './logs/'):
    path = os.path.join(log_path, 'evaluations.npz')
    if not os.path.exists(path):
        return
    data = np.load(path)
    ts, results = data['timesteps'], data['results']
    means, stds = results.mean(axis=1), results.std(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(ts, means, label='Mean Reward')
    plt.fill_between(ts, means - stds, means + stds, alpha=0.3)
    plt.xlabel('Timesteps'); plt.ylabel('Reward')
    plt.title('SAC Training Progress'); plt.grid(True)
    plt.savefig('sac_training_progress.png')
    plt.show()


def visualize_policy(vae_model: VAE, sac_model: SAC, resolution: int = 50):
    # Plot continuous action components over a grid in latent space
    xx, yy = np.meshgrid(
        np.linspace(-3, 3, resolution),
        np.linspace(-3, 3, resolution)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    acts = []
    for pt in grid:
        # forward through policy to get mean action
        mu, _ = sac_model.policy.actor(pt.astype(np.float32))
        acts.append(mu.cpu().numpy())
    acts = np.array(acts).reshape(xx.shape + (2,))

    # Steering
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, acts[..., 0], levels=50, alpha=0.8)
    plt.colorbar(label='Steering'); plt.xlabel('Latent Dim 1'); plt.ylabel('Latent Dim 2')
    plt.title('SAC Steering Policy'); plt.savefig('sac_policy_steering.png'); plt.show()

    # Acceleration
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, acts[..., 1], levels=50, alpha=0.8)
    plt.colorbar(label='Acceleration'); plt.xlabel('Latent Dim 1'); plt.ylabel('Latent Dim 2')
    plt.title('SAC Acceleration Policy'); plt.savefig('sac_policy_acceleration.png'); plt.show()


def main():
    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    os.makedirs('./logs/', exist_ok=True)

    # Load pretrained VAE
    latent_dim = 2
    vae = VAE(latent_dim=latent_dim).to(device)
    weights = 'vae_weights.pth'
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Missing VAE weights at '{weights}'")
    vae.load_state_dict(torch.load(weights, map_location=device))

    # Train SAC
    sac_model, env = train_sac_agent(vae, timesteps=1_000_000)
    sac_model.save("sac_car_racing_vae")

    # Test best model
    best = './logs/best_model.zip'
    if os.path.exists(best):
        sac_model = SAC.load(best, env=env)
        print("Loaded best SAC model.")
    test_sac_agent(sac_model, env, num_episodes=10)

    # Plot & visualize
    plot_training_progress()
    visualize_policy(vae, sac_model)
    print("Done.")


if __name__ == "__main__":
    main()
