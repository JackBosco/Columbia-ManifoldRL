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
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from vae_model import VAE


class LatentPassthroughExtractor(BaseFeaturesExtractor):
    """Pass through the VAE's latent vector unchanged."""
    def __init__(self, observation_space: spaces.Box):
        super().__init__(observation_space, features_dim=observation_space.shape[0])

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations


class CarRacingLatentWrapper(gym.Wrapper):
    """Encodes frames via a pretrained VAE."""
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


def create_env_sac(vae_model: VAE, render=False):
    # Create the environment with optional rendering
    mode = "human" if render else None
    base = gym.make('CarRacing-v2', render_mode=mode)
    base = RecordEpisodeStatistics(base)
    return CarRacingLatentWrapper(base, vae_model)


def test_model(model_path, vae, num_episodes=10, render=False):
    """Test a trained SAC model."""
    print(f"Testing model: {model_path}")
    
    # Create a single (non-vectorized) environment for easier testing
    env = create_env_sac(vae, render=render)
    
    # Load the model - use DummyVecEnv for compatibility but test with a single env
    dummy_env = DummyVecEnv([lambda: create_env_sac(vae, render=False)])
    model = SAC.load(model_path, env=dummy_env)
    print("Model loaded successfully")
    
    # Run the test episodes
    rewards = []
    episode_lengths = []
    tiles_visited = []
    
    for ep in range(num_episodes):
        # Reset for each episode
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        # Track tiles if available
        if hasattr(env.unwrapped, 'tile_visited_count'):
            start_tiles = env.unwrapped.tile_visited_count
        else:
            start_tiles = 0
            
        # Run the episode
        while not done:
            # Get action from model
            action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
            action = action[0]  # Get the action for a single environment
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Check if done
            done = terminated or truncated
            
            # Accumulate stats
            total_reward += reward
            steps += 1
            
            # Safety check (max episode length)
            if steps >= 1000:
                done = True
        
        # Calculate tiles visited
        if hasattr(env.unwrapped, 'tile_visited_count'):
            end_tiles = env.unwrapped.tile_visited_count
            tiles = end_tiles - start_tiles
            tiles_visited.append(tiles)
            tile_info = f", Tiles: {tiles}"
        else:
            tile_info = ""
        
        # Save results
        rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"Episode {ep+1}: Reward = {total_reward:.2f}, Steps = {steps}{tile_info}")
    
    # Print summary statistics
    print("\n===== Test Results =====")
    print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Min/Max Reward: {np.min(rewards):.2f} / {np.max(rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.2f}")
    
    if tiles_visited:
        print(f"Average Tiles Visited: {np.mean(tiles_visited):.2f}")
    
    return rewards, episode_lengths


def plot_training_progress(log_path='./logs/'):
    """Plot the training progress if evaluation logs are available."""
    eval_path = os.path.join(log_path, 'evaluations.npz')
    if not os.path.exists(eval_path):
        print(f"No evaluation logs found at {eval_path}")
        return
    
    try:
        data = np.load(eval_path)
        timestamps = data['timesteps']
        results = data['results']
        
        # Calculate mean and std
        means = results.mean(axis=1)
        stds = results.std(axis=1)
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, means, 'b-', linewidth=2, label='Mean Reward')
        plt.fill_between(timestamps, means - stds, means + stds, alpha=0.3, color='b')
        
        # Add best model line
        best_idx = np.argmax(means)
        best_reward = means[best_idx]
        best_timestep = timestamps[best_idx]
        
        plt.axhline(y=best_reward, color='r', linestyle='--', 
                   label=f'Best: {best_reward:.2f} at {best_timestep/1000:.0f}k steps')
        
        plt.title('SAC Training Progress')
        plt.xlabel('Timesteps')
        plt.ylabel('Reward')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save and show the plot
        plt.savefig('training_progress.png', dpi=300)
        plt.show()
        
    except Exception as e:
        print(f"Error plotting training progress: {e}")


def render_episodes(model_path, vae, num_episodes=3):
    """Render a few episodes with the trained model."""
    print(f"Rendering {num_episodes} episodes with model: {model_path}")
    
    # Test with rendering enabled
    return test_model(model_path, vae, num_episodes=num_episodes, render=True)


def extract_step_number(filename):
    """Safely extract the step number from checkpoint filenames."""
    try:
        # Try to extract step number assuming format like "sac_car_racing_10000_steps.zip"
        parts = filename.split('_')
        for i, part in enumerate(parts):
            if part.isdigit() and i < len(parts) - 1 and parts[i+1] == "steps.zip":
                return int(part)
            if part.isdigit():
                return int(part)
    except (ValueError, IndexError):
        # Return a default value if we can't parse the step number
        return 0
    return 0


if __name__ == "__main__":
    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")
    
    # Load the VAE
    latent_dim = 2
    vae = VAE(latent_dim=latent_dim).to(device)
    vae_weights = 'vae_weights.pth'
    
    if not os.path.exists(vae_weights):
        raise FileNotFoundError(f"VAE weights not found at {vae_weights}")
    
    vae.load_state_dict(torch.load(vae_weights, map_location=device))
    print("VAE loaded successfully")
    
    # Check for available models
    models_to_test = [
        './logs/best_model.zip',  # Best model from training
        'sac_car_racing_vae.zip'  # Final saved model
    ]
    
    # Find checkpoint models
    log_dir = './logs/'
    if os.path.exists(log_dir):
        checkpoints = [f for f in os.listdir(log_dir) 
                      if f.startswith('sac_car_racing_') and f.endswith('.zip')]
        
        # Sort checkpoints using our safe function
        checkpoints.sort(key=extract_step_number)
        
        # Add checkpoint paths
        for ckpt in checkpoints:
            models_to_test.append(os.path.join(log_dir, ckpt))
    
    # Filter to existing models
    available_models = [m for m in models_to_test if os.path.exists(m)]
    
    if not available_models:
        print("No trained models found!")
    else:
        print(f"Found {len(available_models)} models to test:")
        for i, model in enumerate(available_models):
            print(f"{i+1}. {model}")
        
        # Test the best model (or the only one if there's just one)
        best_model = available_models[0]
        print(f"\nTesting model: {best_model}")
        
        # Run tests
        test_model(best_model, vae, num_episodes=10)
        
        # Plot training progress if available
        plot_training_progress()
        
        # Ask if user wants to render episodes
        render = input("\nDo you want to render episodes? (y/n): ").lower().strip() == 'y'
        if render:
            render_episodes(best_model, vae, num_episodes=3)
        
        print("Testing complete!")