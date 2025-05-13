import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

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
    """Create the CarRacing environment with VAE encoding."""
    mode = "human" if render else None
    base = gym.make('CarRacing-v2', render_mode=mode)
    base = RecordEpisodeStatistics(base)
    return CarRacingLatentWrapper(base, vae_model)


def collect_latent_states_and_actions(model, env, n_samples=2000):
    """Collect latent states and corresponding actions from the model."""
    states = []
    actions = []
    action_values = []
    rewards = []
    
    # Reset environment
    obs, _ = env.reset()
    
    # Collect samples
    for i in range(n_samples):
        # Get action from model
        action, _ = model.predict(obs.reshape(1, -1), deterministic=False)
        action = action[0]
        
        # Get Q-values if possible (for coloring)
        with torch.no_grad():
            # Get actor features
            features = torch.tensor(obs.reshape(1, -1), dtype=torch.float32)
            
            # Try to get Q-values if available
            try:
                q_values = model.critic(features)
                if isinstance(q_values, tuple):
                    q_values = q_values[0]  # Get first critic output
                q_value = q_values.mean().item()
            except:
                q_value = 0
        
        # Step environment
        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        # Store data
        states.append(obs)
        actions.append(action)
        action_values.append(q_value)
        rewards.append(reward)
        
        # Update observation
        obs = next_obs
        
        # Reset if done
        if terminated or truncated:
            obs, _ = env.reset()
            
        # Print progress
        if (i+1) % 500 == 0:
            print(f"Collected {i+1}/{n_samples} samples")
    
    return np.array(states), np.array(actions), np.array(action_values), np.array(rewards)


def visualize_with_tsne(states, actions, values, rewards, perplexity=30):
    """Create T-SNE visualizations of states and label with actions."""
    # Apply T-SNE
    print("Computing T-SNE embedding...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=3000, random_state=42)
    states_2d = tsne.fit_transform(states)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot 1: Color by steering action
    sc = axes[0, 0].scatter(states_2d[:, 0], states_2d[:, 1], c=actions[:, 0], cmap='coolwarm', alpha=0.7)
    axes[0, 0].set_title('State Space Colored by Steering Action')
    plt.colorbar(sc, ax=axes[0, 0], label='Steering (left <-> right)')
    
    # Plot 2: Color by acceleration action
    sc = axes[0, 1].scatter(states_2d[:, 0], states_2d[:, 1], c=actions[:, 1], cmap='viridis', alpha=0.7)
    axes[0, 1].set_title('State Space Colored by Acceleration Action')
    plt.colorbar(sc, ax=axes[0, 1], label='Acceleration')
    
    # Plot 3: Color by value
    sc = axes[1, 0].scatter(states_2d[:, 0], states_2d[:, 1], c=values, cmap='plasma', alpha=0.7)
    axes[1, 0].set_title('State Space Colored by Estimated Value')
    plt.colorbar(sc, ax=axes[1, 0], label='Q-Value')
    
    # Plot 4: Color by reward
    sc = axes[1, 1].scatter(states_2d[:, 0], states_2d[:, 1], c=rewards, cmap='RdYlGn', alpha=0.7)
    axes[1, 1].set_title('State Space Colored by Reward')
    plt.colorbar(sc, ax=axes[1, 1], label='Reward')
    
    plt.tight_layout()
    plt.savefig('tsne_visualization.png', dpi=300)
    plt.show()
    
    return states_2d


def cluster_analysis(states_2d, actions, n_clusters=5):
    """Perform cluster analysis on the T-SNE embedding."""
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(states_2d)
    
    # Visualize clusters
    plt.figure(figsize=(10, 8))
    for i in range(n_clusters):
        mask = clusters == i
        plt.scatter(states_2d[mask, 0], states_2d[mask, 1], label=f'Cluster {i+1}', alpha=0.7)
    
    plt.title('Clusters in Latent Space')
    plt.legend()
    plt.savefig('tsne_clusters.png', dpi=300)
    plt.show()
    
    # Analyze average actions per cluster
    print("\nCluster Analysis:")
    for i in range(n_clusters):
        mask = clusters == i
        cluster_actions = actions[mask]
        cluster_size = mask.sum()
        
        # Calculate mean steering and acceleration
        mean_steering = cluster_actions[:, 0].mean()
        mean_accel = cluster_actions[:, 1].mean()
        
        print(f"Cluster {i+1} ({cluster_size} states):")
        print(f"  Average steering: {mean_steering:.3f}")
        print(f"  Average acceleration: {mean_accel:.3f}")
        if cluster_actions.shape[1] > 2:
            mean_brake = cluster_actions[:, 2].mean()
            print(f"  Average brake: {mean_brake:.3f}")
        print()
    
    return clusters


def trajectory_visualization(model, env, states_2d, n_episodes=3, steps_per_episode=1000):
    """Visualize trajectories in latent space."""
    # Collect episode trajectories
    trajectories_raw = []
    
    for ep in range(n_episodes):
        print(f"Collecting trajectory for episode {ep+1}...")
        episode_states = []
        
        obs, _ = env.reset()
        for step in range(steps_per_episode):
            # Store current state
            episode_states.append(obs)
            
            # Get action and step environment
            action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action[0])
            
            if terminated or truncated:
                break
        
        # Save raw states for later embedding
        trajectories_raw.append(np.array(episode_states))
    
    # We need to run TSNE again on all data combined
    print("Running T-SNE on combined data...")
    all_states = np.concatenate([states for states in trajectories_raw])
    
    # Need a much larger perplexity for this smaller dataset
    tsne = TSNE(n_components=2, perplexity=min(30, max(5, len(all_states) // 10)), 
                n_iter=2000, random_state=42)
    
    # Transform all episode states together
    all_states_embedded = tsne.fit_transform(all_states)
    
    # Split embedded states back into episodes
    trajectories = []
    idx = 0
    for states in trajectories_raw:
        n_states = len(states)
        trajectories.append(all_states_embedded[idx:idx + n_states])
        idx += n_states
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Plot background points in gray
    plt.scatter(states_2d[:, 0], states_2d[:, 1], c='lightgray', alpha=0.3, label='Explored States')
    
    # Plot trajectories
    colors = ['red', 'blue', 'green']
    for i, traj in enumerate(trajectories):
        plt.plot(traj[:, 0], traj[:, 1], color=colors[i % len(colors)], 
                 alpha=0.7, linewidth=2, label=f'Episode {i+1}')
        
        # Mark start and end points
        plt.scatter(traj[0, 0], traj[0, 1], color=colors[i % len(colors)], marker='o', s=100, label=f'Start {i+1}')
        plt.scatter(traj[-1, 0], traj[-1, 1], color=colors[i % len(colors)], marker='x', s=100, label=f'End {i+1}')
    
    plt.title('Agent Trajectories in Latent Space')
    plt.legend()
    plt.savefig('trajectory_visualization.png', dpi=300)
    plt.show()


def generate_latent_grid_visualization(model, vae, resolution=20, range_val=3):
    """Visualize how the agent's policy varies across a grid in latent space."""
    # Create a grid of latent vectors
    x = np.linspace(-range_val, range_val, resolution)
    y = np.linspace(-range_val, range_val, resolution)
    xx, yy = np.meshgrid(x, y)
    
    # Flatten the grid
    latent_grid = np.column_stack((xx.flatten(), yy.flatten()))
    
    # Get actions for each point
    actions = []
    values = []
    
    for latent in latent_grid:
        # Reshape for model
        obs = latent.reshape(1, -1)
        
        # Get action
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action[0])
        
        # Try to get Q-value
        try:
            with torch.no_grad():
                features = torch.tensor(obs, dtype=torch.float32)
                q_values = model.critic(features)
                if isinstance(q_values, tuple):
                    q_values = q_values[0]
                value = q_values.mean().item()
                values.append(value)
        except:
            values.append(0)
    
    actions = np.array(actions)
    values = np.array(values)
    
    # Create visualizations
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Steering
    steering = actions[:, 0].reshape(resolution, resolution)
    im = axes[0].imshow(steering, cmap='coolwarm', origin='lower', 
                      extent=[-range_val, range_val, -range_val, range_val])
    axes[0].set_title('Steering Policy')
    axes[0].set_xlabel('Latent Dim 1')
    axes[0].set_ylabel('Latent Dim 2')
    plt.colorbar(im, ax=axes[0])
    
    # Acceleration
    acceleration = actions[:, 1].reshape(resolution, resolution)
    im = axes[1].imshow(acceleration, cmap='viridis', origin='lower',
                     extent=[-range_val, range_val, -range_val, range_val])
    axes[1].set_title('Acceleration Policy')
    axes[1].set_xlabel('Latent Dim 1')
    axes[1].set_ylabel('Latent Dim 2')
    plt.colorbar(im, ax=axes[1])
    
    # Q-values
    values_grid = values.reshape(resolution, resolution)
    im = axes[2].imshow(values_grid, cmap='plasma', origin='lower',
                      extent=[-range_val, range_val, -range_val, range_val])
    axes[2].set_title('Estimated Q-Values')
    axes[2].set_xlabel('Latent Dim 1')
    axes[2].set_ylabel('Latent Dim 2')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('latent_grid_policy.png', dpi=300)
    plt.show()
    
    # Create vector field visualization of policy
    plt.figure(figsize=(10, 8))
    plt.streamplot(xx, yy, 
                  steering.T,
                  acceleration.T,
                  color=values.reshape(resolution, resolution).T,
                  cmap='plasma',
                  density=1.5,
                  linewidth=1.5)
    plt.colorbar(label='Q-Value')
    plt.title('Policy Vector Field')
    plt.xlabel('Latent Dim 1')
    plt.ylabel('Latent Dim 2')
    plt.savefig('policy_vector_field.png', dpi=300)
    plt.show()


def main():
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
    
    # Load model
    model_path = './logs/best_model.zip'
    env = create_env_sac(vae)
    
    # Create dummy vec env for model loading
    dummy_env = DummyVecEnv([lambda: create_env_sac(vae)])
    model = SAC.load(model_path, env=dummy_env)
    print(f"Model loaded successfully from {model_path}")
    
    # Collect data
    print("Collecting data for visualization...")
    states, actions, values, rewards = collect_latent_states_and_actions(model, env, n_samples=5000)
    
    # Visualize with T-SNE
    print("Visualizing with T-SNE...")
    states_2d = visualize_with_tsne(states, actions, values, rewards)
    
    # Perform cluster analysis
    print("Performing cluster analysis...")
    clusters = cluster_analysis(states_2d, actions, n_clusters=5)
    
    # Visualize trajectories - fixed version
    print("Visualizing trajectories...")
    trajectory_visualization(model, env, states_2d, n_episodes=3)
    
    # Generate latent grid visualization
    print("Generating latent grid visualization...")
    generate_latent_grid_visualization(model, vae, resolution=20, range_val=3)
    
    print("Visualization complete!")


if __name__ == "__main__":
    main()