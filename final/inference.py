# inference.py
import torch
import numpy as np
import gymnasium as gym
import os
from env import HumanoidVelocityEnv
from n_p3o import ActorCritic

def evaluate(policy_path='policy_best.pt', num_episodes=20, render=True, record_video=True):
    """
    Evaluate trained policy
    
    Args:
        policy_path: Path to saved policy
        num_episodes: Number of episodes to run
        render: Whether to render (display window)
        record_video: Whether to record video
    """
    
    # Create environment with appropriate render mode
    if record_video:
        # For video recording, we need rgb_array mode
        env = HumanoidVelocityEnv(render_mode='rgb_array')
        os.makedirs('videos', exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env, 
            'videos',
            episode_trigger=lambda x: True,  # Record all episodes
            name_prefix='humanoid'
        )
        print("Recording video to ./videos/")
    elif render:
        # For live rendering
        env = HumanoidVelocityEnv(render_mode='human')
        print("Rendering in window")
    else:
        # No rendering
        env = HumanoidVelocityEnv(render_mode=None)
        print("Running without rendering")
    
    # Load policy
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    policy = ActorCritic(obs_dim, action_dim)
    
    if os.path.exists(policy_path):
        policy.load_state_dict(torch.load(policy_path))
        print(f"✓ Loaded policy from {policy_path}")
    else:
        print(f"Policy file not found: {policy_path}")
        print("Using random policy for testing")
    
    policy.eval()
    
    episode_rewards = []
    episode_costs = []
    episode_lengths = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        total_cost = 0
        steps = 0
        
        while not done and steps < 1000:  # Max 1000 steps per episode
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            with torch.no_grad():
                action, _ = policy.act(obs_tensor, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(
                action.squeeze(0).numpy()
            )
            
            total_reward += reward
            total_cost += info.get('cost', 0)
            steps += 1
            done = terminated or truncated
        
        episode_rewards.append(total_reward)
        episode_costs.append(total_cost)
        episode_lengths.append(steps)
        
        print(f"Episode {ep+1}/{num_episodes}: "
              f"Reward={total_reward:.2f}, "
              f"Cost={total_cost:.2f}, "
              f"Steps={steps}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Cost: {np.mean(episode_costs):.2f} ± {np.std(episode_costs):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Min/Max Reward: {np.min(episode_rewards):.2f} / {np.max(episode_rewards):.2f}")
    
    env.close()
    
    if record_video:
        print(f"\n✓ Videos saved to ./videos/")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained humanoid policy')
    parser.add_argument('--policy', type=str, default='policy_best.pt',
                        help='Path to policy checkpoint')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to evaluate')
    parser.add_argument('--no-video', action='store_true',
                        help='Disable video recording')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering entirely')
    
    args = parser.parse_args()
    
    # Determine render settings
    if args.no_render:
        render = False
        record_video = False
    elif args.no_video:
        render = True
        record_video = False
    else:
        render = False  # Don't show window when recording
        record_video = True
    
    evaluate(
        policy_path=args.policy,
        num_episodes=args.episodes,
        render=render,
        record_video=record_video
    )