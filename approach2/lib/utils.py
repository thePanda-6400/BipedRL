# lib/utils.py (CORRECTED VERSION)

import argparse
import gymnasium as gym
import numpy as np
import torch
import cv2
from typing import Optional


def parse_args_ppo():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='N-P3O Training')
    
    # Environment
    parser.add_argument('--env', type=str, default='Humanoid-v4',
                       help='Environment ID')
    parser.add_argument('--reward_scale', type=float, default=1.0,
                       help='Reward scaling factor')
    
    # Training
    parser.add_argument('--n_envs', type=int, default=8,
                       help='Number of parallel environments')
    parser.add_argument('--n_steps', type=int, default=2048,
                       help='Steps per environment per update')
    parser.add_argument('--n_epochs', type=int, default=1000,
                       help='Total training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Minibatch size')
    parser.add_argument('--train_iters', type=int, default=5,
                       help='Training iterations per epoch')
    
    # Algorithm
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                       help='GAE lambda')
    parser.add_argument('--clip_ratio', type=float, default=0.2,
                       help='PPO clip ratio')
    parser.add_argument('--vf_coef', type=float, default=0.5,
                       help='Value function coefficient')
    parser.add_argument('--ent_coef', type=float, default=0.01,
                       help='Entropy coefficient')
    parser.add_argument('--target_kl', type=float, default=0.02,
                       help='Target KL divergence for early stopping')
    
    # N-P3O specific
    parser.add_argument('--kappa_init', type=float, default=0.1,
                       help='Initial constraint penalty')
    parser.add_argument('--kappa_max', type=float, default=5.0,
                       help='Maximum constraint penalty')
    parser.add_argument('--use_symmetry', action='store_true', default=True,
                       help='Use symmetry augmentation')
    
    # Logging
    parser.add_argument('--render_epoch', type=int, default=10,
                       help='Render video every N epochs')
    
    # Hardware
    parser.add_argument('--cuda', action='store_true', default=True,
                       help='Use CUDA if available')
    
    args = parser.parse_args()
    return args


def make_env(env_id: str, reward_scaling: float = 1.0, render: bool = False):
    """
    Create environment factory function
    
    THIS RETURNS A CALLABLE (function) that creates the environment!
    
    Args:
        env_id: Gymnasium environment ID
        reward_scaling: Scale rewards by this factor
        render: Whether to render
    
    Returns:
        Callable that creates environment instance when called
    """
    def _thunk():
        """This inner function is what gets called by AsyncVectorEnv"""
        render_mode = 'rgb_array' if render else None
        
        # Try to use custom environment wrapper
        try:
            from env import HumanoidVelocityEnv
            if 'Humanoid' in env_id:
                env = HumanoidVelocityEnv(render_mode=render_mode)
            else:
                env = gym.make(env_id, render_mode=render_mode)
        except ImportError:
            env = gym.make(env_id, render_mode=render_mode)
        
        # Reward scaling wrapper
        if reward_scaling != 1.0:
            env = gym.wrappers.TransformReward(env, lambda r: r * reward_scaling)
        
        return env
    
    return _thunk  # Return the function, don't call it!


def log_video(env, agent, device, save_path: str, max_steps: int = 1000):
    """
    Record a video of the agent
    
    Args:
        env: Gymnasium environment
        agent: Trained agent
        device: torch device
        save_path: Path to save video
        max_steps: Maximum steps to record
    """
    agent.eval()
    
    frames = []
    obs, _ = env.reset()
    done = False
    steps = 0
    
    with torch.no_grad():
        while not done and steps < max_steps:
            # Render frame
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            
            # Get action
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            if hasattr(agent, 'get_action_and_value'):
                action, _, _, _, _ = agent.get_action_and_value(obs_tensor)
            else:
                action = agent.act(obs_tensor, deterministic=True)[0]
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy().squeeze())
            done = terminated or truncated
            steps += 1
    
    # Save video using OpenCV
    if frames:
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, 30, (width, height))
        
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"✓ Video saved: {save_path}")
    
    agent.train()