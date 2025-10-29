# final/inference_lipm.py
"""
Evaluate trained policy on LIPM baseline
"""

import torch
import numpy as np
import os
from lipm_env import make_lipm_env
from n_p3o import ActorCritic


def evaluate_lipm_policy(policy_path='policy_lipm_best.pt', 
                         num_episodes=5,
                         render=True):
    """Evaluate LIPM + RL corrections"""
    
    print("\n📊 Evaluating LIPM + RL Policy")
    print("="*50)
    
    # Create environment
    env = make_lipm_env(
        render_mode='human' if render else 'rgb_array',
        use_lipm=True,
        correction_scale=0.3
    )
    
    # Load policy
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    policy = ActorCritic(obs_dim, action_dim, hidden_dim=256)
    
    if os.path.exists(policy_path):
        policy.load_state_dict(torch.load(policy_path, map_location='cpu'))
        print(f"✓ Loaded policy from {policy_path}")
    else:
        print(f"⚠️ Policy not found: {policy_path}")
        print("   Using random policy")
    
    policy.eval()
    
    # Evaluate
    episode_rewards = []
    episode_costs = []
    episode_lengths = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        total_cost = 0
        steps = 0
        
        while not done and steps < 1000:
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
            
            if steps % 100 == 0:
                print(f"  Step {steps}: Phase={info['lipm_phase']:.2f}, "
                      f"LIPM steps={info['lipm_step_count']}")
        
        episode_rewards.append(total_reward)
        episode_costs.append(total_cost)
        episode_lengths.append(steps)
        
        print(f"Episode {ep+1}/{num_episodes}: "
              f"R={total_reward:.2f}, C={total_cost:.2f}, Steps={steps}")
    
    env.close()
    
    # Summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Cost: {np.mean(episode_costs):.2f} ± {np.std(episode_costs):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='policy_lipm_best.pt')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--no-render', action='store_true')
    args = parser.parse_args()
    
    evaluate_lipm_policy(
        policy_path=args.policy,
        num_episodes=args.episodes,
        render=not args.no_render
    )
