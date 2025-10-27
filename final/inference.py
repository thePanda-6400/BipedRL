# inference.py
import torch
import numpy as np
import gymnasium as gym
from environment import HumanoidVelocityEnv
from n_p3o import ActorCritic

def evaluate(policy_path='policy_final.pt', num_episodes=10, render=True):
    env = HumanoidVelocityEnv()
    if render:
        env = gym.wrappers.RecordVideo(env, 'videos', 
                                       episode_trigger=lambda x: True)
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    policy = ActorCritic(obs_dim, action_dim)
    policy.load_state_dict(torch.load(policy_path))
    policy.eval()
    
    episode_rewards = []
    episode_costs = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        total_cost = 0
        steps = 0
        
        while not done:
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
        print(f"Episode {ep+1}: Reward={total_reward:.2f}, "
              f"Cost={total_cost:.2f}, Steps={steps}")
    
    print(f"\nAverage Reward: {np.mean(episode_rewards):.2f} ± "
          f"{np.std(episode_rewards):.2f}")
    print(f"Average Cost: {np.mean(episode_costs):.2f} ± "
          f"{np.std(episode_costs):.2f}")

if __name__ == '__main__':
    evaluate()