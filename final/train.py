# train.py
import torch
import numpy as np
from env import HumanoidVelocityEnv
from symmetry import symmetries
from n_p3o import N_P3O

def train():
    config = {
        # Learning
        'lr': 3e-4,
        'epochs': 10,
        'batch_size': 256,
        'steps_per_iter': 2048,
        'num_iterations': 2000,
        
        # N-P3O specific
        'kappa_init': 0.1,  # Start small as per paper
        'kappa_max': 10.0,  # Maximum penalty weight
        'kappa_growth': 1.0004,  # Exponential growth (paper Section III.D.2)
        
        # PPO parameters
        'clip_param': 0.2,
        'gamma': 0.99,
        'lam': 0.95,
        
        # Entropy (decaying)
        'entropy_coef': 0.01,
        'entropy_decay': 0.9999,
        
        # Network
        'hidden_dim': 256,
    }
    
    # Create environment and agent
    env = HumanoidVelocityEnv()
    agent = N_P3O(env, symmetries, config)
    
    print("Starting N-P3O training with symmetry augmentation...")
    print(f"Observation dim: {env.observation_space.shape[0]}")
    print(f"Action dim: {env.action_space.shape[0]}")
    print(f"Symmetries: {len(symmetries)}")
    
    best_reward = -np.inf
    
    for iteration in range(config['num_iterations']):
        print(f"\n=== Iteration {iteration}/{config['num_iterations']} ===")
        
        # Collect rollouts
        buffer = agent.collect_rollouts(config['steps_per_iter'])
        
        # Update policy
        stats = agent.update(buffer)
        
        # Logging
        if iteration % 1 == 0:
            print(f"Policy Loss: {stats['policy_loss']:.4f}")
            print(f"Cost Loss: {stats['cost_loss']:.4f}")
            print(f"Mean Cost: {stats['mean_cost']:.4f}")
            print(f"Kappa: {stats['kappa']:.4f}")
            print(f"Entropy: {stats['entropy']:.4f}")
            print(f"Approx KL: {stats['approx_kl']:.6f}")
        
        # Save checkpoints
        if iteration % 50 == 0:
            checkpoint = {
                'iteration': iteration,
                'model_state_dict': agent.actor_critic.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'kappa': agent.kappa,
                'config': config,
            }
            torch.save(checkpoint, f'checkpoints/policy_iter_{iteration}.pt')
            print(f"Checkpoint saved at iteration {iteration}")
        
        # Save best policy
        episode_rewards = []
        for i in range(len(buffer['rewards'])):
            if buffer['dones'][i]:
                episode_rewards.append(buffer['rewards'][max(0, i-1000):i].sum().item())
        
        if episode_rewards and np.mean(episode_rewards) > best_reward:
            best_reward = np.mean(episode_rewards)
            torch.save(agent.actor_critic.state_dict(), 'policy_best.pt')
            print(f"New best policy! Reward: {best_reward:.2f}")
    
    # Save final policy
    torch.save(agent.actor_critic.state_dict(), 'policy_final.pt')
    print("\nTraining complete!")

if __name__ == '__main__':
    import os
    os.makedirs('checkpoints', exist_ok=True)
    train()