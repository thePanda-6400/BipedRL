# train.py (Updated with stability fixes)
import torch
import numpy as np
from env import HumanoidVelocityEnv
from symmetry import symmetries
from n_p3o import N_P3O
from diagnostic_logger import TrainingLogger

def train():
    config = {
        # Learning - REDUCED LR
        'lr': 2.5e-4,  # Reduced from 3e-4
        'epochs': 5,  # Reduced from 10
        'batch_size': 256,
        'steps_per_iter': 2048,
        'num_iterations': 500,
        
        # N-P3O specific
        'kappa_init': 0.1,
        'kappa_max': 5.0,  # Reduced from 10.0
        'kappa_growth': 1.0002,  # Slower growth from 1.0004
        
        # PPO parameters
        'clip_param': 0.2,
        'gamma': 0.99,
        'lam': 0.95,
        
        # Entropy (decaying)
        'entropy_coef': 0.01,
        'entropy_decay': 0.9995,  # Slower decay
        
        # Network
        'hidden_dim': 256,
        
        # Stability features
        'max_grad_norm': 0.5,
        'target_kl': 0.02,  # KL divergence threshold
        'lr_decay_rate': 0.9999,  # LR decay
    }
    
    env = HumanoidVelocityEnv()
    agent = N_P3O(env, symmetries, config)
    logger = TrainingLogger()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        agent.optimizer, 
        gamma=config['lr_decay_rate']
    )
    
    print("Starting N-P3O training with stability improvements...")
    
    best_reward = -np.inf
    patience = 100  # Early stopping patience
    patience_counter = 0
    
    for iteration in range(config['num_iterations']):
        print(f"\n=== Iteration {iteration}/{config['num_iterations']} ===")
        
        # Collect rollouts
        buffer = agent.collect_rollouts(config['steps_per_iter'])
        
        # Update policy
        stats = agent.update(buffer)
        
        # Log metrics
        logger.log(iteration, stats, buffer)
        
        # Decay learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Check for KL divergence explosion
        if stats['approx_kl'] > config['target_kl'] * 1.5:
            print(f"KL divergence too high: {stats['approx_kl']:.6f}")
            print("Consider reducing learning rate or clip parameter")
        
        # Compute mean episode reward
        episode_rewards = []
        episode_reward = 0
        for i in range(len(buffer['rewards'])):
            episode_reward += buffer['rewards'][i].item()
            if buffer['dones'][i]:
                episode_rewards.append(episode_reward)
                episode_reward = 0
        
        mean_reward = np.mean(episode_rewards) if episode_rewards else 0
        
        # Logging
        if iteration % 1 == 0:
            print(f"Mean Reward: {mean_reward:.2f}")
            print(f"Policy Loss: {stats['policy_loss']:.4f}")
            print(f"Mean Cost: {stats['mean_cost']:.4f}")
            print(f"Kappa: {stats['kappa']:.4f}")
            print(f"Entropy: {stats['entropy']:.4f}")
            print(f"KL: {stats['approx_kl']:.6f}")
            print(f"LR: {current_lr:.6f}")
        
        # Early stopping and checkpointing
        if mean_reward > best_reward:
            best_reward = mean_reward
            patience_counter = 0
            torch.save(agent.actor_critic.state_dict(), 'policy_best.pt')
            print(f"New best policy! Reward: {best_reward:.2f}")
        else:
            patience_counter += 1
        

        
        # Regular checkpoints
        if iteration % 50 == 0:
            checkpoint = {
                'iteration': iteration,
                'model_state_dict': agent.actor_critic.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'kappa': agent.kappa,
                'best_reward': best_reward,
                'config': config,
            }
            torch.save(checkpoint, f'checkpoints/policy_iter_{iteration}.pt')
        
        # Plot progress
        if iteration % 20 == 0 and iteration > 0:
            logger.plot()
    
    # Save final policy
    torch.save(agent.actor_critic.state_dict(), 'policy_final.pt')
    logger.plot()
    print("\n✓ Training complete!")

if __name__ == '__main__':
    import os
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    train()