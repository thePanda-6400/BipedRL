# final/train_lipm.py
"""
Train N-P3O on top of LIPM baseline
Complete solution for the interview task
"""

import torch
import numpy as np
import os
from lipm_env import make_lipm_env
from symmetry import symmetries
from n_p3o import N_P3O
from diagnostic_logger import TrainingLogger


def train_with_lipm():
    """
    Main training function: N-P3O + LIPM + Symmetry
    
    This implements the complete solution:
    1. LIPM provides baseline walking
    2. N-P3O learns corrections with constraints
    3. Symmetry augmentation for left-right invariance
    """
    
    config = {
        # Learning parameters
        'lr': 3e-4,              # Learning rate
        'epochs': 5,              # Epochs per iteration
        'batch_size': 256,        # Minibatch size
        'steps_per_iter': 2048,   # Steps per environment per iteration
        'num_iterations': 500,    # Total training iterations
        
        # N-P3O constraint handling
        'kappa_init': 0.1,        # Initial constraint penalty
        'kappa_max': 5.0,         # Maximum constraint penalty
        'kappa_growth': 1.0002,   # Exponential growth rate
        
        # PPO parameters
        'clip_param': 0.2,        # PPO clipping parameter
        'gamma': 0.99,            # Discount factor
        'lam': 0.95,              # GAE lambda
        
        # Exploration
        'entropy_coef': 0.01,     # Entropy bonus coefficient
        'entropy_decay': 0.9995,  # Entropy decay rate
        
        # Network architecture
        'hidden_dim': 256,        # Hidden layer size
        
        # Stability
        'max_grad_norm': 0.5,     # Gradient clipping
        'target_kl': 0.02,        # KL divergence threshold
        'lr_decay_rate': 0.9999,  # Learning rate decay
    }
    
    # LIPM settings
    use_lipm_baseline = True   # Use LIPM baseline (set False for pure RL comparison)
    correction_scale = 0.3     # Scale of RL corrections (30% of action range)
    
    # Create LIPM-augmented environment
    env = make_lipm_env(
        render_mode=None,
        use_lipm=use_lipm_baseline,
        correction_scale=correction_scale
    )
    
    # Create N-P3O agent with symmetry
    agent = N_P3O(env, symmetries, config)
    
    # Logger for tracking metrics
    logger = TrainingLogger(log_dir='logs_lipm')
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        agent.optimizer,
        gamma=config['lr_decay_rate']
    )
    
    # Print training configuration
    print("\n" + "="*70)
    print("🚀 TRAINING: N-P3O + LIPM + SYMMETRY")
    print("="*70)
    print("📋 Configuration:")
    print(f"   • LIPM baseline: {use_lipm_baseline}")
    print(f"   • Correction scale: {correction_scale}")
    print(f"   • Symmetry augmentation: Enabled")
    print(f"   • Observation dim: {env.observation_space.shape[0]}")
    print(f"   • Action dim: {env.action_space.shape[0]}")
    print(f"   • Learning rate: {config['lr']}")
    print(f"   • Steps per iteration: {config['steps_per_iter']}")
    print("="*70 + "\n")
    
    # Training loop
    best_reward = -np.inf
    patience = 100
    patience_counter = 0
    
    try:
        for iteration in range(config['num_iterations']):
            print(f"\n{'='*70}")
            print(f"Iteration {iteration}/{config['num_iterations']}")
            print(f"{'='*70}")
            
            # Collect rollouts
            print("📊 Collecting rollouts...")
            buffer = agent.collect_rollouts(config['steps_per_iter'])
            
            # Update policy with N-P3O
            print("🔄 Updating policy...")
            stats = agent.update(buffer)
            
            # Log metrics
            logger.log(iteration, stats, buffer)
            
            # Decay learning rate
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Check KL divergence
            if stats['approx_kl'] > config['target_kl'] * 1.5:
                print(f"⚠️  KL divergence high: {stats['approx_kl']:.6f}")
            
            # Compute episode metrics
            episode_rewards = []
            episode_costs = []
            episode_reward = 0
            episode_cost = 0
            
            for i in range(len(buffer['rewards'])):
                episode_reward += buffer['rewards'][i].item()
                episode_cost += buffer['costs'][i].item()
                if buffer['dones'][i]:
                    episode_rewards.append(episode_reward)
                    episode_costs.append(episode_cost)
                    episode_reward = 0
                    episode_cost = 0
            
            mean_reward = np.mean(episode_rewards) if episode_rewards else 0
            mean_cost = np.mean(episode_costs) if episode_costs else 0
            
            # Print metrics
            print(f"\n📈 Metrics:")
            print(f"   Reward: {mean_reward:.2f}")
            print(f"   Cost: {mean_cost:.2f}")
            print(f"   Policy Loss: {stats['policy_loss']:.4f}")
            print(f"   Value Loss: {stats['value_loss']:.4f}")
            print(f"   Kappa: {stats['kappa']:.4f}")
            print(f"   Entropy: {stats['entropy']:.4f}")
            print(f"   KL Divergence: {stats['approx_kl']:.6f}")
            print(f"   Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if mean_reward > best_reward:
                best_reward = mean_reward
                patience_counter = 0
                torch.save(agent.actor_critic.state_dict(), 'policy_lipm_best.pt')
                print(f"\n🏆 New best model! Reward: {best_reward:.2f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n⏹️  No improvement for {patience} iterations. Early stopping.")
                break
            
            # Save checkpoints
            if iteration % 50 == 0 and iteration > 0:
                os.makedirs('checkpoints_lipm', exist_ok=True)
                checkpoint = {
                    'iteration': iteration,
                    'model_state_dict': agent.actor_critic.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'kappa': agent.kappa,
                    'best_reward': best_reward,
                    'config': config,
                    'use_lipm': use_lipm_baseline,
                    'correction_scale': correction_scale,
                }
                torch.save(checkpoint, f'checkpoints_lipm/policy_iter_{iteration}.pt')
                print(f"💾 Checkpoint saved")
            
            # Plot training curves
            if iteration % 20 == 0 and iteration > 0:
                logger.plot()
        
        # Save final model
        torch.save(agent.actor_critic.state_dict(), 'policy_lipm_final.pt')
        logger.plot()
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"   Best reward: {best_reward:.2f}")
        print(f"   Final iteration: {iteration}")
        print(f"   Model saved: policy_lipm_best.pt")
        print("="*70 + "\n")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        torch.save(agent.actor_critic.state_dict(), 'policy_lipm_interrupted.pt')
        print("💾 Saved interrupted model: policy_lipm_interrupted.pt")
    
    finally:
        env.close()


if __name__ == '__main__':
    # Create directories
    os.makedirs('checkpoints_lipm', exist_ok=True)
    os.makedirs('logs_lipm', exist_ok=True)
    
    # Run training
    train_with_lipm()