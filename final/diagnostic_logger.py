# diagnostic_logger.py
import numpy as np
import matplotlib.pyplot as plt
import os

class TrainingLogger:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.metrics = {
            'iteration': [],
            'mean_reward': [],
            'mean_cost': [],
            'policy_loss': [],
            'value_loss': [],
            'cost_value_loss': [],
            'entropy': [],
            'kappa': [],
            'approx_kl': [],
            'episode_lengths': []
        }
    
    def log(self, iteration, stats, buffer):
        self.metrics['iteration'].append(iteration)
        
        # Compute episode statistics from buffer
        episode_rewards = []
        episode_costs = []
        episode_lengths = []
        
        current_reward = 0
        current_cost = 0
        current_length = 0
        
        for i in range(len(buffer['rewards'])):
            current_reward += buffer['rewards'][i].item()
            current_cost += buffer['costs'][i].item()
            current_length += 1
            
            if buffer['dones'][i]:
                episode_rewards.append(current_reward)
                episode_costs.append(current_cost)
                episode_lengths.append(current_length)
                current_reward = 0
                current_cost = 0
                current_length = 0
        
        self.metrics['mean_reward'].append(np.mean(episode_rewards) if episode_rewards else 0)
        self.metrics['mean_cost'].append(np.mean(episode_costs) if episode_costs else 0)
        self.metrics['episode_lengths'].append(np.mean(episode_lengths) if episode_lengths else 0)
        
        for key in ['policy_loss', 'value_loss', 'cost_value_loss', 'entropy', 'kappa', 'approx_kl']:
            self.metrics[key].append(stats.get(key, 0))
    
    def plot(self):
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        plots = [
            ('mean_reward', 'Mean Episode Reward'),
            ('mean_cost', 'Mean Episode Cost'),
            ('policy_loss', 'Policy Loss'),
            ('value_loss', 'Value Loss'),
            ('entropy', 'Entropy'),
            ('kappa', 'Kappa (Penalty Weight)'),
            ('approx_kl', 'Approx KL Divergence'),
            ('episode_lengths', 'Episode Length'),
        ]
        
        for idx, (key, title) in enumerate(plots):
            ax = axes[idx // 3, idx % 3]
            if self.metrics[key]:
                ax.plot(self.metrics['iteration'], self.metrics[key])
                ax.set_xlabel('Iteration')
                ax.set_ylabel(title)
                ax.set_title(title)
                ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.log_dir}/training_curves.png', dpi=150)
        print(f"✓ Saved plots to {self.log_dir}/training_curves.png")