# n_p3o.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # Actor (policy) - outputs mean of Gaussian
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Reward critic (value function)
        self.reward_critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Cost critics (one per constraint)
        # For simplicity, using one cost critic for all constraints
        self.cost_critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Non-negative output (as per paper appendix)
        )
        
        # Learnable log std
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Initialize with SMALL weights (critical for symmetry!)
        self._init_weights(scale=0.01)
    
    def _init_weights(self, scale=0.01):
        """Small weight initialization for symmetry augmentation"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -scale, scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, obs):
        """Forward pass - returns mean, reward value, cost value"""
        mean = self.actor(obs)
        reward_value = self.reward_critic(obs)
        cost_value = self.cost_critic(obs)
        return mean, reward_value, cost_value
    
    def act(self, obs, deterministic=False):
        """Sample action from policy"""
        mean, _, _ = self(obs)
        if deterministic:
            return mean, None
        
        std = self.log_std.exp()
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob
    
    def evaluate(self, obs, action):
        """Evaluate log prob, values for given state-action pairs"""
        mean, reward_value, cost_value = self(obs)
        std = self.log_std.exp()
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, reward_value, cost_value, entropy


class N_P3O:
    """
    Normalized Penalized Proximal Policy Optimization
    Based on Lee et al. "Evaluation of Constrained RL Algorithms"
    """
    
    def __init__(self, env, symmetries, config):
        self.env = env
        self.symmetries = symmetries
        self.config = config
        
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        self.actor_critic = ActorCritic(obs_dim, action_dim, 
                                       config.get('hidden_dim', 256))
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), 
            lr=config['lr']
        )
        
        # Penalty parameter for constraints (κ in paper)
        # Start small and exponentially increase (as per paper)
        self.kappa = config.get('kappa_init', 0.1)
        self.kappa_max = config.get('kappa_max', 10.0)
        self.kappa_growth = config.get('kappa_growth', 1.0004)
        
        # PPO clipping parameter
        self.clip_param = config.get('clip_param', 0.2)
        
        # Entropy coefficient (decaying as per paper)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.entropy_decay = config.get('entropy_decay', 0.9999)
        
        # GAE parameters
        self.gamma = config.get('gamma', 0.99)
        self.lam = config.get('lam', 0.95)
        
        self.iteration = 0
        
    def collect_rollouts(self, num_steps):
        """Collect experience by running policy in environment"""
        buffer = {
            'obs': [], 'actions': [], 'rewards': [], 'costs': [],
            'log_probs': [], 'reward_values': [], 'cost_values': [], 'dones': []
        }
        
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_cost = 0
        
        for step in range(num_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            with torch.no_grad():
                action, log_prob = self.actor_critic.act(obs_tensor)
                _, reward_value, cost_value = self.actor_critic(obs_tensor)
            
            action_np = action.squeeze(0).numpy()
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated
            
            # Get constraint cost
            cost = info.get('cost', 0.0)
            
            buffer['obs'].append(obs)
            buffer['actions'].append(action_np)
            buffer['rewards'].append(reward)
            buffer['costs'].append(cost)
            buffer['log_probs'].append(log_prob.item())
            buffer['reward_values'].append(reward_value.item())
            buffer['cost_values'].append(cost_value.item())
            buffer['dones'].append(done)
            
            episode_reward += reward
            episode_cost += cost
            obs = next_obs
            
            if done:
                print(f"  Episode: R={episode_reward:.2f}, C={episode_cost:.2f}")
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_cost = 0
        
        return self._process_buffer(buffer)
    
    def _process_buffer(self, buffer):
        """Convert to tensors and compute GAE advantages"""
        # Convert to numpy arrays
        for key in buffer:
            buffer[key] = np.array(buffer[key])
        
        # Compute GAE for rewards
        buffer['reward_advantages'] = self._compute_gae(
            buffer['rewards'], 
            buffer['reward_values'], 
            buffer['dones']
        )
        buffer['reward_returns'] = buffer['reward_advantages'] + buffer['reward_values']
        
        # Compute GAE for costs
        buffer['cost_advantages'] = self._compute_gae(
            buffer['costs'], 
            buffer['cost_values'], 
            buffer['dones']
        )
        
        # Convert to tensors
        for key in buffer:
            buffer[key] = torch.FloatTensor(buffer[key])
        
        return buffer
    
    def _compute_gae(self, rewards, values, dones):
        """Generalized Advantage Estimation"""
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae
        
        return advantages
    
    def augment_buffer(self, buffer):
        """Apply symmetry augmentation (critical step!)"""
        augmented = {key: [buffer[key]] for key in buffer}
        
        # Apply each symmetry transformation
        for sym_name, obs_transform, action_transform in self.symmetries[1:]:
            # Transform observations and actions
            aug_obs = torch.stack([
                torch.FloatTensor(obs_transform(obs.numpy()))
                for obs in buffer['obs']
            ])
            aug_actions = torch.stack([
                torch.FloatTensor(action_transform(act.numpy()))
                for act in buffer['actions']
            ])
            
            augmented['obs'].append(aug_obs)
            augmented['actions'].append(aug_actions)
            
            # CRITICAL: Keep original log probs and values (from paper!)
            augmented['log_probs'].append(buffer['log_probs'])
            augmented['reward_advantages'].append(buffer['reward_advantages'])
            augmented['cost_advantages'].append(buffer['cost_advantages'])
            augmented['reward_values'].append(buffer['reward_values'])
            augmented['cost_values'].append(buffer['cost_values'])
            augmented['reward_returns'].append(buffer['reward_returns'])
            augmented['rewards'].append(buffer['rewards'])
            augmented['costs'].append(buffer['costs'])
            augmented['dones'].append(buffer['dones'])
        
        # Concatenate all augmentations
        for key in augmented:
            augmented[key] = torch.cat(augmented[key], dim=0)
        
        return augmented
    
    def normalize_advantages(self, advantages):
        """Normalize advantages (critical for N-P3O!)"""
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    def update(self, buffer):
        """N-P3O policy update with symmetry augmentation"""
        
        # Apply symmetry augmentation
        buffer = self.augment_buffer(buffer)
        
        # Normalize advantages (CRITICAL for N-P3O!)
        reward_adv_normalized = self.normalize_advantages(buffer['reward_advantages'])
        cost_adv_normalized = self.normalize_advantages(buffer['cost_advantages'])
        
        # Compute mean cost for logging
        mean_cost = buffer['costs'].mean().item()
        
        # Statistics for logging
        stats = {
            'policy_loss': 0,
            'cost_loss': 0,
            'value_loss': 0,
            'cost_value_loss': 0,
            'entropy': 0,
            'kappa': self.kappa,
            'mean_cost': mean_cost,
            'approx_kl': 0
        }
        
        # Multiple epochs of optimization
        for epoch in range(self.config['epochs']):
            # Create minibatches
            indices = torch.randperm(len(buffer['obs']))
            batch_size = self.config.get('batch_size', 256)
            
            for start in range(0, len(indices), batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                
                # Get batch
                obs_batch = buffer['obs'][batch_idx]
                actions_batch = buffer['actions'][batch_idx]
                old_log_probs_batch = buffer['log_probs'][batch_idx]
                reward_adv_batch = reward_adv_normalized[batch_idx]
                cost_adv_batch = cost_adv_normalized[batch_idx]
                returns_batch = buffer['reward_returns'][batch_idx]
                
                # Evaluate current policy
                log_probs, reward_values, cost_values, entropy = \
                    self.actor_critic.evaluate(obs_batch, actions_batch)
                
                # Importance sampling ratio
                ratio = torch.exp(log_probs - old_log_probs_batch)
                
                # Clipped reward objective (Eq. 6 in paper)
                surr1_reward = ratio * reward_adv_batch
                surr2_reward = torch.clamp(ratio, 1 - self.clip_param, 
                                          1 + self.clip_param) * reward_adv_batch
                L_clip_R = torch.min(surr1_reward, surr2_reward).mean()
                
                # Clipped cost objective (Eq. 8 in paper)
                # Using max instead of min for cost (we want to minimize violation)
                surr1_cost = ratio * cost_adv_batch
                surr2_cost = torch.clamp(ratio, 1 - self.clip_param,
                                        1 + self.clip_param) * cost_adv_batch
                L_clip_C = torch.max(surr1_cost, surr2_cost).mean()
                
                # Compute constraint violation term (Eq. 16 in paper)
                # L_VIOL = L_CLIP_C + (1-γ)(J_C(π) - ε) + μ_C/σ_C
                mu_C = buffer['cost_advantages'].mean()
                sigma_C = buffer['cost_advantages'].std() + 1e-8
                constraint_term = ((1 - self.gamma) * mean_cost + mu_C / sigma_C)
                L_viol = L_clip_C + constraint_term
                
                # N-P3O objective (Eq. 7 in paper with normalization)
                # L = L_CLIP_R - κ * max(0, L_VIOL)
                policy_loss = -(L_clip_R - self.kappa * torch.max(
                    torch.tensor(0.0), L_viol
                ))
                
                # Value function losses
                reward_value_loss = F.mse_loss(reward_values.squeeze(), returns_batch)
                cost_value_loss = F.mse_loss(cost_values.squeeze(), 
                                             buffer['cost_values'][batch_idx])
                
                # Total loss
                total_loss = (policy_loss + 
                             0.5 * reward_value_loss + 
                             0.5 * cost_value_loss -
                             self.entropy_coef * entropy.mean())
                
                # Update
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()
                
                # Track statistics
                with torch.no_grad():
                    approx_kl = (old_log_probs_batch - log_probs).mean().item()
                    stats['policy_loss'] += policy_loss.item()
                    stats['cost_loss'] += L_viol.item()
                    stats['value_loss'] += reward_value_loss.item()
                    stats['cost_value_loss'] += cost_value_loss.item()
                    stats['entropy'] += entropy.mean().item()
                    stats['approx_kl'] += approx_kl
        
        # Update kappa (exponential schedule as per paper Section III.D.2)
        self.kappa = min(self.kappa_max, self.kappa * self.kappa_growth)
        
        # Decay entropy coefficient
        self.entropy_coef *= self.entropy_decay
        
        # Average statistics
        num_updates = (len(buffer['obs']) // self.config.get('batch_size', 256)) * \
                      self.config['epochs']
        for key in ['policy_loss', 'cost_loss', 'value_loss', 
                   'cost_value_loss', 'entropy', 'approx_kl']:
            stats[key] /= num_updates
        
        self.iteration += 1
        
        return stats