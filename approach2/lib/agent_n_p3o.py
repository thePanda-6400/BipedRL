import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal


class NP3OAgent(nn.Module):
    """N-P3O Agent with separate reward and cost critics"""
    
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        
        # Actor (policy) - outputs mean
        self.actor_mu = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )
        
        # Learnable log std
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        
        # Reward critic (value function)
        self.reward_critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Cost critic (cost value function)
        self.cost_critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # Non-negative output
        )
        
        # Initialize with small weights for symmetry
        self._init_weights(scale=0.01)
    
    def _init_weights(self, scale=0.01):
        """Small weight initialization (critical for symmetry augmentation)"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -scale, scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def get_value(self, obs):
        """Get reward value estimate"""
        return self.reward_critic(obs)
    
    def get_cost_value(self, obs):
        """Get cost value estimate"""
        return self.cost_critic(obs)
    
    def get_action_and_value(self, obs, action=None):
        """
        Get action, log prob, entropy, and both values
        Compatible with PPO interface
        """
        action_mean = self.actor_mu(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
        
        log_prob = probs.log_prob(action).sum(-1)
        entropy = probs.entropy().sum(-1)
        
        reward_value = self.reward_critic(obs)
        cost_value = self.cost_critic(obs)
        
        return action, log_prob, entropy, reward_value, cost_value