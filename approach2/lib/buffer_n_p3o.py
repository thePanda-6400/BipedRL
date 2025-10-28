import torch
import numpy as np


class NP3OBuffer:
    """
    Buffer for N-P3O that handles both rewards and costs
    """
    
    def __init__(self, obs_dim, act_dim, n_steps, n_envs, device, gamma=0.99, gae_lambda=0.95):
        self.obs_buf = torch.zeros((n_steps, n_envs, *obs_dim), dtype=torch.float32, device=device)
        self.act_buf = torch.zeros((n_steps, n_envs, *act_dim), dtype=torch.float32, device=device)
        self.rew_buf = torch.zeros((n_steps, n_envs), dtype=torch.float32, device=device)
        self.cost_buf = torch.zeros((n_steps, n_envs), dtype=torch.float32, device=device)  # NEW
        self.val_buf = torch.zeros((n_steps, n_envs), dtype=torch.float32, device=device)
        self.cost_val_buf = torch.zeros((n_steps, n_envs), dtype=torch.float32, device=device)  # NEW
        self.logprob_buf = torch.zeros((n_steps, n_envs), dtype=torch.float32, device=device)
        self.terminated_buf = torch.zeros((n_steps, n_envs), dtype=torch.float32, device=device)
        self.truncated_buf = torch.zeros((n_steps, n_envs), dtype=torch.float32, device=device)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.ptr = 0
    
    def store(self, obs, act, rew, cost, val, cost_val, terminated, truncated, logprob):
        """Store a step of experience"""
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.cost_buf[self.ptr] = cost
        self.val_buf[self.ptr] = val
        self.cost_val_buf[self.ptr] = cost_val
        self.terminated_buf[self.ptr] = terminated
        self.truncated_buf[self.ptr] = truncated
        self.logprob_buf[self.ptr] = logprob
        self.ptr = (self.ptr + 1) % self.n_steps
    
    def calculate_advantages(self, next_value, next_cost_value, next_terminated, next_truncated):
        """
        Calculate GAE advantages for both rewards and costs
        Returns: (reward_adv, reward_ret, cost_adv, cost_ret)
        """
        # Reward advantages
        reward_adv = torch.zeros_like(self.rew_buf, device=self.device)
        last_gae_lam = 0
        
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                nextnonterminal = 1.0 - next_terminated
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - self.terminated_buf[t + 1]
                nextvalues = self.val_buf[t + 1]
            
            delta = self.rew_buf[t] + self.gamma * nextvalues * nextnonterminal - self.val_buf[t]
            reward_adv[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * nextnonterminal * last_gae_lam
        
        reward_ret = reward_adv + self.val_buf
        
        # Cost advantages
        cost_adv = torch.zeros_like(self.cost_buf, device=self.device)
        last_gae_lam = 0
        
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                nextnonterminal = 1.0 - next_terminated
                nextvalues = next_cost_value
            else:
                nextnonterminal = 1.0 - self.terminated_buf[t + 1]
                nextvalues = self.cost_val_buf[t + 1]
            
            delta = self.cost_buf[t] + self.gamma * nextvalues * nextnonterminal - self.cost_val_buf[t]
            cost_adv[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * nextnonterminal * last_gae_lam
        
        return reward_adv, reward_ret, cost_adv
    
    def get(self):
        """Get all stored data"""
        return (self.obs_buf, self.act_buf, self.logprob_buf, 
                self.rew_buf, self.cost_buf)
    