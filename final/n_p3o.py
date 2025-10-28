# n_p3o.py (improved version)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=512):
        super().__init__()

        # Actor: outputs mean of Gaussian
        self.actor_mu = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # ensures actions in [-1,1]
        )

        # Learnable diagonal covariance (log σ)
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

        # Reward critic (value function)
        self.reward_critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Cost critic (non-negative output)
        self.cost_critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

        # Small weight init (symmetry friendly)
        self._init_weights(scale=0.01)

    def _init_weights(self, scale=0.01):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -scale, scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs):
        mu = self.actor_mu(obs)
        std = torch.exp(self.actor_logstd).expand_as(mu)
        return mu, std

    def get_values(self, obs):
        return self.reward_critic(obs), self.cost_critic(obs)

    def act(self, obs, deterministic=False):
        mu, std = self.forward(obs)
        dist = Normal(mu, std)
        if deterministic:
            action = mu
        else:
            action = dist.rsample()  # reparameterization trick
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob

    def evaluate(self, obs, action):
        mu, std = self.forward(obs)
        dist = Normal(mu, std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().mean(-1)
        reward_value, cost_value = self.get_values(obs)
        return log_prob, reward_value, cost_value, entropy


class N_P3O:
    """
    Normalized Penalized PPO (N-P3O)
    """

    def __init__(self, env, symmetries, config):
        self.env = env
        self.symmetries = symmetries
        self.config = config

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.actor_critic = ActorCritic(obs_dim, action_dim, config.get("hidden_dim", 512))
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=config["lr"])

        # Parameters for constraint penalty κ
        self.kappa = config.get("kappa_init", 0.1)
        self.kappa_max = config.get("kappa_max", 10.0)
        self.kappa_growth = config.get("kappa_growth", 1.0004)

        # PPO hyperparameters
        self.clip_param = config.get("clip_param", 0.2)
        self.gamma = config.get("gamma", 0.99)
        self.lam = config.get("lam", 0.95)

        # Entropy (for exploration)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.entropy_decay = config.get("entropy_decay", 0.9999)

        self.iteration = 0

    def collect_rollouts(self, num_steps):
        buffer = {
            "obs": [], "actions": [], "rewards": [], "costs": [],
            "log_probs": [], "reward_values": [], "cost_values": [], "dones": []
        }

        obs, _ = self.env.reset()
        episode_reward, episode_cost = 0, 0

        for step in range(num_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

            with torch.no_grad():
                action, log_prob = self.actor_critic.act(obs_tensor)
                reward_value, cost_value = self.actor_critic.get_values(obs_tensor)

            next_obs, reward, terminated, truncated, info = self.env.step(action.squeeze(0).numpy())
            done = terminated or truncated
            cost = info.get("cost", 0.0)

            for k, v in zip(
                ["obs", "actions", "rewards", "costs", "log_probs", "reward_values", "cost_values", "dones"],
                [obs, action.squeeze(0).numpy(), reward, cost, log_prob.item(),
                 reward_value.item(), cost_value.item(), done],
            ):
                buffer[k].append(v)

            episode_reward += reward
            episode_cost += cost
            obs = next_obs

            if done:
                print(f"  Episode done: R={episode_reward:.2f}, C={episode_cost:.2f}")
                obs, _ = self.env.reset()
                episode_reward, episode_cost = 0, 0

        return self._process_buffer(buffer)

    def _process_buffer(self, buffer):
        for key in buffer:
            buffer[key] = np.array(buffer[key])

        buffer["reward_advantages"] = self._compute_gae(
            buffer["rewards"], buffer["reward_values"], buffer["dones"])
        buffer["reward_returns"] = buffer["reward_advantages"] + buffer["reward_values"]

        buffer["cost_advantages"] = self._compute_gae(
            buffer["costs"], buffer["cost_values"], buffer["dones"])

        for key in buffer:
            buffer[key] = torch.FloatTensor(buffer[key])

        return buffer

    def _compute_gae(self, rewards, values, dones):
        advantages = np.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t < len(rewards) - 1 else 0
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae
        return advantages

    def augment_buffer(self, buffer):
        augmented = {key: [buffer[key]] for key in buffer}
        for sym_name, obs_transform, action_transform in self.symmetries[1:]:
            aug_obs = torch.stack([torch.FloatTensor(obs_transform(o.numpy())) for o in buffer["obs"]])
            aug_act = torch.stack([torch.FloatTensor(action_transform(a.numpy())) for a in buffer["actions"]])
            augmented["obs"].append(aug_obs)
            augmented["actions"].append(aug_act)
            for key in ["log_probs", "reward_advantages", "cost_advantages",
                        "reward_values", "cost_values", "reward_returns",
                        "rewards", "costs", "dones"]:
                augmented[key].append(buffer[key])
        for key in augmented:
            augmented[key] = torch.cat(augmented[key], dim=0)
        return augmented

    def normalize_advantages(self, advantages):
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def update(self, buffer):
        buffer = self.augment_buffer(buffer)

        reward_adv = self.normalize_advantages(buffer["reward_advantages"])
        cost_adv = self.normalize_advantages(buffer["cost_advantages"])
        mean_cost = buffer["costs"].mean().item()

        stats = {k: 0 for k in [
            "policy_loss", "cost_loss", "value_loss", "cost_value_loss",
            "entropy", "approx_kl", "clip_fraction"
        ]}
        stats["kappa"] = self.kappa
        stats["mean_cost"] = mean_cost

        num_updates = 0
        for epoch in range(self.config["epochs"]):
            indices = torch.randperm(len(buffer["obs"]))
            batch_size = self.config.get("batch_size", 256)

            for start in range(0, len(indices), batch_size):
                end = start + batch_size
                idx = indices[start:end]

                obs_b = buffer["obs"][idx]
                act_b = buffer["actions"][idx]
                old_log_b = buffer["log_probs"][idx]
                rew_adv_b = reward_adv[idx]
                cost_adv_b = cost_adv[idx]
                returns_b = buffer["reward_returns"][idx]
                old_cost_b = buffer["cost_values"][idx]

                log_prob, reward_value, cost_value, entropy = self.actor_critic.evaluate(obs_b, act_b)
                ratio = torch.exp(log_prob - old_log_b)

                clip_frac = ((ratio > 1 + self.clip_param) | (ratio < 1 - self.clip_param)).float().mean()

                surr_r1 = ratio * rew_adv_b
                surr_r2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * rew_adv_b
                L_clip_R = torch.min(surr_r1, surr_r2).mean()

                surr_c1 = ratio * cost_adv_b
                surr_c2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * cost_adv_b
                L_clip_C = torch.max(surr_c1, surr_c2).mean()

                mu_C = buffer["cost_advantages"].mean()
                sigma_C = buffer["cost_advantages"].std() + 1e-8
                constraint_term = ((1 - self.gamma) * mean_cost + mu_C / sigma_C)
                L_viol = L_clip_C + constraint_term

                policy_loss = -(L_clip_R - self.kappa * torch.relu(L_viol))

                reward_value_loss = F.mse_loss(reward_value.squeeze(), returns_b)
                cost_value_loss = F.mse_loss(cost_value.squeeze(), old_cost_b)

                total_loss = (policy_loss + 0.5 * reward_value_loss +
                              0.5 * cost_value_loss - self.entropy_coef * entropy.mean())

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_log_b - log_prob).mean().item()
                    for key, val in zip(
                        ["policy_loss", "cost_loss", "value_loss",
                         "cost_value_loss", "entropy", "approx_kl", "clip_fraction"],
                        [policy_loss.item(), L_viol.item(), reward_value_loss.item(),
                         cost_value_loss.item(), entropy.mean().item(), approx_kl, clip_frac.item()]
                    ):
                        stats[key] += val
                num_updates += 1

        self.kappa = min(self.kappa_max, self.kappa * self.kappa_growth)
        self.entropy_coef *= self.entropy_decay

        for key in ["policy_loss", "cost_loss", "value_loss", "cost_value_loss",
                    "entropy", "approx_kl", "clip_fraction"]:
            stats[key] /= max(num_updates, 1)

        self.iteration += 1
        return stats
