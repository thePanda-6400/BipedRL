import numpy as np


class HumanoidSymmetry:
    """Reflection symmetry for Humanoid"""
    
    def __init__(self):
        # Humanoid-v4 has 17 actuators
        self.action_swap_indices = [
            (3, 7), (4, 8), (5, 9), (6, 10),      # legs
            (11, 14), (12, 15), (13, 16),         # arms
        ]
        self.action_flip_indices = [1]  # abdomen_y
    
    def transform_obs(self, obs):
        """Transform observation for left-right symmetry"""
        obs_mirror = obs.copy()
        # Flip yaw command (last element)
        obs_mirror[:, -1] *= -1
        # Simple mirroring - you can enhance this
        return obs_mirror
    
    def transform_action(self, action):
        """Transform action for left-right symmetry"""
        action_mirror = action.clone()
        
        # Swap left-right actuators
        for left, right in self.action_swap_indices:
            if left < action.shape[-1] and right < action.shape[-1]:
                action_mirror[:, left], action_mirror[:, right] = \
                    action_mirror[:, right].clone(), action_mirror[:, left].clone()
        
        # Flip signs
        for idx in self.action_flip_indices:
            if idx < action.shape[-1]:
                action_mirror[:, idx] *= -1
        
        return action_mirror


def augment_batch_with_symmetry(obs, actions, logprobs, reward_adv, cost_adv, 
                                 returns, old_values, old_cost_values, symmetry):
    """
    Augment batch with symmetry transformations
    Returns augmented tensors (2x size)
    """
    # Original data
    aug_obs = [obs]
    aug_actions = [actions]
    aug_logprobs = [logprobs]  # Keep original!
    aug_reward_adv = [reward_adv]
    aug_cost_adv = [cost_adv]
    aug_returns = [returns]
    aug_old_values = [old_values]
    aug_old_cost_values = [old_cost_values]
    
    # Apply symmetry transformation
    obs_mirror = symmetry.transform_obs(obs.cpu().numpy())
    actions_mirror = symmetry.transform_action(actions)
    
    aug_obs.append(torch.from_numpy(obs_mirror).to(obs.device))
    aug_actions.append(actions_mirror)
    aug_logprobs.append(logprobs)  # CRITICAL: Keep original logprobs!
    aug_reward_adv.append(reward_adv)
    aug_cost_adv.append(cost_adv)
    aug_returns.append(returns)
    aug_old_values.append(old_values)
    aug_old_cost_values.append(old_cost_values)
    
    # Concatenate
    return (
        torch.cat(aug_obs, dim=0),
        torch.cat(aug_actions, dim=0),
        torch.cat(aug_logprobs, dim=0),
        torch.cat(aug_reward_adv, dim=0),
        torch.cat(aug_cost_adv, dim=0),
        torch.cat(aug_returns, dim=0),
        torch.cat(aug_old_values, dim=0),
        torch.cat(aug_old_cost_values, dim=0),
    )


import torch  # Add at top of file