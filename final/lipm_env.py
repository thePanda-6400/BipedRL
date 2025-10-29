# final/env_lipm.py
"""
LIPM-augmented environment for N-P3O training
Action space: Small corrections to LIPM baseline
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from env import HumanoidVelocityEnv
from lipm_controller import LIPMWalkingController, LIPMParams


class LIPMHumanoidEnv(gym.Wrapper):
    """
    Environment wrapper that:
    1. Provides LIPM baseline walking pattern
    2. Allows RL to learn small corrections
    3. Extends observations with LIPM state
    
    Benefits:
    - Faster training (starting from working controller)
    - Smaller action space (corrections only)
    - More stable learning
    """
    
    def __init__(self, 
                 base_env: HumanoidVelocityEnv = None,
                 render_mode=None,
                 correction_scale: float = 0.3,
                 use_lipm_baseline: bool = True):
        
        # Create base environment if not provided
        if base_env is None:
            base_env = HumanoidVelocityEnv(render_mode=render_mode)
        
        super().__init__(base_env)
        
        # LIPM controller
        self.lipm = LIPMWalkingController(LIPMParams())
        self.correction_scale = correction_scale
        self.use_lipm_baseline = use_lipm_baseline
        self.dt = 0.015  # Control timestep (66.67 Hz)
        
        # Action space: corrections to LIPM baseline [-1, 1]
        # These get scaled by correction_scale and added to LIPM output
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(17,),
            dtype=np.float32
        )
        
        # Extended observation space:
        # Original obs + LIPM baseline action (17) + LIPM state (2)
        original_obs_dim = base_env.observation_space.shape[0]
        extended_obs_dim = original_obs_dim + 17 + 2
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(extended_obs_dim,),
            dtype=np.float32
        )
        
        self.last_lipm_action = np.zeros(17, dtype=np.float32)
        
        print("\n" + "="*60)
        print("🎯 LIPM-Augmented Humanoid Environment")
        print("="*60)
        print(f"   LIPM baseline enabled: {use_lipm_baseline}")
        print(f"   Correction scale: {correction_scale}")
        print(f"   Original obs dim: {original_obs_dim}")
        print(f"   Extended obs dim: {extended_obs_dim}")
        print(f"   Action space: {self.action_space.shape} (corrections)")
        print("="*60 + "\n")
    
    def reset(self, seed=None, options=None):
        """Reset environment and LIPM controller"""
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Reset LIPM controller
        self.lipm.phase = 0.0
        self.lipm.step_count = 0
        
        # Set LIPM command from environment
        if hasattr(self.env, 'command'):
            self.lipm.set_command(self.env.command[0], self.env.command[1])
        
        # Get initial LIPM action
        self.last_lipm_action = self.lipm.get_baseline_action(self.dt)
        
        # Extend observation with LIPM info
        extended_obs = self._extend_observation(obs)
        
        return extended_obs, info
    
    def _extend_observation(self, base_obs: np.ndarray) -> np.ndarray:
        """
        Extend observation with LIPM information
        
        Extended obs = [base_obs, lipm_action, phase, step_count_normalized]
        """
        lipm_state = self.lipm.get_state_info()
        
        # Normalize values for neural network
        phase_normalized = lipm_state['phase']  # Already [0, 1]
        step_count_normalized = min(lipm_state['step_count'] / 100.0, 1.0)
        
        extended_obs = np.concatenate([
            base_obs,
            self.last_lipm_action,
            [phase_normalized, step_count_normalized]
        ], dtype=np.float32)
        
        return extended_obs
    
    def step(self, correction_action: np.ndarray):
        """
        Execute step with LIPM baseline + learned corrections
        
        Args:
            correction_action: RL policy output (corrections in [-1, 1])
        
        Returns:
            extended_obs, reward, terminated, truncated, info
        """
        # Update LIPM command from environment's current goal
        if hasattr(self.env, 'command'):
            self.lipm.set_command(self.env.command[0], self.env.command[1])
        
        # Get LIPM baseline action
        lipm_action = self.lipm.get_baseline_action(self.dt)
        self.last_lipm_action = lipm_action.copy()
        
        if self.use_lipm_baseline:
            # Combine: LIPM baseline + scaled corrections
            final_action = lipm_action + self.correction_scale * correction_action
            final_action = np.clip(final_action, -1.0, 1.0)
        else:
            # Pure RL mode (for ablation comparison)
            final_action = correction_action
        
        # Execute in base environment
        base_obs, reward, terminated, truncated, info = self.env.step(final_action)
        
        # Extend observation
        extended_obs = self._extend_observation(base_obs)
        
        # Add LIPM info to info dict
        info['lipm_action'] = lipm_action
        info['correction'] = correction_action
        info['final_action'] = final_action
        info['lipm_phase'] = self.lipm.phase
        info['lipm_step_count'] = self.lipm.step_count
        
        # Small penalty for large corrections (encourages staying close to LIPM)
        correction_penalty = -0.001 * np.sum(np.abs(correction_action))
        reward += correction_penalty
        
        return extended_obs, reward, terminated, truncated, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()


def make_lipm_env(render_mode=None, use_lipm=True, correction_scale=0.3):
    """
    Factory function to create LIPM-augmented environment
    
    Args:
        render_mode: 'human', 'rgb_array', or None
        use_lipm: If True, use LIPM baseline; if False, pure RL
        correction_scale: Scale of corrections (0.3 = ±30%)
    
    Returns:
        LIPMHumanoidEnv instance
    """
    base_env = HumanoidVelocityEnv(render_mode=render_mode)
    return LIPMHumanoidEnv(
        base_env=base_env,
        correction_scale=correction_scale,
        use_lipm_baseline=use_lipm
    )