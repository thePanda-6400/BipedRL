# environment.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class HumanoidVelocityEnv(gym.Env):
    def __init__(self, render_mode=None):
        # Create base environment with render mode
        self.base_env = gym.make('Humanoid-v4', render_mode=render_mode)
        
        # Get unwrapped MuJoCo environment
        env = self.base_env
        while hasattr(env, 'env'):
            env = env.env
        self.mujoco_env = env
        
        # Verify we have MuJoCo data access
        if not hasattr(self.mujoco_env, 'data'):
            print("Warning: Could not access MuJoCo data, using simplified mode")
            self.use_direct_access = False
        else:
            self.use_direct_access = True
            print("✓ MuJoCo data access successful")
        
        # Command space: [vx, vz]
        self.command = np.zeros(2)
        self.prev_action = None
        
        # Observation: base state + command
        obs_dim = self.base_env.observation_space.shape[0] + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,))
        self.action_space = self.base_env.action_space
        
        # Store render mode
        self.render_mode = render_mode
        
    def reset(self, seed=None):
        obs, info = self.base_env.reset(seed=seed)
        self.command = np.array([
            np.random.uniform(0.0, 1.0),
            np.random.uniform(-0.5, 0.5)
        ])
        self.prev_action = np.zeros(self.action_space.shape[0])
        return self._get_obs(obs), info
    
    def _get_obs(self, base_obs):
        return np.concatenate([base_obs, self.command])
    
# env.py (Better reward function)
    def step(self, action):
        obs, _, terminated, truncated, info = self.base_env.step(action)
    
        if self.use_direct_access:
            qpos = self.mujoco_env.data.qpos
            qvel = self.mujoco_env.data.qvel
        
            vx_actual = qvel[0]
            vz_actual = qvel[5]
            height = qpos[2]
        
            torques = self.mujoco_env.data.qfrc_actuator
            torque_violation = np.maximum(0, np.abs(torques) - 100).sum()
        
            joint_vel = qvel[6:]
            vel_violation = np.maximum(0, np.abs(joint_vel) - 8.0).sum()
        
            constraint_cost = torque_violation + vel_violation
        else:
            vx_actual = obs[22] if len(obs) > 22 else 0.0
            vz_actual = obs[27] if len(obs) > 27 else 0.0
            height = obs[0] if len(obs) > 0 else 1.5
            constraint_cost = np.maximum(0, np.abs(action) - 0.8).sum()
    
    # IMPROVED REWARD SHAPING
    # 1. Velocity tracking (main task)
        vx_error = np.abs(vx_actual - self.command[0])
        vz_error = np.abs(vz_actual - self.command[1])
    
    # Use linear penalty for small errors, quadratic for large
        velocity_reward = 2.0 * np.exp(-3.0 * (vx_error + vz_error))
    
    # 2. Alive bonus (strong incentive to stay upright)
        alive_bonus = 2.0 if 1.0 < height < 2.0 else -5.0
    
    # 3. Height tracking
        height_penalty = -0.5 * (height - 1.3) ** 2
    
    # 4. Action regularization (prevent wild movements)
        action_penalty = -0.005 * np.sum(action ** 2)
    
    # 5. Smoothness (prevent jerky movements)
        smoothness_penalty = -0.01 * np.sum((action - self.prev_action) ** 2)
    
    # Combined reward
        reward = (velocity_reward + 
                 alive_bonus + 
                 height_penalty +
                 action_penalty + 
                 smoothness_penalty)
    
        info['cost'] = constraint_cost
        self.prev_action = action.copy()
    
    # Termination
        terminated = height < 0.8 or height > 2.2
    
        return self._get_obs(obs), reward, terminated, truncated, info
    
    def render(self):
        return self.base_env.render()
    
    def close(self):
        return self.base_env.close()