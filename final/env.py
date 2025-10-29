# environment.py (IMPROVED VERSION)
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class HumanoidVelocityEnv(gym.Env):
    def __init__(self, render_mode=None):
        self.base_env = gym.make('Humanoid-v4', render_mode=render_mode)
        
        # Get unwrapped mujoco env
        env = self.base_env
        while hasattr(env, "env"):
            env = env.env
        self.mujoco_env = env
        self.use_direct_access = hasattr(self.mujoco_env, "data")
        
        if self.use_direct_access:
            print("✓ MuJoCo direct access enabled.")
        else:
            print("⚠ Running in limited observation mode.")
        
        # Command: [forward vx, yaw rate vz]
        self.command = np.zeros(2)
        self.prev_action = np.zeros(self.base_env.action_space.shape[0])
        
        # Track progress
        self.initial_x = 0.0
        self.total_distance = 0.0
        self.episode_steps = 0
        
        # Extended observation
        obs_dim = self.base_env.observation_space.shape[0] + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = self.base_env.action_space
        self.render_mode = render_mode
    
    def reset(self, seed=None, options=None):
        obs, info = self.base_env.reset(seed=seed)
        
        # Sample commands
        self.command = np.array([
            np.random.uniform(0.3, 1.0),    # INCREASED minimum: force movement
            np.random.uniform(-0.5, 0.5)
        ])
        
        self.prev_action = np.zeros(self.action_space.shape[0])
        
        # Reset tracking
        if self.use_direct_access:
            self.initial_x = self.mujoco_env.data.qpos[0]
        else:
            self.initial_x = 0.0
        self.total_distance = 0.0
        self.episode_steps = 0
        
        return self._get_obs(obs), info
    
    def _get_obs(self, base_obs):
        return np.concatenate([base_obs, self.command])
    
    def step(self, action):
        obs, _, terminated, truncated, info = self.base_env.step(action)
        self.episode_steps += 1
        
        if self.use_direct_access:
            qpos = self.mujoco_env.data.qpos
            qvel = self.mujoco_env.data.qvel
            
            # Current state
            x_pos = qpos[0]
            vx_actual = qvel[0]
            vz_actual = qvel[5]
            height = qpos[2]
            
            # Track distance traveled
            self.total_distance = abs(x_pos - self.initial_x)
            
            # Constraints
            torques = np.abs(self.mujoco_env.data.qfrc_actuator)
            torque_violation = np.sum(np.maximum(0, torques - 100))
            joint_vel = np.abs(qvel[6:])
            vel_violation = np.sum(np.maximum(0, joint_vel - 8.0))
            constraint_cost = torque_violation + vel_violation
            
        else:
            vx_actual = obs[22] if len(obs) > 22 else 0.0
            vz_actual = obs[27] if len(obs) > 27 else 0.0
            height = obs[0] if len(obs) > 0 else 1.5
            self.total_distance += abs(vx_actual) * 0.015  # Approximate
            constraint_cost = np.maximum(0, np.abs(action) - 0.8).sum()
        
        # =========================================
        # IMPROVED REWARD DESIGN FOR WALKING
        # =========================================
        
        # 1. VELOCITY TRACKING (stronger, more precise)
        vx_err = abs(self.command[0] - vx_actual)
        vz_err = abs(self.command[1] - vz_actual)
        
        # Use quadratic penalty for small errors, linear for large
        if vx_err < 0.1:
            vel_tracking = 2.0 * (1.0 - 10 * vx_err)  # Tight tracking
        else:
            vel_tracking = 2.0 * np.exp(-2.0 * vx_err)
        
        yaw_tracking = 0.5 * np.exp(-2.0 * vz_err)
        velocity_reward = vel_tracking + yaw_tracking
        
        # 2. FORWARD PROGRESS REWARD (critical for walking!)
        # Reward distance covered, scaled by command
        progress_reward = 1.0 * vx_actual * (vx_actual > 0.1)  # Only reward forward motion
        
        # 3. ALIVE BONUS (reduced from 5.0 to 1.0)
        alive_bonus = 1.0 if 1.0 < height < 2.0 else -5.0
        
        # 4. ENERGY EFFICIENCY
        action_penalty = -0.001 * np.sum(action ** 2)
        
        # 5. SMOOTHNESS (prevent jerky movements)
        smoothness_penalty = -0.01 * np.sum((action - self.prev_action) ** 2)
        
        # 6. HEIGHT STABILITY (stay near natural walking height)
        target_height = 1.3
        height_penalty = -0.3 * (height - target_height) ** 2
        
        # 7. UPRIGHT POSTURE (penalize falling)
        if self.use_direct_access:
            # Torso orientation (quaternion)
            quat = qpos[3:7]
            # z-component of up vector (should be close to 1)
            z_up = 2 * (quat[1] * quat[3] + quat[0] * quat[2])
            upright_reward = 0.5 * z_up
        else:
            upright_reward = 0.0
        
        # 8. STEP FREQUENCY REWARD (encourage dynamic gait)
        if self.use_direct_access:
            # Measure leg joint velocities
            # Humanoid joints: hip_x(7), hip_z(8), hip_y(9), knee(10) for right leg
            #                  hip_x(11), hip_z(12), hip_y(13), knee(14) for left leg
            right_leg_vel = np.mean(np.abs(qvel[7:11]))
            left_leg_vel = np.mean(np.abs(qvel[11:15]))
            
            # Encourage movement but not excessive
            leg_activity = min(right_leg_vel + left_leg_vel, 10.0)
            gait_reward = 0.1 * leg_activity if vx_actual > 0.2 else 0.0
        else:
            gait_reward = 0.0
        
        # TOTAL REWARD
        reward = (
            velocity_reward +      # 2.5 (main task)
            progress_reward +      # ~0.5-1.0 (encourages walking)
            alive_bonus +          # 1.0 (reduced)
            action_penalty +       # -0.0X
            smoothness_penalty +   # -0.0X
            height_penalty +       # -0.0X
            upright_reward +       # 0.5
            gait_reward           # 0.X
        )
        
        # TERMINATION
        terminated = (height < 0.8 or height > 2.2)
        
        # Add detailed info for debugging
        info['cost'] = constraint_cost
        info['velocity_reward'] = velocity_reward
        info['progress_reward'] = progress_reward
        info['vx_actual'] = vx_actual
        info['vx_error'] = vx_err
        info['distance'] = self.total_distance
        
        self.prev_action = action.copy()
        
        return self._get_obs(obs), reward, terminated, truncated, info
    
    def render(self):
        return self.base_env.render()
    
    def close(self):
        return self.base_env.close()