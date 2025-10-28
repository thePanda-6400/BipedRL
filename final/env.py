# environment.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class HumanoidVelocityEnv(gym.Env):
    def __init__(self, render_mode=None):
        # Base humanoid
        self.base_env = gym.make('Humanoid-v4', render_mode=render_mode)

        # Get unwrapped mujoco env for low-level access
        env = self.base_env
        while hasattr(env, "env"):
            env = env.env
        self.mujoco_env = env

        self.use_direct_access = hasattr(self.mujoco_env, "data")
        if self.use_direct_access:
            print("✓ MuJoCo direct access enabled.")
        else:
            print(" Running in limited observation mode (no MuJoCo data).")

        # Commanded velocities: [forward vx, yaw rate vz]
        self.command = np.zeros(2)
        self.prev_action = np.zeros(self.base_env.action_space.shape[0])

        # Extend observation to include command
        obs_dim = self.base_env.observation_space.shape[0] + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = self.base_env.action_space

        self.render_mode = render_mode

    # Reset episode and sample new velocity commands
    def reset(self, seed=None):
        obs, info = self.base_env.reset(seed=seed)
        self.command = np.array([
            np.random.uniform(0.0, 1.0),     # forward vx
            np.random.uniform(-0.5, 0.5)     # yaw rate
        ])
        self.prev_action = np.zeros(self.action_space.shape[0])
        return self._get_obs(obs), info

    def _get_obs(self, base_obs):
        return np.concatenate([base_obs, self.command])

    # Main step logic
    def step(self, action):
        obs, _, terminated, truncated, info = self.base_env.step(action)

        if self.use_direct_access:
            qpos = self.mujoco_env.data.qpos
            qvel = self.mujoco_env.data.qvel

            # Extract forward & yaw velocity
            vx_actual = qvel[0]
            vz_actual = qvel[5]
            height = qpos[2]

            # Physical constraint tracking
            torques = np.abs(self.mujoco_env.data.qfrc_actuator)
            torque_violation = np.sum(np.maximum(0, torques - 100))
            joint_vel = np.abs(qvel[6:])
            vel_violation = np.sum(np.maximum(0, joint_vel - 8.0))
            constraint_cost = torque_violation + vel_violation
        else:
            vx_actual = obs[22] if len(obs) > 22 else 0.0
            vz_actual = obs[27] if len(obs) > 27 else 0.0
            height = obs[0] if len(obs) > 0 else 1.5
            constraint_cost = np.maximum(0, np.abs(action) - 0.8).sum()

        # === REWARD DESIGN ===
        vx_err = self.command[0] - vx_actual
        vz_err = self.command[1] - vz_actual

        # 1. Velocity tracking (primary term)
        vel_reward = 3.0 * np.exp(-3.0 * (abs(vx_err) + 0.5 * abs(vz_err)))

        # 2. Upright posture reward
        alive_bonus = 5.0 if 1.0 < height < 2.0 else -10.0

        # 3. Energy efficiency & smoothness
        act_penalty = -0.002 * np.sum(action ** 2)
        smooth_penalty = -0.005 * np.sum((action - self.prev_action) ** 2)

        # 4. Step rhythm shaping — encourage alternating leg velocity
        if self.use_direct_access:
            leg_vel = np.mean(np.abs(qvel[7:13]))  # approximate leg motion
            gait_reward = 0.2 * leg_vel
        else:
            gait_reward = 0.0

        # 5. Height regularization
        height_penalty = -0.5 * (height - 1.3) ** 2

        reward = vel_reward + alive_bonus + act_penalty + smooth_penalty + gait_reward + height_penalty

        # Termination: fall or excessive height
        terminated = height < 0.8 or height > 2.2
        info['cost'] = constraint_cost

        self.prev_action = action.copy()
        return self._get_obs(obs), reward, terminated, truncated, info

    def render(self):
        return self.base_env.render()

    def close(self):
        return self.base_env.close()
