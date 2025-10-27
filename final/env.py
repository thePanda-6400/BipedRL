import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

class G1VelocityTrackEnv(gym.Env):
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path("unitree_mujoco/unitree_robots/g1/scene.xml")
        self.data  = mujoco.MjData(self.model)
        n_joints = self.model.nu
        self.action_space = spaces.Box(low=-1, high=1, shape=(n_joints,))
        # obs: base state + joint + target cmd
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_joints*2 + 6,))
        self.cmd = np.zeros(2)  # vx, vz
    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        self.cmd = np.random.uniform([0.0, -0.5], [1.0, 0.5])   # sample target velocity
        return self._get_obs(), {}
    def _get_obs(self):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        return np.concatenate([qpos, qvel, self.cmd])
    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        v_forward = self.data.qvel[0]
        yaw_rate  = self.data.qvel[5]
        reward_v  = -((v_forward - self.cmd[0])**2 + 0.2*(yaw_rate - self.cmd[1])**2)
        reward = reward_v - 0.001*np.sum(action**2)
        done = False
        return obs, reward, done, False, {}
