# test_env.py
"""Test environment setup"""

from env import HumanoidVelocityEnv
import numpy as np

def test_environment():
    print("Testing HumanoidVelocityEnv...")
    
    env = HumanoidVelocityEnv(render_mode='rgb_array')
    print(f"✓ Environment created")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")
    
    obs, info = env.reset()
    print(f"✓ Reset successful")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Command: vx={env.command[0]:.2f}, vz={env.command[1]:.2f}")
    
    total_reward = 0
    total_cost = 0
    
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        total_cost += info.get('cost', 0)
        
        if terminated or truncated:
            print(f"  Episode ended at step {i}")
            break
    
    print(f"✓ Environment test passed!")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Total cost: {total_cost:.2f}")
    
    env.close()

if __name__ == '__main__':
    test_environment()