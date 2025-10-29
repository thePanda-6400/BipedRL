# final/test_lipm.py
"""
Test LIPM controller without RL
"""

import numpy as np
from lipm_env import make_lipm_env
import time


def test_pure_lipm(num_steps=2000, render=True):
    """Test pure LIPM controller (no RL corrections)"""
    
    print("🧪 Testing pure LIPM walking controller...")
    
    # Create environment
    env = make_lipm_env(
        render_mode='human' if render else None,
        use_lipm=True,
        correction_scale=0.0  # No corrections, pure LIPM
    )
    
    obs, _ = env.reset()
    
    total_reward = 0
    total_cost = 0
    
    for step in range(num_steps):
        # Zero correction (pure LIPM)
        correction = np.zeros(17)
        
        obs, reward, terminated, truncated, info = env.step(correction)
        
        total_reward += reward
        total_cost += info.get('cost', 0)
        
        if step % 100 == 0:
            print(f"Step {step}: R={total_reward:.1f}, C={total_cost:.1f}, "
                  f"Phase={info['lipm_phase']:.2f}, Steps={info['lipm_step_count']}")
        
        if terminated or truncated:
            print(f"\n❌ Episode ended at step {step}")
            break
        
        if render:
            time.sleep(0.015)
    
    env.close()
    
    print(f"\n✅ Test complete!")
    print(f"   Total reward: {total_reward:.1f}")
    print(f"   Total cost: {total_cost:.1f}")
    print(f"   Steps: {info['lipm_step_count']}")
    
    return total_reward


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render visualization')
    parser.add_argument('--steps', type=int, default=2000, help='Number of steps')
    args = parser.parse_args()
    
    test_pure_lipm(num_steps=args.steps, render=args.render)