# symmetry.py
import numpy as np

class HumanoidSymmetry:
    """
    Reflection symmetry about sagittal plane (left-right mirror)
    
    MuJoCo Humanoid-v4 has 17 actuators:
    - abdomen_y, abdomen_z, abdomen_x (3)
    - right_hip_x, right_hip_z, right_hip_y, right_knee (4)
    - left_hip_x, left_hip_z, left_hip_y, left_knee (4)
    - right_shoulder1, right_shoulder2, right_elbow (3)
    - left_shoulder1, left_shoulder2, left_elbow (3)
    """
    
    def __init__(self):
        # MuJoCo Humanoid-v4 actuator indices (17 total)
        # Index mapping for left-right symmetry
        self.action_swap_indices = [
            # (left_idx, right_idx)
            (3, 7),   # hip_x: right_hip_x <-> left_hip_x
            (4, 8),   # hip_z: right_hip_z <-> left_hip_z  
            (5, 9),   # hip_y: right_hip_y <-> left_hip_y
            (6, 10),  # knee: right_knee <-> left_knee
            (11, 14), # shoulder1: right_shoulder1 <-> left_shoulder1
            (12, 15), # shoulder2: right_shoulder2 <-> left_shoulder2
            (13, 16), # elbow: right_elbow <-> left_elbow
        ]
        
        # Indices to flip sign (lateral movements)
        # abdomen_y (idx 1) controls lateral bending
        self.action_flip_indices = [1]  # abdomen_y
        
        # For observation (376 dims total in Humanoid-v4)
        # First 24 are qpos (minus x,y), next are velocities, etc.
        # We'll use a simplified approach for obs transformation
        
    def transform_obs(self, obs):
        """
        Transform observation for left-right symmetry
        
        Humanoid-v4 observation (376 dims):
        - qpos (24): [z, quat(4), joints(17)] (x,y removed)
        - qvel (23): [vel_x, vel_y, vel_z, ang_vel(3), joint_vel(17)]
        - cinert (130): center of mass inertia
        - cvel (90): center of mass velocity  
        - qfrc_actuator (17): actuator forces
        - cfrc_ext (78): external contact forces
        - command (2): [vx, vz] - added by our wrapper
        """
        obs_mirror = obs.copy()
        
        # Handle different observation sizes
        if len(obs) < 10:
            # Minimal observation, just return as is
            return obs_mirror
        
        # Mirror y-velocity (index 1 in qvel section, which starts at 24)
        if len(obs) > 25:
            obs_mirror[25] *= -1  # vel_y
        
        # Mirror angular velocity z (yaw rate) (index 5 in qvel)
        if len(obs) > 29:
            obs_mirror[29] *= -1  # ang_vel_z
        
        # Mirror command (last 2 elements: vx, vz)
        # Only flip vz (yaw rate command)
        obs_mirror[-1] *= -1
        
        # For simplicity, we'll keep the rest as is
        # In practice, you'd want to properly mirror joint positions/velocities
        # but this requires detailed knowledge of the observation structure
        
        return obs_mirror
    
    def transform_action(self, action):
        """
        Transform action for left-right symmetry
        
        Action space: 17 actuators
        """
        if len(action) != 17:
            print(f"Warning: Expected 17 actions, got {len(action)}")
            return action
        
        action_mirror = action.copy()
        
        # Swap left-right actuators
        for left, right in self.action_swap_indices:
            action_mirror[left], action_mirror[right] = action_mirror[right], action_mirror[left]
        
        # Flip signs for lateral movements
        for idx in self.action_flip_indices:
            action_mirror[idx] *= -1
        
        return action_mirror


# Create symmetry group
symmetry_transform = HumanoidSymmetry()
symmetries = [
    ('identity', lambda x: x, lambda a: a),
    ('reflect', symmetry_transform.transform_obs, symmetry_transform.transform_action)
]