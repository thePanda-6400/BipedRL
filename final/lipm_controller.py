# final/lipm_controller.py
"""
Linear Inverted Pendulum Model (LIPM) Walking Controller
Generates baseline walking pattern for Humanoid-v4
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class LIPMParams:
    """LIPM walking parameters optimized for Humanoid-v4"""
    com_height: float = 0.9           # Center of mass height (m)
    step_length: float = 0.25          # Forward step length (m)
    step_width: float = 0.15           # Lateral distance between feet (m)
    step_height: float = 0.04          # Swing foot lift height (m)
    step_duration: float = 0.5         # Time per step (s)
    
    # Gains for velocity tracking
    velocity_gain: float = 0.6
    yaw_gain: float = 0.15


class LIPMWalkingController:
    """
    LIPM-based walking controller for Humanoid-v4
    Outputs 17-DOF joint commands that produce stable bipedal walking
    """
    
    def __init__(self, params: LIPMParams = None):
        self.params = params or LIPMParams()
        self.phase = 0.0  # Walking phase [0, 1]
        self.step_count = 0
        
        # Target velocities
        self.target_vx = 0.5  # m/s forward
        self.target_vz = 0.0  # rad/s yaw rate
        
        # Walking frequency
        self.omega = 2 * np.pi / self.params.step_duration
        
        print("🚶 LIPM Walking Controller Initialized")
        print(f"   Step duration: {self.params.step_duration}s")
        print(f"   Step length: {self.params.step_length}m")
        print(f"   Frequency: {1.0/self.params.step_duration:.2f} Hz")
    
    def set_command(self, vx: float, vz: float = 0.0):
        """
        Set target walking velocity
        
        Args:
            vx: Forward velocity (m/s) [0.0, 1.5]
            vz: Yaw rate (rad/s) [-0.5, 0.5]
        """
        self.target_vx = np.clip(vx, 0.0, 1.5)
        self.target_vz = np.clip(vz, -0.5, 0.5)
    
    def update_phase(self, dt: float):
        """Update walking phase based on desired velocity"""
        # Scale frequency by velocity (walk faster = higher frequency)
        velocity_scale = max(0.3, min(self.target_vx / 0.5, 1.5))
        actual_duration = self.params.step_duration / velocity_scale
        
        self.phase += dt / actual_duration
        
        # Wrap phase to [0, 1]
        if self.phase >= 1.0:
            self.phase -= 1.0
            self.step_count += 1
    
    def get_baseline_action(self, dt: float = 0.015) -> np.ndarray:
        """
        Generate LIPM baseline walking pattern
        
        Args:
            dt: Control timestep (default 15ms)
        
        Returns:
            action: 17-DOF joint commands for Humanoid-v4
        """
        self.update_phase(dt)
        
        # Initialize action vector
        action = np.zeros(17, dtype=np.float32)
        
        # Current phase [0, 1]
        t = self.phase
        omega = self.omega
        
        # Scale factors based on target velocity
        vel_scale = self.params.velocity_gain * max(0.3, min(self.target_vx, 1.0))
        yaw_scale = self.params.yaw_gain * self.target_vz
        
        # ================================================================
        # ABDOMEN (joints 0-2): Torso stabilization and turning
        # ================================================================
        action[0] = 0.05 * np.sin(omega * t)           # abdomen_y: pitch oscillation
        action[1] = yaw_scale                           # abdomen_z: yaw for turning
        action[2] = 0.03 * np.sin(2 * omega * t)       # abdomen_x: slight roll
        
        # ================================================================
        # LEGS: Alternating gait with LIPM-inspired trajectories
        # Phase 0.0-0.5: Right leg stance, Left leg swing
        # Phase 0.5-1.0: Left leg stance, Right leg swing
        # ================================================================
        
        # === RIGHT LEG (joints 3-6) ===
        if t < 0.5:
            # Stance phase: support body weight
            hip_pitch_r = -0.15 + 0.25 * vel_scale * np.sin(omega * t)
            hip_roll_r = 0.08   # Slight abduction for balance
            hip_yaw_r = -yaw_scale * 0.1
            knee_r = 0.25 + 0.15 * vel_scale * np.sin(omega * t)
        else:
            # Swing phase: leg moves forward with knee flexion
            swing_t = (t - 0.5) * 2  # Normalize to [0, 1]
            hip_pitch_r = 0.25 + 0.5 * vel_scale * np.sin(np.pi * swing_t)
            hip_roll_r = 0.08
            hip_yaw_r = yaw_scale * 0.1
            knee_r = 0.5 * vel_scale * np.sin(np.pi * swing_t)  # Flex during swing
        
        action[3] = hip_pitch_r   # right_hip_x (hip flexion/extension)
        action[4] = hip_roll_r    # right_hip_z (hip abduction/adduction)
        action[5] = hip_yaw_r     # right_hip_y (hip rotation)
        action[6] = knee_r        # right_knee (knee flexion)
        
        # === LEFT LEG (joints 7-10) ===
        if t < 0.5:
            # Swing phase: leg moves forward
            swing_t = t * 2
            hip_pitch_l = 0.25 + 0.5 * vel_scale * np.sin(np.pi * swing_t)
            hip_roll_l = -0.08  # Opposite side abduction
            hip_yaw_l = yaw_scale * 0.1
            knee_l = 0.5 * vel_scale * np.sin(np.pi * swing_t)
        else:
            # Stance phase: support body
            hip_pitch_l = -0.15 + 0.25 * vel_scale * np.sin(omega * (t - 0.5))
            hip_roll_l = -0.08
            hip_yaw_l = -yaw_scale * 0.1
            knee_l = 0.25 + 0.15 * vel_scale * np.sin(omega * (t - 0.5))
        
        action[7] = hip_pitch_l   # left_hip_x
        action[8] = hip_roll_l    # left_hip_z
        action[9] = hip_yaw_l     # left_hip_y
        action[10] = knee_l       # left_knee
        
        # ================================================================
        # ARMS (joints 11-16): Natural arm swing opposite to legs
        # ================================================================
        arm_swing_amplitude = 0.25 * vel_scale
        
        # Right arm swings with left leg (opposite coordination)
        action[11] = -arm_swing_amplitude * np.sin(omega * t)  # right_shoulder1
        action[12] = 0.0                                        # right_shoulder2
        action[13] = 0.15                                       # right_elbow: slight bend
        
        # Left arm swings with right leg
        action[14] = arm_swing_amplitude * np.sin(omega * t)   # left_shoulder1
        action[15] = 0.0                                        # left_shoulder2
        action[16] = 0.15                                       # left_elbow: slight bend
        
        # Safety clipping
        action = np.clip(action, -1.0, 1.0)
        
        return action
    
    def get_state_info(self) -> dict:
        """Return current controller state"""
        return {
            'phase': self.phase,
            'step_count': self.step_count,
            'target_vx': self.target_vx,
            'target_vz': self.target_vz,
        }