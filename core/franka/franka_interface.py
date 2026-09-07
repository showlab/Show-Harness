"""Standalone Franka robot interface via ZeroRPC.

This is a simplified version extracted from franka_interpolation_controller.py
that only includes the FrankaInterface class for direct robot control.
"""

import numpy as np
import zerorpc


class FrankaInterface:
    """Interface to Franka robot via ZeroRPC server on NUC."""

    def __init__(self, ip="127.0.0.1", port=4242):
        """
        Initialize connection to Franka robot.

        Args:
            ip: IP address of NUC running Polymetis server
            port: Port of ZeroRPC server
        """
        self.server = zerorpc.Client(heartbeat=20)
        self.server.connect(f"tcp://{ip}:{port}")

    def get_ee_pose(self):
        """Get end-effector pose (7D: position + quaternion orientation).

        Returns:
            np.ndarray: [x, y, z, qx, qy, qz, qw] - position (3D) + quaternion (4D)
        """
        ee_pose = np.array(self.server.get_ee_pose())
        return ee_pose

    def get_joint_positions(self):
        """Get current joint positions (7D)."""
        return np.array(self.server.get_joint_positions())

    def get_joint_velocities(self):
        """Get current joint velocities (7D)."""
        return np.array(self.server.get_joint_velocities())

    def get_ee_pose_w_gripper(self):
        """Get end-effector pose with gripper state (7D)."""
        ee_pose = self.get_ee_pose()
        gripper_state = self.get_gripper_position()
        return np.concatenate([ee_pose, gripper_state])

    def get_joint_positions_w_gripper(self):
        """Get joint positions with gripper state (9D: 7 joints + 2x gripper)."""
        joint_pos = self.get_joint_positions()
        gripper_state = self.get_gripper_position()
        return np.concatenate([joint_pos, gripper_state, gripper_state])

    def get_joint_velocities_w_gripper(self):
        """Get joint velocities with gripper state (9D: 7 joints + 2x gripper)."""
        joint_vel = self.get_joint_velocities()
        gripper_state = self.get_gripper_position()
        return np.concatenate([joint_vel, gripper_state, gripper_state])

    def move_to_joint_positions(self, positions: np.ndarray, time_to_go: float):
        """
        Move to target joint positions.

        Args:
            positions: Target joint positions (7D)
            time_to_go: Time to reach target (seconds)
        """
        self.server.move_to_joint_positions(positions.tolist(), time_to_go)

    def start_cartesian_impedance(self, Kx: np.ndarray, Kxd: np.ndarray):
        """
        Start Cartesian impedance controller.

        Args:
            Kx: Position and orientation stiffness (6D: [x, y, z, rx, ry, rz])
            Kxd: Position and orientation damping (6D: [x, y, z, rx, ry, rz])
        """
        self.server.start_cartesian_impedance(Kx.tolist(), Kxd.tolist())

    def start_joint_impedance(self, Kq: np.ndarray = None, Kqd: np.ndarray = None):
        """
        Start joint impedance controller.

        Args:
            Kq: Joint stiffness (7D) or None for defaults
            Kqd: Joint damping (7D) or None for defaults
        """
        self.server.start_joint_impedance(Kq.tolist() if Kq is not None else None, Kqd.tolist() if Kqd is not None else None)

    def update_desired_ee_pose(self, pose: np.ndarray):
        """
        Update desired end-effector pose (for Cartesian impedance control).

        Args:
            pose: Desired end-effector pose (7D: [x, y, z, qx, qy, qz, qw])
                  position (3D) + quaternion orientation (4D)
        """
        self.server.update_desired_ee_pose(pose.tolist())

    def update_desired_joint_pos(self, pos: np.ndarray):
        """
        Update desired joint positions (for joint impedance control).

        Args:
            pos: Desired joint positions (7D)
        """
        self.server.update_desired_joint_pos(pos.tolist())

    def control_gripper(self, gripper_action: bool):
        """
        Control gripper (binary: open/close).

        Args:
            gripper_action: True to close, False to open
        """
        self.server.control_gripper(gripper_action)

    def get_gripper_position(self):
        """Get current gripper position (1D)."""
        gripper_position = np.array(self.server.get_gripper_position()).reshape(
            [
                1,
            ]
        )
        return gripper_position

    def set_gripper_position(self, pos: float):
        """
        Set gripper to specific position.

        Args:
            pos: Gripper width in meters (0.0 = fully closed, 0.08 = fully open)
        """
        self.server.set_gripper_position(pos)

    def get_gripper_prev_cmd_success(self):
        """Get whether the previous gripper command succeeded."""
        prev_cmd_success = self.server.get_gripper_prev_cmd_success()
        print(f"  [Gripper] Previous command success: {prev_cmd_success}")
        return prev_cmd_success

    def get_force_torque(self):
        """Get current force/torque readings (6D)."""
        return np.array(self.server.get_force_torque())

    def terminate_current_policy(self):
        """Terminate the currently running policy/controller."""
        self.server.terminate_current_policy()

    def close(self):
        """Close the connection to the robot."""
        self.server.close()


class MockRobot:
    """Mock robot for testing without hardware.

    Simulates a Franka robot by maintaining internal state and responding
    to commands without actually moving hardware.
    """

    def __init__(self, ip="127.0.0.1", port=4242):
        """
        Initialize mock robot.

        Args:
            ip: Ignored (for compatibility with FrankaInterface)
            port: Ignored (for compatibility with FrankaInterface)
        """
        print(f"  [MockRobot] Simulating robot at {ip}:{port} (no real connection)")

        # Initial joint configuration (roughly home position)
        self.joint_positions = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.joint_velocities = np.zeros(7)
        self.gripper_position = np.array([0.04])  # Half-open

        # Control state
        self.controller_active = False
        self.controller_type = None

        print(f"  [MockRobot] Initial joint positions: {self.joint_positions}")
        print(f"  [MockRobot] Initial gripper position: {self.gripper_position[0]:.4f}m")

    def get_ee_pose(self):
        """Get mock end-effector pose (7D: position + quaternion orientation).

        Returns:
            np.ndarray: [x, y, z, qx, qy, qz, qw] - position (3D) + quaternion (4D)
        """
        # Return a fixed pose for simplicity (identity quaternion: w=1, x=y=z=0)
        return np.array([0.3, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])

    def get_joint_positions(self):
        """Get current joint positions (7D)."""
        return self.joint_positions.copy()

    def get_joint_velocities(self):
        """Get current joint velocities (7D)."""
        return self.joint_velocities.copy()

    def get_ee_pose_w_gripper(self):
        """Get end-effector pose with gripper state (8D).

        Returns:
            np.ndarray: [x, y, z, qx, qy, qz, qw, gripper] - position (3D) + quaternion (4D) + gripper (1D)
        """
        ee_pose = self.get_ee_pose()
        gripper_state = self.get_gripper_position()
        return np.concatenate([ee_pose, gripper_state])

    def get_joint_positions_w_gripper(self):
        """Get joint positions with gripper state (9D: 7 joints + 2x gripper)."""
        joint_pos = self.get_joint_positions()
        gripper_state = self.get_gripper_position()
        return np.concatenate([joint_pos, gripper_state, gripper_state])

    def get_joint_velocities_w_gripper(self):
        """Get joint velocities with gripper state (9D: 7 joints + 2x gripper)."""
        joint_vel = self.get_joint_velocities()
        gripper_state = self.get_gripper_position()
        return np.concatenate([joint_vel, gripper_state, gripper_state])

    def move_to_joint_positions(self, positions: np.ndarray, time_to_go: float):
        """
        Simulate moving to target joint positions.

        Args:
            positions: Target joint positions (7D)
            time_to_go: Time to reach target (seconds)
        """
        # Instantly update to target (in real robot this would be smooth)
        self.joint_positions = np.array(positions).copy()
        self.joint_velocities = np.zeros(7)

    def start_cartesian_impedance(self, Kx: np.ndarray, Kxd: np.ndarray):
        """
        Simulate starting Cartesian impedance controller.

        Args:
            Kx: Position stiffness (6D)
            Kxd: Velocity damping (6D)
        """
        self.controller_active = True
        self.controller_type = "cartesian"
        print("  [MockRobot] Started Cartesian impedance controller")

    def start_joint_impedance(self, Kq: np.ndarray = None, Kqd: np.ndarray = None):
        """
        Simulate starting joint impedance controller.

        Args:
            Kq: Joint stiffness (7D) or None for defaults
            Kqd: Joint damping (7D) or None for defaults
        """
        self.controller_active = True
        self.controller_type = "joint"
        print("  [MockRobot] Started joint impedance controller")

    def update_desired_ee_pose(self, pose: np.ndarray):
        """
        Simulate updating desired end-effector pose.

        Args:
            pose: Desired end-effector pose (7D: [x, y, z, qx, qy, qz, qw])
                  position (3D) + quaternion orientation (4D)
        """
        # In real robot, this would move the arm smoothly
        # For mock, we just acknowledge the command
        pass

    def update_desired_joint_pos(self, pos: np.ndarray):
        """
        Simulate updating desired joint positions.

        Args:
            pos: Desired joint positions (7D)
        """
        # Smoothly update towards target (simple integration)
        target = np.array(pos)
        diff = target - self.joint_positions

        # Move 10% of the way each time (simple smoothing)
        self.joint_positions += 0.1 * diff
        self.joint_velocities = 0.1 * diff  # Approximate velocity

    def control_gripper(self, gripper_action: bool):
        """
        Simulate gripper control (binary: open/close).

        Args:
            gripper_action: True to close, False to open
        """
        if gripper_action:
            # Close gripper
            self.gripper_position = np.array([0.0])
        else:
            # Open gripper
            self.gripper_position = np.array([0.08])

    def get_gripper_position(self):
        """Get current gripper position (1D)."""
        return self.gripper_position.copy()

    def set_gripper_position(self, pos: float):
        """
        Simulate setting gripper to specific position.

        Args:
            pos: Gripper width in meters (0.0 = fully closed, 0.08 = fully open)
        """
        self.gripper_position = np.array([np.clip(pos, 0.0, 0.08)])

    def get_force_torque(self):
        """Get mock force/torque readings (6D)."""
        # Return small random values to simulate sensor noise
        return np.random.randn(6) * 0.1

    def terminate_current_policy(self):
        """Simulate terminating the currently running policy/controller."""
        self.controller_active = False
        self.controller_type = None
        print("  [MockRobot] Terminated controller")

    def close(self):
        """Close the mock connection."""
        print("  [MockRobot] Closed (mock)")
