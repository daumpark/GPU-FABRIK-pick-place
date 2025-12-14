#!/usr/bin/env python3
import sys
import time
import math
import argparse
import rclpy
import torch
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
from piper_gpu_ik.srv import BatchIk
from gazebo_msgs.srv import GetEntityState
from linkattacher_msgs.srv import AttachLink, DetachLink
from piper_gpu_ik_py.gpu_fabrik_core import (
    URDFKinematicsTorch,
    build_default_robot_urdf_path,
    get_default_link_names,
)

# Robot spawn position from launch file
ROBOT_BASE_X = 11.4
ROBOT_BASE_Y = -12.2
ROBOT_BASE_Z = 0.81

class PickAndPlace(Node):
    def __init__(self):
        super().__init__('pick_and_place')
        self.ik_client = self.create_client(BatchIk, 'batch_ik')
        self.state_client = self.create_client(GetEntityState, '/get_entity_state')
        self.attach_client = self.create_client(AttachLink, '/ATTACHLINK')
        self.detach_client = self.create_client(DetachLink, '/DETACHLINK')
        
        # Publishers
        self.arm_pub = self.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.gripper_pub = self.create_publisher(JointTrajectory, '/gripper_controller/joint_trajectory', 10)

        # Kinematics for path collision check (CPU)
        urdf_path = build_default_robot_urdf_path()
        self.kin = URDFKinematicsTorch(urdf_path, get_default_link_names(), device=torch.device("cpu"), dtype=torch.float64)
        self.last_arm_q = None
        self.obstacles = []
        self.obs_clearance = 0.02
        
        self.get_logger().info('Waiting for IK service...')
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('IK service not available, waiting again...')
            
        self.get_logger().info('Waiting for Gazebo State service...')
        while not self.state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Gazebo State service not available, waiting again...')

        # LinkAttacher might not be available if plugin is not loaded, but let's try to wait a bit or just proceed
        # It's better not to block infinitely if user forgot to add plugin
        if not self.attach_client.wait_for_service(timeout_sec=2.0):
             self.get_logger().warn('LinkAttacher service not available. Grasping might fail.')
            
        self.get_logger().info('Services connected.')

    def attach_object(self, object_name):
        if not self.attach_client.service_is_ready():
            return

        req = AttachLink.Request()
        req.model1_name = 'piper'
        req.link1_name = 'link7'
        req.model2_name = object_name
        req.link2_name = 'link_2' # Default for cubes
        
        future = self.attach_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        # Service returns success bool in result
        # Check definition: bool success, string message
        if future.result():
             self.get_logger().info(f'Attach request sent for {object_name}')
        else:
             self.get_logger().warn(f'Failed to call Attach for {object_name}')

    def detach_object(self, object_name):
        if not self.detach_client.service_is_ready():
            return

        req = DetachLink.Request()
        req.model1_name = 'piper'
        req.link1_name = 'link7'
        req.model2_name = object_name
        req.link2_name = 'link_2'
        
        future = self.detach_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result():
             self.get_logger().info(f'Detach request sent for {object_name}')
        else:
             self.get_logger().warn(f'Failed to call Detach for {object_name}')

    def get_cube_pose(self, name):
        req = GetEntityState.Request()
        req.name = name
        req.reference_frame = 'world'
        
        future = self.state_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()
        
        if not resp.success:
            self.get_logger().error(f'Failed to get pose for {name}')
            return None
            
        # Convert World Frame -> Robot Base Frame
        # Robot is at (11.4, -12.2, 0.81) with identity rotation
        
        wx = resp.state.pose.position.x
        wy = resp.state.pose.position.y
        wz = resp.state.pose.position.z
        
        rx = wx - ROBOT_BASE_X
        ry = wy - ROBOT_BASE_Y
        rz = wz - ROBOT_BASE_Z
        
        self.get_logger().info(f"Found {name} at World: ({wx:.3f}, {wy:.3f}, {wz:.3f}) -> Robot: ({rx:.3f}, {ry:.3f}, {rz:.3f})")
        return (rx, ry, rz)

    def get_ik(self, x, y, z, roll, pitch, yaw, tool_len=0.13, w_ori=0.5, tol_ang_deg=5.0):
        req = BatchIk.Request()
        req.target.position.x = float(x)
        req.target.position.y = float(y)
        req.target.position.z = float(z)
        
        # Euler to Quat
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        req.target.orientation.w = cr * cp * cy + sr * sp * sy
        req.target.orientation.x = sr * cp * cy - cr * sp * sy
        req.target.orientation.y = cr * sp * cy + sr * cp * sy
        req.target.orientation.z = cr * cp * sy - sr * sp * cy

        # Defaults
        req.num_seeds = 1000
        req.include_q0 = True
        req.tol_pos = 0.01
        req.tol_ang_deg = float(tol_ang_deg)
        req.use_cuda = True 
        req.w_pos = 10.0
        req.w_ori = float(w_ori) 
        req.tool_len = float(tool_len)

        future = self.ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def send_arm_traj(self, joints, duration=2.0):
        # Path collision check (linear in joint space)
        if self.last_arm_q is not None and len(self.obstacles) > 0:
            ok = self._check_path_collision(self.last_arm_q, joints)
            if not ok:
                self.get_logger().error('Path collision predicted; skipping arm command')
                return False

        msg = JointTrajectory()
        msg.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        pt = JointTrajectoryPoint()
        # Convert to standard python float (double) to satisfy 'd' typecode expectation
        pt.positions = [float(j) for j in joints]
        pt.time_from_start.sec = int(duration)
        pt.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
        msg.points.append(pt)
        self.arm_pub.publish(msg)
        time.sleep(duration + 0.1)
        self.last_arm_q = list(joints)
        return True

    def send_gripper(self, pos, duration=1.0, wait=True):
        msg = JointTrajectory()
        msg.joint_names = ['joint7', 'joint8'] 
        pt = JointTrajectoryPoint()
        # joint8 mirrors joint7 in the opposite direction
        pt.positions = [float(pos), float(-pos)]
        pt.time_from_start.sec = int(duration)
        pt.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
        msg.points.append(pt)
        self.gripper_pub.publish(msg)
        if wait:
            time.sleep(duration + 0.1)

    def _check_path_collision(self, q_start, q_goal, steps=10):
        q_start = torch.tensor(q_start, dtype=torch.float64)
        q_goal = torch.tensor(q_goal, dtype=torch.float64)
        t_vals = torch.linspace(0.0, 1.0, steps)
        qs = q_start.unsqueeze(0) * (1 - t_vals).unsqueeze(1) + q_goal.unsqueeze(0) * t_vals.unsqueeze(1)
        pts, _ = self.kin.fk_points_axes(qs)
        pen = self._collision_penalty(pts)
        return float(pen.max()) <= 1e-6

    def _collision_penalty(self, pts):
        """
        pts: (B, M, 3) torch
        obstacles: list of (R, p, h)
        """
        if len(self.obstacles) == 0:
            return torch.zeros(pts.shape[0], dtype=pts.dtype)

        penalties = []
        for R, p, h in self.obstacles:
            pts_rel = pts - p
            pts_local = torch.einsum("bkj,ij->bki", pts_rel, R.transpose(0, 1))
            q = torch.abs(pts_local) - h
            dist_out = torch.linalg.norm(torch.clamp(q, min=0.0), dim=-1)
            max_q = torch.max(q, dim=-1).values
            signed = torch.where(dist_out > 0, dist_out, max_q)
            pen = torch.clamp(self.obs_clearance - signed, min=0.0)
            penalties.append(pen)
        pen_all = torch.stack(penalties, dim=0).max(dim=0).values
        return pen_all.max(dim=1).values
def main():
    rclpy.init()
    node = PickAndPlace()

    parser = argparse.ArgumentParser(description='Piper Pick and Place Demo')
    parser.add_argument('--cube', type=str, default='', help='Name of the cube to pick (e.g., cube_b)')
    parser.add_argument('--pick_x', type=float, default=0.3, help='Pick X coordinate')
    parser.add_argument('--pick_y', type=float, default=0.0, help='Pick Y coordinate')
    parser.add_argument('--pick_z', type=float, default=0.15, help='Pick Z coordinate')
    parser.add_argument('--place_x', type=float, default=0.3, help='Place X coordinate')
    parser.add_argument('--place_y', type=float, default=-0.3, help='Place Y coordinate')
    parser.add_argument('--place_z', type=float, default=0.15, help='Place Z coordinate')
    parser.add_argument('--roll', type=float, default=0.0, help='Gripper Roll (rad)')
    parser.add_argument('--pitch', type=float, default=0.0, help='Gripper Pitch (rad) - 1.57 is usually down')
    parser.add_argument('--yaw', type=float, default=0.0, help='Gripper Yaw (rad)')
    parser.add_argument('--grip_width', type=float, default=0.1, help='Gripper close width (m)')
    parser.add_argument('--obstacles', type=str, default='', help='Comma-separated list of obstacle cube names')
    parser.add_argument('--obs_half', type=float, default=0.035, help='Half extent of obstacle cubes (m)')
    
    args, unknown = parser.parse_known_args()

    try:
        pick_x, pick_y, pick_z = args.pick_x, args.pick_y, args.pick_z
        place_x, place_y, place_z = args.place_x, args.place_y, args.place_z
        node.obs_clearance = 0.02

        # If cube name is provided, fetch its pose
        if args.cube:
            node.get_logger().info(f'Getting pose for cube: {args.cube}')
            pose = node.get_cube_pose(args.cube)
            if pose:
                pick_x, pick_y, pick_z = pose
            else:
                return

        # Obstacles: other cubes (excluding the target cube)
        obs_names = [n.strip() for n in args.obstacles.split(',') if n.strip()]
        # Add fixed props to avoid during motion
        obs_names.extend(['OpenRobotics/AdjTable', '100mmbin', 'unit_box'])
        obstacles = []
        for nm in obs_names:
            if nm == args.cube:
                continue
            pose = node.get_cube_pose(nm)
            if pose:
                x, y, z = pose
                T = torch.eye(4, dtype=torch.float64)
                T[0, 3] = x
                T[1, 3] = y
                T[2, 3] = z
                half = torch.tensor([args.obs_half, args.obs_half, args.obs_half], dtype=torch.float64)
                obstacles.append((T[:3, :3], T[:3, 3], half))
        node.obstacles = obstacles

        # 1. Open Gripper (Max width 0.035)
        node.get_logger().info('Opening Gripper')
        node.send_gripper(0.035)

        # 2. Go to Pre-Pick (Above)
        node.get_logger().info(f'Moving to Pre-Pick: {pick_x}, {pick_y}, {pick_z + 0.1}')
        res = node.get_ik(
            pick_x, pick_y, pick_z + 0.1, args.roll, args.pitch, args.yaw,
            tool_len=0.13, w_ori=0.5, tol_ang_deg=10.0,
        )
        if res.success:
            if not node.send_arm_traj(res.solution):
                return
        else:
            node.get_logger().error(f'IK Failed for Pre-Pick: {res.message}')
            return

        # 3. Go to Pick (Down)
        node.get_logger().info(f'Moving to Pick: {pick_x}, {pick_y}, {pick_z}')
        res = node.get_ik(
            pick_x, pick_y, pick_z, args.roll, args.pitch, args.yaw,
            tool_len=0.13, w_ori=0.5, tol_ang_deg=10.0,
        )
        if res.success:
            if not node.send_arm_traj(res.solution):
                return
        else:
             node.get_logger().error(f'IK Failed for Pick: {res.message}')
             return

        # 4. Close Gripper
        half_width = args.grip_width / 2.0
        node.get_logger().info(f'Closing Gripper to Width {args.grip_width} (Joint: {half_width})')
        # Arm is stationary here; close gripper synchronously to avoid moving while gripping
        node.send_gripper(half_width, wait=True)
        # Small extra settle time
        time.sleep(0.2)
        if args.cube:
            node.attach_object(args.cube)

        # 5. Lift (Pick Z + 0.1)
        node.get_logger().info('Lifting')
        res = node.get_ik(
            pick_x, pick_y, pick_z + 0.1, args.roll, args.pitch, args.yaw,
            tool_len=0.13, w_ori=0.0, tol_ang_deg=180.0,
        )
        if res.success:
            if not node.send_arm_traj(res.solution):
                return

        # 6. Move to Place (Above)
        node.get_logger().info(f'Moving to Pre-Place: {args.place_x}, {args.place_y}, {args.place_z + 0.1}')
        res = node.get_ik(
            args.place_x, args.place_y, args.place_z + 0.1, args.roll, args.pitch, args.yaw,
            tool_len=0.13, w_ori=0.0, tol_ang_deg=180.0,
        )
        if res.success:
            if not node.send_arm_traj(res.solution):
                return
        else:
            node.get_logger().error('IK Failed for Pre-Place')

        # 7. Lower to Place
        node.get_logger().info('Lowering to Place')
        res = node.get_ik(
            args.place_x, args.place_y, args.place_z, args.roll, args.pitch, args.yaw,
            tool_len=0.13, w_ori=0.0, tol_ang_deg=180.0,
        )
        if res.success:
            if not node.send_arm_traj(res.solution):
                return

        # 8. Open Gripper
        node.get_logger().info('Releasing')
        # Arm is stationary; open gripper after motion completes
        node.send_gripper(0.035, wait=True)
        if args.cube:
            node.detach_object(args.cube)
        
        # 9. Retract
        node.get_logger().info('Retracting')
        res = node.get_ik(
            args.place_x, args.place_y, args.place_z + 0.1, args.roll, args.pitch, args.yaw,
            tool_len=0.13, w_ori=0.0, tol_ang_deg=180.0,
        )
        if res.success:
            if not node.send_arm_traj(res.solution):
                return
            
        node.get_logger().info('Pick and Place sequence completed.')

    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
