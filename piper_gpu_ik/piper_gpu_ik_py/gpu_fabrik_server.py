#!/usr/bin/env python3
import math
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose

from piper_gpu_ik.srv import BatchIk
from piper_gpu_ik_py.gpu_fabrik_core import build_default_robot_urdf_path, get_default_link_names, run_parallel_seeds


def pose_to_matrix(pose: Pose) -> np.ndarray:
    qx, qy, qz, qw = pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, xy, xz = qx * x2, qx * y2, qx * z2
    yy, yz, zz = qy * y2, qy * z2, qz * z2
    wx, wy, wz = qw * x2, qw * y2, qw * z2

    R = np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ], dtype=float)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[0, 3] = pose.position.x
    T[1, 3] = pose.position.y
    T[2, 3] = pose.position.z
    return T


class GpuFabrikServer(Node):
    def __init__(self):
        super().__init__('gpu_fabrik_server')
        self.urdf_path = build_default_robot_urdf_path()
        self.link_names = get_default_link_names()
        self.q0 = None

        self.declare_parameter('default_num_seeds', 256)
        self.declare_parameter('default_tol_pos', 0.005)
        self.declare_parameter('default_tol_ang_deg', 5.0)
        self.declare_parameter('default_w_pos', 1.0)
        self.declare_parameter('default_w_ori', 0.05)
        self.declare_parameter('use_cuda', True)

        self.srv = self.create_service(BatchIk, 'batch_ik', self.handle_request)
        self.get_logger().info('gpu_fabrik_server ready (service: /batch_ik)')

    def handle_request(self, request: BatchIk.Request, response: BatchIk.Response):
        num_seeds = request.num_seeds if request.num_seeds > 0 else int(self.get_parameter('default_num_seeds').value)
        tol_pos = request.tol_pos if request.tol_pos > 0 else float(self.get_parameter('default_tol_pos').value)
        tol_ang_deg = request.tol_ang_deg if request.tol_ang_deg > 0 else float(self.get_parameter('default_tol_ang_deg').value)
        w_pos = request.w_pos if request.w_pos > 0 else float(self.get_parameter('default_w_pos').value)
        w_ori = request.w_ori if request.w_ori > 0 else float(self.get_parameter('default_w_ori').value)

        use_cuda = request.use_cuda and torch.cuda.is_available() and bool(self.get_parameter('use_cuda').value)
        device = torch.device('cuda') if use_cuda else torch.device('cpu')

        T_goal = pose_to_matrix(request.target)

        try:
            results, info = run_parallel_seeds(
                urdf_path=self.urdf_path,
                link_names=self.link_names,
                T_ee_target=T_goal,
                num_seeds=num_seeds,
                include_q0=bool(request.include_q0),
                q0=self.q0,
                tool_len=request.tool_len,
                w_pos=w_pos,
                w_ori=w_ori,
                tol_pos_n1=tol_pos,
                tol_ang_n1_deg=tol_ang_deg,
                device=device,
                do_warmup=True,
                warmup_rounds=1,
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'IK solve failed: {exc}')
            response.success = False
            response.solution = []
            response.err_pos = float('nan')
            response.err_ang = float('nan')
            response.message = f'exception: {exc}'
            return response

        if not results:
            response.success = False
            response.solution = []
            response.err_pos = float('inf')
            response.err_ang = float('inf')
            response.message = 'no solutions from solver'
            return response

        best = results[0]
        response.success = bool(best['ok'])
        response.solution = [float(x) for x in best['q_full']]
        response.err_pos = float(best['err_pos'])
        response.err_ang = float(best['err_ang_rad'])
        response.message = (
            f"batch={info.get('batch')} dev={info.get('device')} "
            f"pos_err={best['err_pos']:.4f} ang_err={math.degrees(best['err_ang_rad']):.2f}deg"
        )
        return response


def main(args=None):
    rclpy.init(args=args)
    node = GpuFabrikServer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
