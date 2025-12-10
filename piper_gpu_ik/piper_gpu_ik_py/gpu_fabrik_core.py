import math
import time
from typing import Optional, Tuple, List
import os

import numpy as np
import torch
from urdf_parser_py.urdf import URDF
from ament_index_python.packages import get_package_share_directory

def to_torch(x, device=None, dtype=torch.float64):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)

def normalize(v, eps=1e-12):
    n = torch.linalg.norm(v, dim=-1, keepdim=True).clamp(min=eps)
    return v / n

def rpy_to_matrix_torch(rpy, device=None, dtype=torch.float64):
    """
    Convert (roll, pitch, yaw) to 3x3 rotation matrix.
    rpy: (3,) tensor or list
    """
    if not isinstance(rpy, torch.Tensor):
        rpy = torch.tensor(rpy, device=device, dtype=dtype)
    
    r, p, y = rpy[0], rpy[1], rpy[2]
    
    cr, sr = torch.cos(r), torch.sin(r)
    cp, sp = torch.cos(p), torch.sin(p)
    cy, sy = torch.cos(y), torch.sin(y)
    
    # Rz * Ry * Rx
    R = torch.zeros(3, 3, device=device, dtype=dtype)
    
    R[0, 0] = cy * cp
    R[0, 1] = cy * sp * sr - sy * cr
    R[0, 2] = cy * sp * cr + sy * sr
    
    R[1, 0] = sy * cp
    R[1, 1] = sy * sp * sr + cy * cr
    R[1, 2] = sy * sp * cr - cy * sr
    
    R[2, 0] = -sp
    R[2, 1] = cp * sr
    R[2, 2] = cp * cr
    
    return R

def so3_log(R):
    tr = (R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]).clamp(-1.0, 3.0)
    cos_theta = (tr - 1.0) * 0.5
    theta = torch.acos(cos_theta.clamp(-1.0, 1.0))
    B = R.shape[0]
    w = torch.zeros(B, 3, dtype=R.dtype, device=R.device)

    small = theta < 1e-7
    large = ~small

    if small.any():
        Rs = R[small]
        w_s = torch.stack([
            (Rs[:, 2, 1] - Rs[:, 1, 2]) / 2.0,
            (Rs[:, 0, 2] - Rs[:, 2, 0]) / 2.0,
            (Rs[:, 1, 0] - Rs[:, 0, 1]) / 2.0
        ], dim=-1)
        w[small] = w_s

    if large.any():
        Rl = R[large]
        tl = theta[large].unsqueeze(-1)
        w_l = torch.stack([
            (Rl[:, 2, 1] - Rl[:, 1, 2]),
            (Rl[:, 0, 2] - Rl[:, 2, 0]),
            (Rl[:, 1, 0] - Rl[:, 0, 1])
        ], dim=-1) * (0.5 / torch.sin(tl))
        w[large] = w_l * tl

    return w


class URDFKinematicsTorch:
    def __init__(self, urdf_path, link_names, joint_limits=None, device=None, dtype=torch.float64):
        """
        urdf_path: Path to URDF file
        link_names: List of link names in order (Base -> ... -> EndEffector)
        joint_limits: Optional list of (min, max) for each active joint. If None, read from URDF.
        """
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.dtype = dtype
        
        # Load URDF
        # Manually read file to avoid encoding issues with lxml used by urdf_parser_py
        with open(urdf_path, 'r') as f:
            xml_string = f.read()
        
        # Remove XML declaration if present to avoid lxml "Unicode strings with encoding declaration" error
        if xml_string.lstrip().startswith('<?xml'):
            end_idx = xml_string.find('?>')
            if end_idx != -1:
                xml_string = xml_string[end_idx+2:]
                
        robot = URDF.from_xml_string(xml_string)
        self.robot = robot
        
        self.link_names = link_names
        self.N = len(link_names) - 1 # Number of joints
        
        # Parse kinematics chain
        self.fixed_transforms = [] # T_fixed for each joint (parent to child frame before rotation)
        self.joint_axes = [] # Rotation axis for each joint
        self.limits = []
        
        # Traverse chain
        # Assuming link_names[i] -> joint[i] -> link_names[i+1]
        for i in range(self.N):
            parent = link_names[i]
            child = link_names[i+1]
            
            # Find joint connecting parent to child
            joint = None
            for j in robot.joints:
                if j.parent == parent and j.child == child:
                    joint = j
                    break
            
            if joint is None:
                raise ValueError(f"No joint found between {parent} and {child}")
            
            # Fixed transform (Origin)
            xyz = joint.origin.xyz if joint.origin else [0, 0, 0]
            rpy = joint.origin.rpy if joint.origin else [0, 0, 0]
            
            R_fixed = rpy_to_matrix_torch(rpy, device=self.device, dtype=self.dtype)
            p_fixed = torch.tensor(xyz, device=self.device, dtype=self.dtype)
            
            T_fixed = torch.eye(4, device=self.device, dtype=self.dtype)
            T_fixed[:3, :3] = R_fixed
            T_fixed[:3, 3] = p_fixed
            self.fixed_transforms.append(T_fixed)
            
            # Joint Axis
            axis = joint.axis if joint.axis else [0, 0, 1] 
            self.joint_axes.append(torch.tensor(axis, device=self.device, dtype=self.dtype))
            
            # Limits
            if joint_limits is not None:
                self.limits.append(joint_limits[i])
            else:
                if joint.limit:
                    self.limits.append((joint.limit.lower, joint.limit.upper))
                else:
                    self.limits.append((-3.14, 3.14)) # Default unlimited?

        # Stack tensors for batch processing
        self.fixed_transforms_stack = torch.stack(self.fixed_transforms) # (N, 4, 4)
        self.joint_axes_stack = torch.stack(self.joint_axes) # (N, 3)
        
        lo = [lo for (lo, hi) in self.limits]
        hi = [hi for (lo, hi) in self.limits]
        self.lower = to_torch(lo, self.device, self.dtype)
        self.upper = to_torch(hi, self.device, self.dtype)

    def clamp(self, q):
        return torch.max(torch.min(q, self.upper), self.lower)

    def fk_Ts(self, q_batch):
        """
        Compute Forward Kinematics
        q_batch: (B, N) tensor of joint angles
        Returns: List of transform matrices [T_base, T_1, ..., T_N] relative to base
        """
        q_batch = to_torch(q_batch, self.device, self.dtype)
        B, N = q_batch.shape
        assert N == self.N
        
        Ts = []
        T_curr = torch.eye(4, dtype=self.dtype, device=self.device).expand(B, 4, 4).clone()
        Ts.append(T_curr.clone()) # Base frame (Identity)
        
        for i in range(self.N):
            # T_fixed (parent -> child_pre_rot)
            T_f = self.fixed_transforms_stack[i] # (4, 4)
            
            # Joint rotation
            axis = self.joint_axes_stack[i] # (3,)
            theta = q_batch[:, i] # (B,)
            
            # Axis-angle rotation matrix
            # R = I + sin(th)*K + (1-cos(th))*K^2
            kx, ky, kz = axis[0], axis[1], axis[2]
            K = torch.zeros(3, 3, device=self.device, dtype=self.dtype)
            K[0, 1] = -kz; K[0, 2] = ky
            K[1, 0] = kz;  K[1, 2] = -kx
            K[2, 0] = -ky; K[2, 1] = kx
            
            K = K.unsqueeze(0) # (1, 3, 3)
            I = torch.eye(3, device=self.device, dtype=self.dtype).unsqueeze(0)
            
            sin_th = torch.sin(theta).view(B, 1, 1)
            cos_th = torch.cos(theta).view(B, 1, 1)
            
            R_rot = I + sin_th * K + (1.0 - cos_th) * (K @ K)
            
            # Construct T_rot
            T_rot = torch.eye(4, device=self.device, dtype=self.dtype).expand(B, 4, 4).clone()
            T_rot[:, :3, :3] = R_rot
            
            # T_i = T_fixed @ T_rot
            T_i = T_f.unsqueeze(0) @ T_rot
            
            # Accumulate
            T_curr = T_curr @ T_i
            Ts.append(T_curr.clone())
            
        return Ts

    def fk_points_axes(self, q_batch, return_R_last=False):
        Ts = self.fk_Ts(q_batch)
        pts = torch.stack([T[:, :3, 3] for T in Ts], dim=1)
        
        # Calculate global axes
        z_axes_list = []
        for i in range(self.N):
            T_parent = Ts[i]
            T_f = self.fixed_transforms_stack[i].unsqueeze(0) # (1, 4, 4)
            T_pivot = T_parent @ T_f
            
            R_pivot = T_pivot[:, :3, :3]
            axis_local = self.joint_axes_stack[i].unsqueeze(0).unsqueeze(-1) # (1, 3, 1)
            axis_global = R_pivot @ axis_local
            z_axes_list.append(axis_global.squeeze(-1))
            
        z_axes = torch.stack(z_axes_list, dim=1)
        
        if return_R_last:
            R = Ts[-1][:, :3, :3]
            return pts, z_axes, R
        return pts, z_axes


class PoseFABRIKBatch:
    def __init__(self, kin: URDFKinematicsTorch):
        self.kin = kin
        self.N = kin.N
        self.max_iter_fabrik = 5
        self.tol_pos = 1e-3
        self.use_plane = False
        self.max_iter_ori = 1
        self.tol_ori = math.radians(2.0)
        self.ori_gain = 1.0
        self.max_rounds = 8
        self._prev_plane_n: Optional[torch.Tensor] = None

    @torch.no_grad()
    def warmup(self, T_target_n1: torch.Tensor, q0: torch.Tensor, rounds: int = 1):
        q = q0.clone()
        target_pos = T_target_n1[:, :3, 3]
        R_target = T_target_n1[:, :3, :3]

        for _ in range(max(1, int(rounds))):
            P_tar = self._fabrik_positions(q, target_pos)
            q = self._align_positions(q, P_tar)
            q = self._align_orientation(q, R_target)
            _ = self.kin.fk_points_axes(q, return_R_last=True)

        if q.is_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def _place(prev, curr, L):
        d = torch.linalg.norm(curr - prev, dim=-1, keepdim=True).clamp(min=1e-12)
        return prev + (curr - prev) * (L / d)

    @staticmethod
    def _project_plane(p, n, p0):
        n = normalize(n)
        return p - torch.sum((p - p0) * n, dim=-1, keepdim=True) * n

    def _plane_normal(self, q, pts):
        Pcur, z_axes = self.kin.fk_points_axes(q)
        a1 = normalize(z_axes[:, 0, :])
        base = Pcur[:, 0, :]

        K = torch.arange(2, self.N + 1, device=pts.device)
        v = pts[:, K, :] - base.unsqueeze(1)
        a1e = a1.unsqueeze(1)
        dot = torch.sum(a1e * v, dim=-1, keepdim=True)
        v_perp = v - a1e * dot
        k_rel = torch.argmax(torch.linalg.norm(v_perp, dim=-1), dim=-1)
        Bidx = torch.arange(q.shape[0], device=q.device)
        v_best = v_perp[Bidx, k_rel, :]

        small = torch.linalg.norm(v_best, dim=-1) < 1e-9
        fallback = torch.where(
            (torch.abs(a1[:, 0]) < 0.9).unsqueeze(-1),
            torch.tensor([1.0, 0.0, 0.0], device=a1.device).expand_as(a1),
            torch.tensor([0.0, 1.0, 0.0], device=a1.device).expand_as(a1),
        )
        v_best = torch.where(small.unsqueeze(-1), fallback, v_best)

        nrm_vec = normalize(torch.cross(a1, v_best, dim=-1))

        if self._prev_plane_n is None:
            self._prev_plane_n = nrm_vec
        else:
            self._prev_plane_n = normalize(0.8 * self._prev_plane_n + 0.2 * nrm_vec)
        return self._prev_plane_n, base

    @torch.no_grad()
    def _fabrik_positions(self, q, target_pos):
        P0, _ = self.kin.fk_points_axes(q)
        L = torch.linalg.norm(P0[:, 1:] - P0[:, :-1], dim=-1)
        pts = P0.clone()
        p0 = pts[:, 0, :].clone()

        for _ in range(self.max_iter_fabrik):
            pts[:, -1, :] = target_pos

            if self.use_plane:
                n_plane, p_plane = self._plane_normal(q, pts)

            for i in range(self.N - 1, 0, -1):
                curr = pts[:, i, :]
                if self.use_plane and (1 < i < self.N - 1):
                    curr = self._project_plane(curr, n_plane, p_plane)
                pts[:, i, :] = self._place(pts[:, i + 1, :], curr, L[:, i].unsqueeze(-1))

            pts[:, 0, :] = p0

            if self.use_plane:
                n_plane, p_plane = self._plane_normal(q, pts)

            for i in range(self.N):
                nxt = pts[:, i + 1, :]
                if self.use_plane and (1 < (i + 1) < self.N - 1):
                    nxt = self._project_plane(nxt, n_plane, p_plane)
                pts[:, i + 1, :] = self._place(pts[:, i, :], nxt, L[:, i].unsqueeze(-1))

            err = torch.linalg.norm(pts[:, -1, :] - target_pos, dim=-1)
            if torch.max(err) < self.tol_pos:
                break

        return pts

    @torch.no_grad()
    def _align_positions(self, q, P_target):
        for _ in range(3):
            P_cur, z_axes = self.kin.fk_points_axes(q)
            changed = torch.zeros(q.shape[0], dtype=torch.bool, device=q.device)
            for i in range(1, self.N):
                a = normalize(z_axes[:, i - 1, :])
                base = P_cur[:, i, :]

                v = P_target[:, i + 1 :, :] - base.unsqueeze(1)
                a_exp = a.unsqueeze(1)
                v_perp = v - a_exp * torch.sum(a_exp * v, dim=-1, keepdim=True)

                k_rel = torch.argmax(torch.linalg.norm(v_perp, dim=-1), dim=-1)
                Bidx = torch.arange(q.shape[0], device=q.device)
                k = k_rel + (i + 1)

                p_i = P_cur[:, i, :]
                p_k = P_cur[Bidx, k, :]
                r_cur = p_k - p_i

                t_k = P_target[Bidx, k, :]
                r_tgt = t_k - P_target[:, i, :]

                r_p = r_cur - a * torch.sum(a * r_cur, dim=-1, keepdim=True)
                t_p = r_tgt - a * torch.sum(a * r_tgt, dim=-1, keepdim=True)

                dot = torch.sum(r_p * t_p, dim=-1)
                crs = torch.sum(a * torch.cross(r_p, t_p, dim=-1), dim=-1)
                th = torch.atan2(crs, dot)

                valid = (torch.linalg.norm(r_p, dim=-1) > 1e-10) & (torch.linalg.norm(t_p, dim=-1) > 1e-10)
                th = torch.where(valid, th, torch.zeros_like(th))

                q[:, i - 1] += th
                q[:, i - 1] = torch.clamp(q[:, i - 1], self.kin.lower[i - 1], self.kin.upper[i - 1])
                changed |= (torch.abs(th) > 1e-6)
            if not changed.any():
                break
        return q

    @torch.no_grad()
    def _align_orientation(self, q, R_target):
        for _ in range(self.max_iter_ori):
            _, z_axes, R_cur = self.kin.fk_points_axes(q, return_R_last=True)
            Re = R_target @ R_cur.transpose(1, 2)
            w = so3_log(Re)
            ang = torch.linalg.norm(w, dim=-1)
            if torch.max(ang) < self.tol_ori:
                break

            for i in range(self.N, 0, -1):
                zi = normalize(z_axes[:, i - 1, :])
                dtheta = self.ori_gain * torch.sum(zi * w, dim=-1)
                q[:, i - 1] += dtheta
                q[:, i - 1] = torch.clamp(q[:, i - 1], self.kin.lower[i - 1], self.kin.upper[i - 1])

        return q

    @torch.no_grad()
    def solve_batch(self, T_target_n1: torch.Tensor, q0: torch.Tensor):
        q = q0.clone()
        B = q.shape[0]
        target_pos = T_target_n1[:, :3, 3]
        R_target = T_target_n1[:, :3, :3]

        best_q = q.clone()
        best_pos = torch.full((B,), float('inf'), dtype=q.dtype, device=q.device)
        best_ori = torch.full((B,), float('inf'), dtype=q.dtype, device=q.device)

        solved_mask = torch.zeros(B, dtype=torch.bool, device=q.device)
        per_seed_time_ms = torch.full((B,), float("nan"), dtype=torch.float32, device=q.device)
        per_seed_outer_iters = torch.full((B,), -1, dtype=torch.int32, device=q.device)

        use_cuda_timer = q.is_cuda and torch.cuda.is_available()

        if use_cuda_timer:
            def _now():
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                return ev
            def _elapsed_ms(t0):
                ev1 = torch.cuda.Event(enable_timing=True)
                ev1.record()
                torch.cuda.synchronize()
                return float(t0.elapsed_time(ev1))
            torch.cuda.synchronize()
        else:
            def _now():
                return time.perf_counter()
            def _elapsed_ms(t0):
                return float((time.perf_counter() - t0) * 1000.0)

        _total_t0 = _now()

        if use_cuda_timer:
            _seed_start_evt = torch.cuda.Event(enable_timing=True)
            _seed_start_evt.record()
        else:
            _seed_wall_start = time.perf_counter()

        tb_total = {"fabrik_ms": 0.0, "align_pos_ms": 0.0, "align_ori_ms": 0.0, "fk_eval_ms": 0.0, "bookkeep_ms": 0.0}
        per_round = []

        for r in range(self.max_rounds):
            t0 = _now()
            P_tar = self._fabrik_positions(q, target_pos)
            fabrik_ms = _elapsed_ms(t0); tb_total["fabrik_ms"] += fabrik_ms

            t0 = _now()
            q = self._align_positions(q, P_tar)
            align_pos_ms = _elapsed_ms(t0); tb_total["align_pos_ms"] += align_pos_ms

            t0 = _now()
            q = self._align_orientation(q, R_target)
            align_ori_ms = _elapsed_ms(t0); tb_total["align_ori_ms"] += align_ori_ms

            t0 = _now()
            P_cur, _, R_cur = self.kin.fk_points_axes(q, return_R_last=True)
            pos_err = torch.linalg.norm(P_cur[:, -1, :] - target_pos, dim=-1)
            Re = R_target @ R_cur.transpose(1, 2)
            ori_err = torch.linalg.norm(so3_log(Re), dim=-1)
            fk_eval_ms = _elapsed_ms(t0); tb_total["fk_eval_ms"] += fk_eval_ms

            t0 = _now()
            improved = (pos_err < best_pos) | ((pos_err <= best_pos + 1e-12) & (ori_err < best_ori))
            best_q[improved] = q[improved]
            best_pos[improved] = pos_err[improved]
            best_ori[improved] = ori_err[improved]

            just_ok = (pos_err < self.tol_pos) & (ori_err < self.tol_ori) & (~solved_mask)
            if just_ok.any():
                if use_cuda_timer:
                    _evt = torch.cuda.Event(enable_timing=True)
                    _evt.record()
                    torch.cuda.synchronize()
                    elapsed_ms = _seed_start_evt.elapsed_time(_evt)
                else:
                    elapsed_ms = (time.perf_counter() - _seed_wall_start) * 1000.0
                per_seed_time_ms[just_ok] = float(elapsed_ms)
                per_seed_outer_iters[just_ok] = int(r + 1)
                solved_mask |= just_ok

            per_round.append({
                "round": int(r + 1),
                "fabrik_ms": float(fabrik_ms),
                "align_pos_ms": float(align_pos_ms),
                "align_ori_ms": float(align_ori_ms),
                "fk_eval_ms": float(fk_eval_ms),
                "pos_err_max": float(pos_err.max().item()),
                "ori_err_max": float(ori_err.max().item()),
            })
            bookkeep_ms = _elapsed_ms(t0); tb_total["bookkeep_ms"] += bookkeep_ms

            if solved_mask.all() or ((pos_err.max() < self.tol_pos) and (ori_err.max() < self.tol_ori)):
                break

        total_ms = _elapsed_ms(_total_t0)

        success = (best_pos < self.tol_pos) & (best_ori < self.tol_ori)
        info = {
            "mean_pos_err": float(best_pos.mean().item()),
            "mean_ori_err": float(best_ori.mean().item()),
            "solve_time_ms_per_seed": per_seed_time_ms.detach().cpu().numpy(),
            "outer_iters_to_solve": per_seed_outer_iters.detach().cpu().numpy(),
            "time_breakdown_ms": {
                "total_ms": float(total_ms),
                "fabrik_ms": float(tb_total["fabrik_ms"]),
                "align_pos_ms": float(tb_total["align_pos_ms"]),
                "align_ori_ms": float(tb_total["align_ori_ms"]),
                "fk_eval_ms": float(tb_total["fk_eval_ms"]),
                "bookkeep_ms": float(tb_total["bookkeep_ms"]),
                "per_round": per_round,
                "device": "cuda" if use_cuda_timer else "cpu",
            },
        }
        return best_q, success, info


def wrist_target_from_ee_target_np(T_ee, tool_len):
    Tn = np.eye(4)
    Tn[:3, :3] = T_ee[:3, :3]
    # Subtract tool offset to find wrist position
    Tn[:3, 3] = T_ee[:3, 3] - T_ee[:3, :3] @ np.array([0, 0, tool_len], float)
    return Tn


def run_parallel_seeds(
    urdf_path: str,
    link_names: List[str],
    T_ee_target: np.ndarray,
    num_seeds: int = 1024,
    include_q0: bool = True,
    q0: Optional[np.ndarray] = None,
    tool_len: float = 0.0,
    w_pos: float = 1.0,
    w_ori: float = 1.0,
    tol_pos_n1: float = 1e-2,
    tol_ang_n1_deg: float = 5.0,
    device=None,
    dtype=torch.float64,
    *,
    do_warmup: bool = True,
    warmup_rounds: int = 1,
):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    
    # Initialize URDF Kinematics
    kin_full = URDFKinematicsTorch(urdf_path, link_names, device=device, dtype=dtype)
    N = kin_full.N
    
    link_names_sub = link_names[:-1]
    kin_sub = URDFKinematicsTorch(urdf_path, link_names_sub, device=device, dtype=dtype)
    
    rng = np.random.default_rng()
    seeds = []
    if include_q0 and (q0 is not None):
        seeds.append(np.asarray(q0, float))
        
    joint_limits_np = kin_full.limits
    joint_limits_np = np.array(joint_limits_np, float)
    
    lo, hi = joint_limits_np[:, 0], joint_limits_np[:, 1]
    if num_seeds > (1 if include_q0 and (q0 is not None) else 0):
        u = rng.random((num_seeds - len(seeds), N))
        seeds_rand = lo + u * (hi - lo)
        seeds.extend(seeds_rand)
    seeds = np.asarray(seeds, float)
    B = seeds.shape[0]

    Tn_target = wrist_target_from_ee_target_np(T_ee_target, tool_len)
    
    T_f_N = kin_full.fixed_transforms_stack[-1].cpu().numpy() # Last joint fixed transform
    axis_N = kin_full.joint_axes_stack[-1].cpu().numpy()
    
    Tn1_targets = []
    for b in range(B):
        qn = float(seeds[b, N - 1])
        
        # Calculate T_rot_N(qn) numpy
        kx, ky, kz = axis_N
        ct, st = math.cos(qn), math.sin(qn)
        R_rot = np.array([
            [ct + kx**2*(1-ct),    kx*ky*(1-ct) - kz*st, kx*kz*(1-ct) + ky*st],
            [ky*kx*(1-ct) + kz*st, ct + ky**2*(1-ct),    ky*kz*(1-ct) - kx*st],
            [kz*kx*(1-ct) - ky*st, kz*ky*(1-ct) + kx*st, ct + kz**2*(1-ct)]
        ])
        # T_LinkN = T_Link(N-1) @ T_fixed @ T_rot
        # Tn1 = Tn_target @ inv(T_fixed @ T_rot)
        
        T_inv = np.linalg.inv(T_f_N @ np.eye(4)) # Wait, T_rot should be part of it.
        # Actually T_Link(N-1) is frame BEFORE joint N rotation and BEFORE fixed transform of Joint N.
        # Joint N connects Link N-1 -> Link N.
        # T_LinkN = T_Link(N-1) * T_fixed_N * T_rot_N
        
        T_rot_mat = np.eye(4)
        T_rot_mat[:3, :3] = R_rot
        
        T_inv = np.linalg.inv(T_f_N @ T_rot_mat)
        Tn1 = Tn_target @ T_inv
        Tn1_targets.append(Tn1)
        
    Tn1_targets_t = torch.from_numpy(np.stack(Tn1_targets, 0)).to(device=device, dtype=dtype)

    solver = PoseFABRIKBatch(kin_sub)
    solver.tol_pos = float(tol_pos_n1)
    solver.tol_ori = math.radians(float(tol_ang_n1_deg))

    q0_sub = seeds[:, : N - 1]
    q0_sub_t = torch.from_numpy(q0_sub).to(device=device, dtype=dtype)

    if do_warmup:
        solver.warmup(Tn1_targets_t, q0_sub_t, rounds=warmup_rounds)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    q_sol_sub, success, info = solver.solve_batch(Tn1_targets_t, q0_sub_t)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000.0

    qn_t = torch.from_numpy(seeds[:, N - 1]).to(device=device, dtype=dtype).unsqueeze(1)
    q_full_t = torch.cat([q_sol_sub, qn_t], dim=1)

    Ts = kin_full.fk_Ts(q_full_t)
    R_cur = Ts[-1][:, :3, :3]
    p_cur = Ts[-1][:, :3, 3]
    R_tgt = torch.from_numpy(T_ee_target[:3, :3]).to(device=device, dtype=dtype).unsqueeze(0).expand(B, 3, 3).contiguous()
    p_tgt = torch.from_numpy(T_ee_target[:3, 3]).to(device=device, dtype=dtype).unsqueeze(0).expand(B, 3).contiguous()

    pos_err = torch.linalg.norm(p_cur - p_tgt, dim=-1)
    Re = R_tgt @ R_cur.transpose(1, 2)
    ang_err = torch.linalg.norm(so3_log(Re), dim=-1)
    score = w_pos * pos_err + w_ori * ang_err

    time_ms_arr = np.asarray(info.get("solve_time_ms_per_seed", np.full((B,), np.nan, float)), float)
    outer_it_arr = np.asarray(info.get("outer_iters_to_solve", np.full((B,), -1, int)), int)

    pos_err_np = pos_err.detach().cpu().numpy()
    ang_err_np = ang_err.detach().cpu().numpy()
    score_np = score.detach().cpu().numpy()
    q_full_np = q_full_t.detach().cpu().numpy()
    success_np = success.detach().cpu().numpy().astype(bool)

    order = np.argsort(score_np)
    results = []
    for i in order:
        results.append(
            dict(
                q_full=q_full_np[i].copy(),
                ok=bool(success_np[i]),
                err_pos=float(pos_err_np[i]),
                err_ang_rad=float(ang_err_np[i]),
                score=float(score_np[i]),
                solve_time_ms=float(time_ms_arr[i]),
                outer_iters_to_solve=int(outer_it_arr[i]),
            )
        )

    out_info = dict(
        mean_subchain_pos_err=float(info.get("mean_pos_err", 0.0)),
        mean_subchain_ori_err=float(info.get("mean_ori_err", 0.0)),
        batch=B,
        device=str(device),
        measured_batch_ms=float(elapsed),
    )
    return results, out_info

def build_default_robot_urdf_path() -> str:
    try:
        share_dir = get_package_share_directory('piper_description')
        return os.path.join(share_dir, 'urdf', 'piper_description.urdf')
    except Exception:
        # Fallback for local testing if not sourced properly (though highly unlikely in ROS env)
        return "/home/daumpark/rop_ws/src/piper_description/urdf/piper_description.urdf"

def get_default_link_names() -> List[str]:
    # Chain: Base -> Link1 -> Link2 -> ... -> Link6
    # URDF link names
    return ["base_link", "link1", "link2", "link3", "link4", "link5", "link6"]