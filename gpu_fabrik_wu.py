# -*- coding: utf-8 -*-
"""
sample_fabrik_pose.py — GPU 병렬(FABRIK per-sample) 버전

- 여러 초기 포즈(q seeds)를 샘플링하여 '각 샘플별' FABRIK을 순차가 아닌
  PyTorch 텐서 배치로 동시에 수행합니다.
- 병렬화/배치 FK 및 정렬 패턴은 gpu_wrist.py의 PyTorch 기반 구현을 참고했습니다.  # ref: gpu_wrist.py
"""

import math
from typing import List, Tuple

import time
import numpy as np
import torch
import matplotlib.pyplot as plt


# =========================
# Torch helpers (ref: gpu_wrist.py)
# =========================
def to_torch(x, device=None, dtype=torch.float64):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)

def normalize(v, eps=1e-12):
    n = torch.linalg.norm(v, dim=-1, keepdim=True).clamp(min=eps)
    return v / n

def so3_log(R):
    """
    batched matrix log for SO(3)
    R: (B,3,3) -> w: (B,3), ||w|| = angle
    """
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


# =========================
# MDH (Craig-style) in Torch (ref: gpu_wrist.py)
# =========================
def mdh_transform_torch(alpha, a, d, theta):
    """
    MDH transform (batched theta)
    alpha,a,d: scalars; theta: (B,)
    returns T: (B,4,4)
    """
    ca = torch.cos(alpha); sa = torch.sin(alpha)
    ct = torch.cos(theta); st = torch.sin(theta)
    B = theta.shape[0]
    T = torch.zeros(B, 4, 4, dtype=theta.dtype, device=theta.device)
    T[:, 0, 0] = ct
    T[:, 0, 1] = -st
    T[:, 0, 3] = a

    T[:, 1, 0] = st * ca
    T[:, 1, 1] = ct * ca
    T[:, 1, 2] = -sa
    T[:, 1, 3] = -sa * d

    T[:, 2, 0] = st * sa
    T[:, 2, 1] = ct * sa
    T[:, 2, 2] = ca
    T[:, 2, 3] = ca * d

    T[:, 3, 3] = 1.0
    return T

class MDHKinematicsTorch:
    """
    Revolute-only, batched FK
    """
    def __init__(self, mdh_params, joint_limits, device=None, dtype=torch.float64):
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.dtype = dtype
        self.mdh = to_torch(mdh_params, self.device, self.dtype)  # (N,4) alpha,a,d,theta0
        self.N = self.mdh.shape[0]
        lo = [lo for (lo, hi) in joint_limits]
        hi = [hi for (lo, hi) in joint_limits]
        self.lower = to_torch(lo, self.device, self.dtype)
        self.upper = to_torch(hi, self.device, self.dtype)

    def clamp(self, q):
        return torch.max(torch.min(q, self.upper), self.lower)

    def fk_Ts(self, q_batch):
        """
        q_batch: (B,N)
        return: list of T_0_i (i=0..N), each (B,4,4)
        """
        q_batch = to_torch(q_batch, self.device, self.dtype)
        B, N = q_batch.shape
        Ts = []
        T = torch.eye(4, dtype=self.dtype, device=self.device).expand(B,4,4).clone()
        Ts.append(T.clone())
        for i in range(self.N):
            alpha, a, d, th0 = self.mdh[i]
            theta = th0 + q_batch[:, i]
            T_i = mdh_transform_torch(alpha, a, d, theta)
            T = T @ T_i
            Ts.append(T.clone())
        return Ts

    def fk_points_axes(self, q_batch, return_R_last=False):
        Ts = self.fk_Ts(q_batch)
        pts = torch.stack([T[:, :3, 3] for T in Ts], dim=1)        # (B,N+1,3)
        z_axes = torch.stack([Ts[i+1][:, :3, 2] for i in range(self.N)], dim=1)  # (B,N,3)
        if return_R_last:
            R = Ts[-1][:, :3, :3]
            return pts, z_axes, R
        return pts, z_axes


# =========================
# Pose-aware FABRIK (Batch) — ref: gpu_wrist.py
# (subchain(1..n-1)의 끝 프레임을 위치+자세로 맞추는 배치 솔버)
# =========================
class PoseFABRIKBatch:
    def __init__(self, kin: MDHKinematicsTorch):
        self.kin = kin
        self.N = kin.N
        # FABRIK (positions)
        self.max_iter_fabrik = 5
        self.tol_pos = 1e-3   # 1 mm
        self.use_plane = False
        # Orientation align
        self.max_iter_ori = 1
        self.tol_ori = math.radians(2.0)  # 2 deg
        self.ori_gain = 1.0
        # Outer
        self.max_rounds = 8
        self._prev_plane_n = None  # (B,3)

    @torch.no_grad()
    def warmup(self, T_target_n1: torch.Tensor, q0: torch.Tensor, rounds: int = 1):
        """
        CUDA 컨텍스트/커널/라이브러리 로딩을 미리 끝내기 위한 더미 패스.
        - 입력과 같은 shape/device/dtype로 실행하는 것이 핵심
        - 외부 상태/출력에 영향 없음(결과는 버림)
        """
        q = q0.clone()
        target_pos = T_target_n1[:, :3, 3]
        R_target = T_target_n1[:, :3, :3]

        for _ in range(max(1, int(rounds))):
            P_tar = self._fabrik_positions(q, target_pos)   # 위치 FABRIK
            q = self._align_positions(q, P_tar)             # 위치 정렬
            q = self._align_orientation(q, R_target)        # 자세 정렬
            _ = self.kin.fk_points_axes(q, return_R_last=True)  # FK도 한 번 태우기

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

        K = torch.arange(2, self.N+1, device=pts.device)
        v = pts[:, K, :] - base.unsqueeze(1)
        a1e = a1.unsqueeze(1)
        dot = torch.sum(a1e * v, dim=-1, keepdim=True)
        v_perp = v - a1e * dot
        nrm = torch.linalg.norm(v_perp, dim=-1)
        k_rel = torch.argmax(nrm, dim=-1)
        Bidx = torch.arange(q.shape[0], device=q.device)
        v_best = v_perp[Bidx, k_rel, :]

        small = torch.linalg.norm(v_best, dim=-1) < 1e-9
        fallback = torch.where(
            (torch.abs(a1[:, 0]) < 0.9).unsqueeze(-1),
            torch.tensor([1.,0.,0.], device=a1.device).expand_as(a1),
            torch.tensor([0.,1.,0.], device=a1.device).expand_as(a1)
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
        """
        q: (B,N), target_pos: (B,3)
        return P (joint positions) that reach target_pos at the end.
        """
        P0, _ = self.kin.fk_points_axes(q)
        L = torch.linalg.norm(P0[:,1:] - P0[:,:-1], dim=-1)  # (B,N)
        pts = P0.clone()
        p0 = pts[:, 0, :].clone()

        for _ in range(self.max_iter_fabrik):
            pts[:, -1, :] = target_pos

            if self.use_plane:
                n_plane, p_plane = self._plane_normal(q, pts)

            # forward pass
            for i in range(self.N-1, 0, -1):
                curr = pts[:, i, :]
                if self.use_plane and (1 < i < self.N-1):
                    curr = self._project_plane(curr, n_plane, p_plane)
                pts[:, i, :] = self._place(pts[:, i+1, :], curr, L[:, i].unsqueeze(-1))

            # root lock
            pts[:, 0, :] = p0

            # backward pass
            if self.use_plane:
                n_plane, p_plane = self._plane_normal(q, pts)

            for i in range(self.N):
                nxt = pts[:, i+1, :]
                if self.use_plane and (1 < (i+1) < self.N-1):
                    nxt = self._project_plane(nxt, n_plane, p_plane)
                pts[:, i+1, :] = self._place(pts[:, i, :], nxt, L[:, i].unsqueeze(-1))

            err = torch.linalg.norm(pts[:, -1, :] - target_pos, dim=-1)
            if torch.max(err) < self.tol_pos:
                break

        return pts

    @torch.no_grad()
    def _align_positions(self, q, P_target):
        """
        Jacobi-like angle updates to align joint positions to P_target.
        """
        for _ in range(3):
            P_cur, z_axes = self.kin.fk_points_axes(q)
            changed = torch.zeros(q.shape[0], dtype=torch.bool, device=q.device)
            for i in range(1, self.N):
                a = normalize(z_axes[:, i-1, :])
                base = P_cur[:, i, :]

                v = P_target[:, i+1:, :] - base.unsqueeze(1)     # (B,N-i,3)
                a_exp = a.unsqueeze(1)
                v_perp = v - a_exp * torch.sum(a_exp * v, dim=-1, keepdim=True)

                k_rel = torch.argmax(torch.linalg.norm(v_perp, dim=-1), dim=-1)
                Bidx = torch.arange(q.shape[0], device=q.device)
                k = k_rel + (i+1)

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

                q[:, i-1] += th
                q[:, i-1] = torch.clamp(q[:, i-1], self.kin.lower[i-1], self.kin.upper[i-1])
                changed |= (torch.abs(th) > 1e-6)
            if not changed.any():
                break
        return q
    @torch.no_grad()

    def _align_orientation(self, q, R_target):
        """
        마지막 프레임의 R을 R_target에 수렴시키는 간단 업데이트 (Jacobian-transpose형)
        """
        for _ in range(self.max_iter_ori):
            _, z_axes, R_cur = self.kin.fk_points_axes(q, return_R_last=True)
            Re = R_target @ R_cur.transpose(1, 2)    # (B,3,3)
            w = so3_log(Re)                           # (B,3)
            ang = torch.linalg.norm(w, dim=-1)
            if torch.max(ang) < self.tol_ori:
                break

            for i in range(self.N, 0, -1):
                zi = normalize(z_axes[:, i-1, :])    # (B,3)
                dtheta = self.ori_gain * torch.sum(zi * w, dim=-1)
                q[:, i-1] += dtheta
                q[:, i-1] = torch.clamp(q[:, i-1], self.kin.lower[i-1], self.kin.upper[i-1])

        return q

    @torch.no_grad()
    def solve_batch(self, T_target_n1: torch.Tensor, q0: torch.Tensor):
        """
        subchain(1..n-1) 배치 솔브
        T_target_n1: (B,4,4)
        q0: (B,N-1)
        return: q_best(B,N-1), success(B,), info(dict: mean errs + per-seed time/iters + time breakdown)
        """
        q = q0.clone()
        B = q.shape[0]
        target_pos = T_target_n1[:, :3, 3]
        R_target = T_target_n1[:, :3, :3]

        best_q = q.clone()
        best_pos = torch.full((B,), float('inf'), dtype=q.dtype, device=q.device)
        best_ori = torch.full((B,), float('inf'), dtype=q.dtype, device=q.device)

        # --- per-seed 완료 시간/라운드 계측 ---
        solved_mask = torch.zeros(B, dtype=torch.bool, device=q.device)
        per_seed_time_ms = torch.full((B,), float("nan"), dtype=torch.float32, device=q.device)
        per_seed_outer_iters = torch.full((B,), -1, dtype=torch.int32, device=q.device)

        use_cuda_timer = q.is_cuda and torch.cuda.is_available()

        # === 타이머 유틸 ===
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
            import time
            def _now():
                return time.perf_counter()
            def _elapsed_ms(t0):
                return float((time.perf_counter() - t0) * 1000.0)

        _total_t0 = _now()

        # seed 완료 시각 기준
        if use_cuda_timer:
            _seed_start_evt = torch.cuda.Event(enable_timing=True)
            _seed_start_evt.record()
        else:
            import time
            _seed_wall_start = time.perf_counter()

        # --- 누적/라운드별 breakdown ---
        tb_total = {"fabrik_ms": 0.0, "align_pos_ms": 0.0, "align_ori_ms": 0.0, "fk_eval_ms": 0.0, "bookkeep_ms": 0.0}
        per_round = []

        for r in range(self.max_rounds):
            # (1) 위치 FABRIK
            t0 = _now()
            P_tar = self._fabrik_positions(q, target_pos)
            fabrik_ms = _elapsed_ms(t0); tb_total["fabrik_ms"] += fabrik_ms

            # (2) joint 각도 위치 정렬
            t0 = _now()
            q = self._align_positions(q, P_tar)
            align_pos_ms = _elapsed_ms(t0); tb_total["align_pos_ms"] += align_pos_ms

            # (3) 자세 정렬
            t0 = _now()
            q = self._align_orientation(q, R_target)
            align_ori_ms = _elapsed_ms(t0); tb_total["align_ori_ms"] += align_ori_ms

            # (4) 에러 평가(FK)
            t0 = _now()
            P_cur, _, R_cur = self.kin.fk_points_axes(q, return_R_last=True)
            pos_err = torch.linalg.norm(P_cur[:, -1, :] - target_pos, dim=-1)
            Re = R_target @ R_cur.transpose(1, 2)
            ori_err = torch.linalg.norm(so3_log(Re), dim=-1)
            fk_eval_ms = _elapsed_ms(t0); tb_total["fk_eval_ms"] += fk_eval_ms

            # (5) best 갱신 + seed별 solve 시각 기록
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
                    import time
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


# =========================
# numpy helpers (ref: gpu_wrist.py)
# =========================
def local_A_i_np(mdh_i, q_i):
    """ numpy MDH A_i(q) (revolute) """
    alpha, a, d0, th0 = [float(x) for x in mdh_i]
    th = th0 + float(q_i)
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(th), math.sin(th)
    T = np.array([
        [ct, -st, 0, a],
        [st*ca, ct*ca, -sa, -sa*d0],
        [st*sa, ct*sa,  ca,  ca*d0],
        [0, 0, 0, 1]
    ], float)
    return T

def se3_inv_np(T):
    R = T[:3,:3]; p = T[:3,3]
    Ti = np.eye(4)
    Ti[:3,:3] = R.T
    Ti[:3,3] = -R.T @ p
    return Ti

def wrist_target_from_ee_target_np(T_ee, tool_len):
    """
    EEF(T_ee) -> wrist(joint n) target T_n (p shift by tool_len * z_ee)
    """
    Tn = np.eye(4)
    Tn[:3,:3] = T_ee[:3,:3]
    Tn[:3,3]  = T_ee[:3,3] + T_ee[:3,:3] @ np.array([0,0,tool_len], float)
    return Tn


# =========================
# Batch pipeline: random seed 병렬 FABRIK
# =========================
def run_parallel_seeds(
    mdh_params: np.ndarray,
    joint_limits: np.ndarray,
    T_ee_target: np.ndarray,
    num_seeds: int = 1024,
    include_q0: bool = True,
    q0: np.ndarray | None = None,
    tool_len: float = 0.0,
    w_pos: float = 1.0,
    w_ori: float = 1.0,
    tol_pos_n1: float = 1e-2,      # subchain end pos tol
    tol_ang_n1_deg: float = 5.0,   # subchain end ori tol
    device=None,
    dtype=torch.float64,
    *,
    do_warmup: bool = True,        # ★ 추가: 실측 전에 워밍업 여부
    warmup_rounds: int = 1,        # ★ 추가: 워밍업 반복 수
):
    """
    - seeds를 배치로 생성하고, 각 seed의 q_n을 이용해 T_{n-1} 타깃을 개별 생성
    - PoseFABRIKBatch(subchain 1..n-1)를 '한 번'의 배치 호출로 해결
    - full FK로 EEF 타깃에 대한 (pos,ori) 평가
    """
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    N = int(mdh_params.shape[0])
    mdh_params = np.asarray(mdh_params, float)
    joint_limits = np.asarray(joint_limits, float)

    # seed들 만들기
    rng = np.random.default_rng()
    seeds = []
    if include_q0 and (q0 is not None):
        seeds.append(np.asarray(q0, float))
    lo, hi = joint_limits[:,0], joint_limits[:,1]
    if num_seeds > (1 if include_q0 and (q0 is not None) else 0):
        u = rng.random((num_seeds - len(seeds), N))
        seeds_rand = lo + u * (hi - lo)
        seeds.extend(seeds_rand)
    seeds = np.asarray(seeds, float)  # (B, N)
    B = seeds.shape[0]

    # torch kinematics
    kin_sub = MDHKinematicsTorch(mdh_params[:N-1], list(map(tuple, joint_limits[:N-1])), device=device, dtype=dtype)
    kin_full = MDHKinematicsTorch(mdh_params,        list(map(tuple, joint_limits)),      device=device, dtype=dtype)

    # subchain target(T_{n-1}) per seed: Tn = T_ee * inv(A_n(q_n^seed))
    Tn_target = wrist_target_from_ee_target_np(T_ee_target, tool_len)
    Tn1_targets = []
    for b in range(B):
        qn = float(seeds[b, N-1])
        An = local_A_i_np(mdh_params[N-1], qn)
        Tn1 = Tn_target @ se3_inv_np(An)
        Tn1_targets.append(Tn1)
    Tn1_targets_t = torch.from_numpy(np.stack(Tn1_targets, 0)).to(device=device, dtype=dtype)  # (B,4,4)

    # subchain FABRIK batch solve
    solver = PoseFABRIKBatch(kin_sub)
    solver.tol_pos = float(tol_pos_n1)
    solver.tol_ori = math.radians(float(tol_ang_n1_deg))

    q0_sub = seeds[:, :N-1]
    q0_sub_t = torch.from_numpy(q0_sub).to(device=device, dtype=dtype)

    # ---------- 워밍업(실측 제외) ----------
    if do_warmup:
        solver.warmup(Tn1_targets_t, q0_sub_t, rounds=warmup_rounds)  # 결과 버림

    # ---------- 여기부터 실측 ----------
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    q_sol_sub, success, info = solver.solve_batch(Tn1_targets_t, q0_sub_t)  # (B,N-1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000.0  # ms

    # full q 구성: subchain 결과 + seed의 q_n 유지
    qn_t = torch.from_numpy(seeds[:, N-1]).to(device=device, dtype=dtype).unsqueeze(1)
    q_full_t = torch.cat([q_sol_sub, qn_t], dim=1)  # (B,N)

    # 평가: EEF vs target (position + orientation)
    Ts = kin_full.fk_Ts(q_full_t)
    R_cur = Ts[-1][:, :3, :3]      # (B,3,3)
    p_cur = Ts[-1][:, :3, 3]       # (B,3)
    R_tgt = torch.from_numpy(T_ee_target[:3,:3]).to(device=device, dtype=dtype).unsqueeze(0).expand(B,3,3).contiguous()
    p_tgt = torch.from_numpy(T_ee_target[:3,3]).to(device=device, dtype=dtype).unsqueeze(0).expand(B,3).contiguous()

    pos_err = torch.linalg.norm(p_cur - p_tgt, dim=-1)                    # (B,)
    Re = R_tgt @ R_cur.transpose(1,2)
    ang_err = torch.linalg.norm(so3_log(Re), dim=-1)                      # (B,)
    score = w_pos * pos_err + w_ori * ang_err

    time_ms_arr = np.asarray(info.get("solve_time_ms_per_seed", np.full((B,), np.nan, float)), float)
    outer_it_arr = np.asarray(info.get("outer_iters_to_solve", np.full((B,), -1, int)), int)

    # CPU로 결과 정리
    pos_err_np = pos_err.detach().cpu().numpy()
    ang_err_np = ang_err.detach().cpu().numpy()
    score_np   = score.detach().cpu().numpy()
    q_full_np  = q_full_t.detach().cpu().numpy()
    success_np = success.detach().cpu().numpy().astype(bool)

    order = np.argsort(score_np)
    results = []
    for i in order:
        results.append(dict(
            q_full=q_full_np[i].copy(),
            ok=bool(success_np[i]),
            err_pos=float(pos_err_np[i]),
            err_ang_rad=float(ang_err_np[i]),
            score=float(score_np[i]),
            solve_time_ms=float(time_ms_arr[i]),
            outer_iters_to_solve=int(outer_it_arr[i]),
        ))

    out_info = dict(
        mean_subchain_pos_err=float(info.get("mean_pos_err", 0.0)),
        mean_subchain_ori_err=float(info.get("mean_ori_err", 0.0)),
        batch=B,
        device=str(device),
        measured_batch_ms=float(elapsed),  # ★ 선택: 배치 전체 실측 시간(ms)
    )
    return results, out_info

# =========================
# 기본 유틸(플롯) — 원래 파일의 간단 버전
# =========================
def set_equal_aspect(ax, pts, pad=1.2):
    if pts.size == 0:
        return
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    c = 0.5 * (mn + mx)
    r = max(1e-3, 0.5 * float(np.max(mx - mn)) * pad)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)

def draw_axes(ax, T, length=0.1, rgb=True, z_only=False):
    p = T[:3, 3]
    R = T[:3, :3]
    x, y, z = R[:, 0], R[:, 1], R[:, 2]
    if not z_only and rgb:
        ax.quiver(p[0], p[1], p[2], x[0], x[1], x[2], length=length, normalize=True, color='r')
        ax.quiver(p[0], p[1], p[2], y[0], y[1], y[2], length=length, normalize=True, color='g')
    ax.quiver(p[0], p[1], p[2], z[0], z[1], z[2], length=length, normalize=True, color='b')

def render_robot_frames(ax, transforms, axis_scale=0.08):
    for i, T in enumerate(transforms):
        if i == 0 or i == len(transforms) - 1:
            draw_axes(ax, T, axis_scale, rgb=True, z_only=False)
        else:
            draw_axes(ax, T, axis_scale, rgb=False, z_only=True)
    pts = np.array([T[:3, 3] for T in transforms])
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], '.-', linewidth=1.2, markersize=3)
    return pts


# =========================
# Demo
# =========================
def build_default_robot():
    """AGILEX PIPER DH PARAMS"""
    a2 = 0.28503; a3 = 0.02198
    th2 = np.deg2rad(-174.22); th3 = np.deg2rad(-100.78)
    d1 = 0.123; d4 = 0.25075; d6 = 0.091
    mdh_params = np.array([
        [0.0,        0.0,   d1,        0.0],
        [-np.pi/2,   0.0,   0.0,       th2],
        [0.0,        a2,    0.0,       th3],
        [np.pi/2,    a3,    d4,        0.0],
        [-np.pi/2,   0.0,   0.0,       0.0],
        [np.pi/2,    0.0,   d6,        0.0],
    ], float)
    joint_limits = np.array([
        (-2.618, 2.618),
        (0.0,    3.14),
        (-2.967, 0.0),
        (-1.745, 1.745),
        (-1.22,  1.22),
        (-2.0944, 2.0944)
    ], float)
    q0 = np.deg2rad([0, 30, -30, 0, 0, 0]).astype(float)

    """FRANKA EMIKA PANDA DH PARAMS"""
    # mdh_params = np.array([
    #     [0.0,        0.0,     0.333,    0.0],
    #     [-np.pi/2,   0.0,     0.0,      0.0],
    #     [np.pi/2,    0.0,     0.316,    0.0],
    #     [np.pi/2,    0.0825,  0.0,      0.0],
    #     [-np.pi/2,  -0.0825,  0.384,    0.0],
    #     [np.pi/2,    0.0,     0.0,      0.0],
    #     [np.pi/2,    0.088,   0.0,      0.0]
    # ], float)
    # joint_limits = np.array([
    #     (-2.8973, 2.8973),  # Joint 1
    #     (-1.7628, 1.7628),  # Joint 2
    #     (-2.8973, 2.8973),  # Joint 3
    #     (-3.0718, -0.0698), # Joint 4
    #     (-2.8973, 2.8973),  # Joint 5
    #     (-0.0175, 3.7525),  # Joint 6
    #     (-2.8973, 2.8973)   # Joint 7
    # ], float)
    # q0 = np.deg2rad([0, 30, -30, 0, 0, 0, 0]).astype(float)
    tool_len = 0.0
    return mdh_params, joint_limits, q0, tool_len

def visualize_best(q_full, mdh_params):
    device = torch.device('cpu')
    kin = MDHKinematicsTorch(mdh_params, [(-10,10)]*mdh_params.shape[0], device=device, dtype=torch.float64)
    Ts = kin.fk_Ts(torch.from_numpy(q_full[None,:]).double())
    T_list = [Ts[i][0].cpu().numpy() for i in range(len(Ts))]

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    pts = render_robot_frames(ax, T_list, axis_scale=0.08)
    set_equal_aspect(ax, pts, pad=1.3)
    ax.view_init(elev=30, azim=45)
    ax.set_title("Best configuration (GPU-parallel seeds)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    mdh_params, joint_limits, q0, tool_len = build_default_robot()

    # 타깃 EEF pose (원본 예제와 동일)
    T_ee_target = np.array([
        [0,  0, 1,  0.20],
        [-1, 0, 0, -0.20],
        [0, -1, 0,  0.40],
        [0,  0, 0,  1.00]
    ], dtype=float)

    # GPU 병렬 seed FABRIK 실행
    results, info = run_parallel_seeds(
        mdh_params=mdh_params,
        joint_limits=joint_limits,
        T_ee_target=T_ee_target,
        num_seeds=1000,         # 병렬 샘플 수
        include_q0=True,
        q0=q0,
        tool_len=tool_len,
        w_pos=1.0,
        w_ori=0.05,
        tol_pos_n1=1e-2,
        tol_ang_n1_deg=5.0,
    )


    if not results:
        print("[WARN] no results.")
    else:
        print(f"[Batch] device={info['device']}, B={info['batch']}, "
              f"subchain mean pos={info['mean_subchain_pos_err']:.3e} m, "
              f"ori={math.degrees(info['mean_subchain_ori_err']):.2f} deg")

        # 상위 10개 표
        top_k = min(10, len(results))
        print("\n== Ranked configs (by w_pos*pos + w_ori*ang[rad]) ==")
        print(" rank | pos_err[m] | ori_err[deg] | score       | solve_ms | outer_it")
        for i in range(top_k):
            r = results[i]
            print(f" {i+1:4d} | {r['err_pos']:10.4e} | {math.degrees(r['err_ang_rad']):10.3f} | "
                f"{r['score']:10.4e} | {r['solve_time_ms']:8.2f} | {r['outer_iters_to_solve']:8d}")

        # 최상 해 시각화
        visualize_best(results[0]["q_full"], mdh_params)
