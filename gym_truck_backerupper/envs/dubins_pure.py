"""Pure-Python Dubins path planner."""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple


def _mod2pi(theta: float) -> float:
    return theta - 2.0 * math.pi * math.floor(theta / (2.0 * math.pi))


def _LSL(alpha: float, beta: float, d: float):
    sa, sb = math.sin(alpha), math.sin(beta)
    ca, cb = math.cos(alpha), math.cos(beta)
    c_ab = math.cos(alpha - beta)
    tmp0 = d + sa - sb
    p2_squared = 2 + d * d - 2 * c_ab + 2 * d * (sa - sb)
    if p2_squared < 0:
        return None
    tmp1 = math.atan2(cb - ca, tmp0)
    t = _mod2pi(-alpha + tmp1)
    p = math.sqrt(p2_squared)
    q = _mod2pi(beta - tmp1)
    return t, p, q


def _RSR(alpha: float, beta: float, d: float):
    sa, sb = math.sin(alpha), math.sin(beta)
    ca, cb = math.cos(alpha), math.cos(beta)
    c_ab = math.cos(alpha - beta)
    tmp0 = d - sa + sb
    p2_squared = 2 + d * d - 2 * c_ab + 2 * d * (sb - sa)
    if p2_squared < 0:
        return None
    tmp1 = math.atan2(ca - cb, tmp0)
    t = _mod2pi(alpha - tmp1)
    p = math.sqrt(p2_squared)
    q = _mod2pi(-beta + tmp1)
    return t, p, q


def _LSR(alpha: float, beta: float, d: float):
    sa, sb = math.sin(alpha), math.sin(beta)
    ca, cb = math.cos(alpha), math.cos(beta)
    c_ab = math.cos(alpha - beta)
    p2_squared = -2 + d * d + 2 * c_ab + 2 * d * (sa + sb)
    if p2_squared < 0:
        return None
    p2 = math.sqrt(p2_squared)
    tmp2 = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p2)
    t = _mod2pi(-alpha + tmp2)
    p = p2
    q = _mod2pi(-beta + tmp2)
    return t, p, q


def _RSL(alpha: float, beta: float, d: float):
    sa, sb = math.sin(alpha), math.sin(beta)
    ca, cb = math.cos(alpha), math.cos(beta)
    c_ab = math.cos(alpha - beta)
    p2_squared = -2 + d * d + 2 * c_ab - 2 * d * (sa + sb)
    if p2_squared < 0:
        return None
    p2 = math.sqrt(p2_squared)
    tmp2 = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, p2)
    t = _mod2pi(alpha - tmp2)
    p = p2
    q = _mod2pi(beta - tmp2)
    return t, p, q


_PATHS = {
    "LSL": (_LSL, ("L", "S", "L")),
    "RSR": (_RSR, ("R", "S", "R")),
    "LSR": (_LSR, ("L", "S", "R")),
    "RSL": (_RSL, ("R", "S", "L")),
}


class DubinsPath:
    def __init__(self, q0: Sequence[float], rho: float, types: Tuple[str, str, str], lengths: Tuple[float, float, float]):
        self.q0 = list(q0)
        self.rho = rho
        self.types = types
        self.segment_lengths = lengths
        self.path_length = sum(lengths) * rho

    def sample_many(self, step_size: float):
        if step_size <= 0:
            raise ValueError("step_size must be > 0")
        n = max(int(self.path_length // step_size) + 1, 2)
        dists = [i * step_size for i in range(n)]
        if dists[-1] < self.path_length:
            dists.append(self.path_length)
        qs = [self._sample(d) for d in dists]
        return qs, dists

    def _sample(self, dist: float):
        t = dist / self.rho
        x, y, th = self.q0[0], self.q0[1], self.q0[2]
        remaining = t
        for seg_type, seg_len in zip(self.types, self.segment_lengths):
            if remaining <= 1e-12:
                break
            take = min(remaining, seg_len)
            x, y, th = self._propagate(x, y, th, seg_type, take)
            remaining -= take
        return [x, y, _mod2pi(th)]

    def _propagate(self, x: float, y: float, th: float, seg_type: str, length: float):
        if seg_type == "S":
            return (
                x + self.rho * length * math.cos(th),
                y + self.rho * length * math.sin(th),
                th,
            )
        if seg_type == "L":
            th2 = th + length
            return (
                x + self.rho * (math.sin(th2) - math.sin(th)),
                y + self.rho * (-math.cos(th2) + math.cos(th)),
                th2,
            )
        if seg_type == "R":
            th2 = th - length
            return (
                x + self.rho * (-math.sin(th2) + math.sin(th)),
                y + self.rho * (math.cos(th2) - math.cos(th)),
                th2,
            )
        raise ValueError(seg_type)


def shortest_path(q0: Sequence[float], q1: Sequence[float], turning_radius: float) -> DubinsPath:
    if turning_radius <= 0:
        raise ValueError("turning_radius must be > 0")
    dx = q1[0] - q0[0]
    dy = q1[1] - q0[1]
    d = math.hypot(dx, dy) / turning_radius
    theta = math.atan2(dy, dx)
    alpha = _mod2pi(q0[2] - theta)
    beta = _mod2pi(q1[2] - theta)

    best = None
    best_cost = float("inf")
    best_types = None
    for _name, (fn, types) in _PATHS.items():
        res = fn(alpha, beta, d)
        if res is None:
            continue
        cost = sum(res)
        if cost < best_cost:
            best_cost = cost
            best = res
            best_types = types
    if best is None:
        # Fallback: almost-straight LSL with tiny arcs
        best = (0.0, d, 0.0)
        best_types = ("L", "S", "L")
    return DubinsPath(q0, turning_radius, best_types, best)
