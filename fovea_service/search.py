# -*- coding: utf-8 -*-
"""fovea_service.search —— 两种（三种）搜索策略，带逐帧回调，供网页可视化。

greedy / lookahead 对应论文原版；hrd 是本项目的小改进（分层 + 覆盖先验 + 验证回退）。
每个策略都通过 on_step() 上报一次“探测”，前端据此画出候选框与分数。
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from fovea_lite.geometry import Box, area, center, clamp, inside, jitter, scale

ProbeFn = Callable[[Box], float]
OnStep = Callable[[Dict], None]


def _prior_mass(box: Box, seed: Box, sigma: float = 0.28, floor: float = 0.08) -> float:
    cx, cy = center(box)
    sx, sy = center(seed)
    d2 = (cx - sx) ** 2 + (cy - sy) ** 2
    return floor + (1.0 - floor) * math.exp(-d2 / (2 * sigma ** 2))


def _subcells(cell: Box, m: int = 3) -> List[Box]:
    u, v, w, h = cell
    return [(u + c * w / m, v + r * h / m, w / m, h / m) for r in range(m) for c in range(m)]


def _grid(m: int = 3) -> List[Box]:
    s = 1.0 / m
    return [(c * s, r * s, s, s) for r in range(m) for c in range(m)]


def _emit(on_step: OnStep, box: Box, score: float, tag: str, k: int = 1) -> None:
    on_step({"box": [round(x, 5) for x in box], "score": round(score, 4), "tag": tag, "samples": k})


def _better(s: float, b: Box, best_s: float, best_b: Box) -> bool:
    """同分时取更紧的框：都是“够看清”，那分辨率更高者更优。"""
    if s > best_s + 1e-9:
        return True
    if abs(s - best_s) <= 1e-9 and area(b) < area(best_b) - 1e-12:
        return True
    return False


# ---------------------------------------------------------------- greedy
def run_greedy(seed: Box, probe: ProbeFn, on_step: OnStep,
               k: int = 3, factors: Sequence[float] = (1.0, 0.7, 0.5)) -> Dict:
    """论文默认档：三个尺度候选，每个探 k 次取均值，选最高（同分取更紧的框）。"""
    best_s, best_b = -1.0, seed
    for f in factors:
        cand = scale(seed, f)
        s = sum(probe(cand) for _ in range(k)) / k
        _emit(on_step, cand, s, f"scale×{f:g}", k)
        if _better(s, cand, best_s, best_b):
            best_s, best_b = s, cand
    return {"box": best_b, "score": best_s, "mode": "greedy"}


# ---------------------------------------------------------------- lookahead
def run_lookahead(seed: Box, probe: ProbeFn, on_step: OnStep,
                  k: int = 2, factors: Sequence[float] = (1.0, 0.7, 0.5)) -> Dict:
    """一步前瞻：value(d)=0.5*J(d)+0.5*J(下一步最可能的细化)。"""
    best_v, best_b, best_s = -1.0, seed, -1.0
    for f in factors:
        cand = scale(seed, f)
        s = sum(probe(cand) for _ in range(k)) / k
        _emit(on_step, cand, s, f"scale×{f:g}", k)
        nxt = scale(cand, 0.7)
        s2 = sum(probe(nxt) for _ in range(k)) / k
        _emit(on_step, nxt, s2, f"scale×{f:g} → 下一步", k)
        v = 0.5 * s + 0.5 * s2
        if v > best_v + 1e-9 or (abs(v - best_v) <= 1e-9 and area(cand) < area(best_b)):
            best_v, best_b, best_s = v, cand, s
    return {"box": best_b, "score": best_s, "mode": "lookahead", "value": best_v}


# ---------------------------------------------------------------- hrd（本项目改进）
def run_hrd(seed: Box, probe: ProbeFn, on_step: OnStep,
            m: int = 3, depth: int = 2, prior_w: float = 0.55,
            tau: float = 0.40, max_bt: int = 2, k_verify: int = 3,
            k_grid: int = 2, size_fn: Optional[Callable[[Box], float]] = None) -> Dict:
    """分层探测 + 覆盖先验 + 叶子验证回退。k_grid 次采样降低二值打平。"""
    # 搜索范围 = seed：第一次分层就在 seed 框内切 3×3，不再扫全图
    def rank(cells: List[Box], k: int) -> List[Tuple[float, float, Box]]:
        rows = []
        for c in cells:
            s = sum(probe(c) for _ in range(k)) / k
            _emit(on_step, c, s, "grid", k)
            rows.append((s + prior_w * _prior_mass(c, seed), s, c))
        rows.sort(key=lambda t: t[0], reverse=True)
        return rows

    level1 = rank(_subcells(seed, m), k_grid)

    def descend(start: Box) -> Box:
        best = start
        for _ in range(max(0, depth - 1)):
            subs = rank(_subcells(best, m), k_grid)
            best = subs[0][2]
            second = subs[1][1] if len(subs) > 1 else 0.0
            if subs[0][1] - second >= tau:
                break
        return best

    chosen, chosen_s = level1[0][2], level1[0][1]
    for attempt in range(max_bt + 1):
        leaf = descend(level1[attempt][2])
        s = sum(probe(leaf) for _ in range(k_verify)) / k_verify
        _emit(on_step, leaf, s, "verify", k_verify)
        chosen, chosen_s = leaf, s
        if s >= 0.5:
            break
    return {"box": chosen, "score": chosen_s, "mode": "hrd"}


POLICIES = {"greedy": run_greedy, "lookahead": run_lookahead, "hrd": run_hrd}


def run(mode: str, seed: Box, probe: ProbeFn, on_step: OnStep) -> Dict:
    fn = POLICIES.get(mode)
    if fn is None:
        raise ValueError(f"unknown mode: {mode}")
    return fn(clamp(seed), probe, on_step)
