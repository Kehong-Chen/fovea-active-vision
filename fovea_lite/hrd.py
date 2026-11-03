# -*- coding: utf-8 -*-
"""fovea_lite.hrd —— 小突破的正式实现：HRD-P / HRD-PB。

HRD-P（Hierarchical Resolvability Descent with Prior Mass）三步：
  1) 自顶向下切格（3×3），每格用 k 次 Yes/No 探测估计 resolvability；
  2) 排序分 = 探测分 + prior_w × 先验质量（论文 J(d) 的覆盖项），
     先验来自 VLM 的种子提案——这是解决"二值探测打平"的关键；
  3) 领先边际 >= tau 则早停（不继续往下切）。

HRD-PB = HRD-P + 叶子验证回退：
  下降到底后对叶子框再探 k_verify 次；若分数 < 0.5（说明这一步下错分支），
  回退到第一层排名第 2 的格子重新下降，最多 max_bt 次。
  这对应论文附录 H.2 的核心结论：失败主要是"目标进了候选池却没被选中"，
  属于选择器问题——回退是在选择器层面修，而不是把候选池暴力扩大。
"""
from __future__ import annotations

import math
from typing import List, Tuple

from .geometry import Box, center
from .policies import Prober, RunRecord, _subcells


def prior_mass(cell: Box, seed: Box, sigma: float = 0.28, floor: float = 0.08) -> float:
    """先验质量 ≈ 论文 J(d) 里的 ∫_{x∈d} p_t(x) dx（高斯近似 + 均匀兜底）。"""
    cx, cy = center(cell)
    sx, sy = center(seed)
    d2 = (cx - sx) ** 2 + (cy - sy) ** 2
    g = math.exp(-d2 / (2 * sigma ** 2))
    return floor + (1.0 - floor) * g


def _grid(m: int) -> List[Box]:
    s = 1.0 / m
    return [(c * s, r * s, s, s) for r in range(m) for c in range(m)]


def _rank(p: Prober, cells: List[Box], seed: Box, k: int, prior_w: float) -> List[Tuple[float, float, Box]]:
    rows = []
    for c in cells:
        s = p(c, k)
        rows.append((s + prior_w * prior_mass(c, seed), s, c))
    rows.sort(key=lambda t: t[0], reverse=True)
    return rows


def _descend(p: Prober, start: Box, seed: Box, m: int, depth: int, k: int,
             prior_w: float, tau: float) -> Box:
    best = start
    for _ in range(max(0, depth - 1)):
        subs = _rank(p, _subcells(best, m), seed, k, prior_w)
        best = subs[0][2]
        second = subs[1][1] if len(subs) > 1 else 0.0
        if subs[0][1] - second >= tau:
            break
    return best


def hrd_p(seed: Box, probe_fn, rng, m: int = 3, depth: int = 2, k_cell: int = 1,
          prior_w: float = 0.35, tau: float = 0.40, name: str = "hrd_p") -> RunRecord:
    rec = RunRecord(name, seed)
    p = Prober(probe_fn, rec)
    level1 = _rank(p, _grid(m), seed, k_cell, prior_w)
    rec.chosen = _descend(p, level1[0][2], seed, m, depth, k_cell, prior_w, tau)
    rec.levels = depth
    return rec


def hrd_pb(seed: Box, probe_fn, rng, m: int = 3, depth: int = 2, k_cell: int = 1,
           prior_w: float = 0.35, tau: float = 0.40, max_bt: int = 2,
           k_verify: int = 2, name: str = "hrd_pb") -> RunRecord:
    rec = RunRecord(name, seed)
    p = Prober(probe_fn, rec)
    level1 = _rank(p, _grid(m), seed, k_cell, prior_w)

    chosen = level1[0][2]
    for attempt in range(max_bt + 1):
        cell = level1[attempt][2]
        leaf = _descend(p, cell, seed, m, depth, k_cell, prior_w, tau)
        chosen = leaf
        if p(leaf, k_verify) >= 0.5:   # 叶子验证通过：确实看清了
            break
    rec.chosen = chosen
    rec.levels = depth
    return rec
