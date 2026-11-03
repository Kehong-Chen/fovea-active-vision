# -*- coding: utf-8 -*-
"""fovea_lite.policies —— 裁剪策略集合。

基线（复现论文的四个策略）：
  react        直接执行模型给的种子框，不做精修
  greedy3      {1.5,1.0,0.8} 三个尺度候选，各探 3 次，取最高（论文默认档）
  mcmc         Metropolis-Hastings，步长=边长 15%，≤6 轮，3 次探测/候选
  lookahead    一步前瞻：候选 -> 模拟下一步 -> 用叶节点分数给当前候选估值
论文自己的补救（附录 H.2）：
  multiseed9   9 个种子 × 3 尺度 = 27 候选，各探 3 次（81 次探测）
本报告的小突破（两版，第二版才是有效的）：
  bpa_oneshot  反例：粗扫后只对格子做一次缩放 => 失效（说明"分配预算"不等于"分层定位"）
  bpa          分层探测：粗扫 -> 在最可能格子内再切 3×3 -> 再切一层 -> 边际早停
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Tuple

from .geometry import Box, clamp, inside, jitter, scale, tiles


@dataclass
class RunRecord:
    name: str
    chosen: Box
    probes: int = 0
    pool: List[Box] = field(default_factory=list)   # 实际评估过的候选（用于失败分解）
    levels: int = 0                                  # 分层探测用了几层

    @property
    def cost(self) -> float:
        return float(self.probes)


class Prober:
    def __init__(self, probe_fn: Callable[[Box, int], float], rec: RunRecord):
        self.probe_fn = probe_fn
        self.rec = rec

    def __call__(self, box: Box, n: int = 1) -> float:
        s = self.probe_fn(box, n)
        self.rec.probes += n
        self.rec.pool.append(box)
        return s

    def many(self, boxes: List[Box], n: int = 1) -> List[Tuple[float, Box]]:
        return [(self(b, n), b) for b in boxes]


# ---------------------------------------------------------------- 基线
def policy_react(seed: Box, probe_fn, rng) -> RunRecord:
    return RunRecord("react", seed)


def policy_greedy3(seed: Box, probe_fn, rng, k: int = 3) -> RunRecord:
    rec = RunRecord("greedy3", seed)
    p = Prober(probe_fn, rec)
    scored = p.many([scale(seed, f) for f in (1.5, 1.0, 0.8)], k)
    rec.chosen = max(scored, key=lambda t: t[0])[1]
    return rec


def policy_mcmc(seed: Box, probe_fn, rng, k: int = 3, iters: int = 6) -> RunRecord:
    rec = RunRecord("mcmc", seed)
    p = Prober(probe_fn, rec)
    cur, cur_s = seed, p(seed, k)
    for _ in range(iters):
        cand = jitter(cur, 0.15, rng)
        s = p(cand, k)
        if s >= cur_s or rng.random() < 0.1:
            cur, cur_s = cand, s
        if cur_s >= 1.0:
            break
    rec.chosen = cur
    return rec


def policy_lookahead(seed: Box, probe_fn, rng, k: int = 3) -> RunRecord:
    """一步前瞻：value(d) = 0.5*J(d) + 0.5*J(下一步最可能的细化)。"""
    rec = RunRecord("lookahead", seed)
    p = Prober(probe_fn, rec)
    best, best_v = seed, -1.0
    for f in (1.5, 1.0, 0.8):
        c = scale(seed, f)
        s = p(c, k)
        s2 = p(scale(c, 0.7), k)
        v = 0.5 * s + 0.5 * s2
        if v > best_v:
            best, best_v = c, v
    rec.chosen = best
    return rec


def policy_multiseed(seed: Box, probe_fn, rng, k: int = 3, seeds: int = 9) -> RunRecord:
    """论文附录 H.2 的多种子补救：9 个种子 × 3 尺度。"""
    rec = RunRecord(f"multiseed{seeds}k{k}", seed)
    p = Prober(probe_fn, rec)
    seed_boxes = [seed] + [jitter(seed, 0.9, rng) for _ in range(seeds - 1)]
    best, best_s = seed, -1.0
    for sb in seed_boxes:
        for f in (1.5, 1.0, 0.8):
            c = scale(sb, f)
            s = p(c, k)
            if s > best_s:
                best, best_s = c, s
    rec.chosen = best
    return rec


# ---------------------------------------------------------------- 小突破 v0（反例）
def policy_bpa_oneshot(seed: Box, probe_fn, rng, m: int = 3, top_cells: int = 2) -> RunRecord:
    """v0：粗扫 3×3 -> 对前 top_cells 个格子各取 3 个缩放候选 -> 取最高分。

    为什么失效：格子中心与目标位置无关，缩放只是"以格中心放大"，
    目标仍可能落在框外；而 probe 在格级别的分数已经饱和（φ≈1），
    区分不出"格内哪个位置"，于是后验分配得到的预算被浪费。
    """
    rec = RunRecord("bpa_oneshot", seed)
    p = Prober(probe_fn, rec)
    cells = p.many(tiles(m), 1)
    cells.sort(key=lambda t: t[0], reverse=True)
    fine = Prober(probe_fn, rec)
    cands: List[Tuple[float, Box]] = []
    for _, cell in cells[:top_cells]:
        cands += fine.many([scale(cell, f) for f in (1.0, 0.62, 0.45)], 1)
    cands.sort(key=lambda t: t[0], reverse=True)
    rec.chosen = cands[0][1]
    rec.levels = 1
    return rec


# ---------------------------------------------------------------- 小突破（有效版）
def _subcells(cell: Box, m: int = 3) -> List[Box]:
    u, v, w, h = cell
    out = []
    for r in range(m):
        for c in range(m):
            out.append((u + c * w / m, v + r * h / m, w / m, h / m))
    return out


def policy_bpa(seed: Box, probe_fn, rng, m: int = 3, depth: int = 3,
               tau: float = 0.35, k_final: int = 3) -> RunRecord:
    """分层可解性探测（Hierarchical Resolvability Descent, HRD）。

    与论文的单步精修不同，这里把"看哪里"做成一次自顶向下的下降：
      L0：3×3 网格各探 1 次（回答"目标大约在哪一格"）
      L1：选中的格子内再切 3×3，各探 1 次（分辨到 1/9 图宽）
      L2：再切一层（1/27 图宽），此时目标约占叶子框 1/4，φ≈1
      早停：某层第一名领先第二名 >= tau 且分数 >= 0.5，就直接用它作答，
            后面的层不再探测（省下 9–18 次探测）
      收尾：对最终叶子框再取 3 个尺度（1.0/0.7/0.5）复核一次，取最高
    为什么能治冷启动：种子框漏掉目标时（论文 Fig.7a），分层下降仍然
    覆盖全图——因为 L0 的 9 个格子并起来就是整张图，不存在"目标从未进过候选池"。
    """
    rec = RunRecord("bpa", seed)
    p = Prober(probe_fn, rec)
    levels = max(2, min(depth, 4))

    # 没有种子先验时从整图开始；有种子就把它当作额外的一次先验探测
    cands = p.many(tiles(m), 1)
    cands.sort(key=lambda t: t[0], reverse=True)
    best_s, best_b = cands[0]
    second_s = cands[1][0] if len(cands) > 1 else 0.0
    rec.levels = 1

    if best_s <= 0.0:
        # 全零：说明没有任何格子能分辨，退化为"种子所在格"继续下降
        best_b = _cell_of(seed, m)
        best_s, second_s = 0.0, 0.0

    if not (best_s - second_s >= tau and best_s >= 0.5):
        for lv in range(2, levels + 1):
            sub = p.many(_subcells(best_b, m), 1)
            sub.sort(key=lambda t: t[0], reverse=True)
            s1, b1 = sub[0]
            s2 = sub[1][0] if len(sub) > 1 else 0.0
            best_b = b1
            if s1 > 0.0:
                best_s = s1
            second_s = s2
            rec.levels = lv
            if s1 - s2 >= tau and s1 >= 0.5:
                break

    # 收尾复核：在最终框上取 3 个尺度，探 k_final 次，防止"太紧裁掉目标/太松看不清"
    fine = Prober(probe_fn, rec)
    final = fine.many([scale(best_b, f) for f in (1.0, 0.7, 0.5)], k_final)
    final.sort(key=lambda t: t[0], reverse=True)
    rec.chosen = final[0][1] if final[0][0] > 0 else best_b
    return rec


def policy_bpa_nostop(seed: Box, probe_fn, rng, m: int = 3, depth: int = 3,
                      k_final: int = 3) -> RunRecord:
    """消融：分层但永不早停（每层都探满）。"""
    rec = policy_bpa(seed, probe_fn, rng, m=m, depth=depth, tau=-1.0, k_final=k_final)
    rec.name = "bpa_nostop"
    return rec


def policy_bpa_k1(seed: Box, probe_fn, rng, m: int = 3, depth: int = 3) -> RunRecord:
    """消融：分层 + 收尾只探 1 次（省探测，但分数更噪）。"""
    rec = policy_bpa(seed, probe_fn, rng, m=m, depth=depth, tau=0.35, k_final=1)
    rec.name = "bpa_k1"
    return rec


def _cell_of(box: Box, m: int) -> Box:
    u, v, w, h = box
    cx, cy = u + w / 2, v + h / 2
    c = min(m - 1, max(0, int(cx * m)))
    r = min(m - 1, max(0, int(cy * m)))
    s = 1.0 / m
    return (c * s, r * s, s, s)


ALL_POLICIES = [
    "react", "greedy3", "mcmc", "lookahead",
    "multiseed9", "multiseed9_k1",
    "bpa_oneshot", "bpa_k1", "bpa_nostop", "bpa",
]


def build_policy(name: str):
    table = {
        "react": policy_react,
        "greedy3": policy_greedy3,
        "mcmc": policy_mcmc,
        "lookahead": policy_lookahead,
        "multiseed9": lambda s, p, r: policy_multiseed(s, p, r, k=3, seeds=9),
        "multiseed9_k1": lambda s, p, r: policy_multiseed(s, p, r, k=1, seeds=9),
        "bpa_oneshot": policy_bpa_oneshot,
        "bpa_k1": policy_bpa_k1,
        "bpa_nostop": policy_bpa_nostop,
        "bpa": policy_bpa,
    }
    if name not in table:
        raise KeyError(f"unknown policy {name}")
    return table[name]
