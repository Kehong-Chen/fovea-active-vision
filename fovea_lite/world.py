# -*- coding: utf-8 -*-
"""fovea_lite.world —— 合成“千兆像素”场景与观测量模型。

目的：把论文里的“搜索策略”单独隔出来验证。
论文自己用 oracle-crop 分析证明了：搜索失败与识别失败是两个独立瓶颈
（oracle 裁剪只有 68%）。所以研究搜索策略时，可以用一个可解析的观测模型
替代真 VLM，结论仍然对“策略好坏”有效——前提是把模型参数写清楚、可复现。

模型与论文一一对应：
  感知带宽 B            -> budget_px（裁剪被重采样到的边长像素数）
  信息密度 ρ(d)=B/A(d)  -> tokens_for_target()（目标占多少有效像素）
  分辨率概率 φ(d)       -> sigmoid(k*(tokens - T_nyq))，即论文 Eq.1 的 f_sat
  可见事件 S            -> 目标在框内 且 采样后“看清了”
  观测混合模型          -> 看不清时只能猜（p_guess），看清后仍有识别错误（p_rec）
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

from .geometry import Box, center, inside, sigmoid, tokens_for_target


@dataclass
class WorldConfig:
    # 观测物理（对应论文 Sec.2.2）
    budget_px: float = 1024.0     # 裁剪重采样后的边长；≈ VLM 输入分辨率
    k_res: float = 0.25           # sigmoid 斜率（每个有效像素）
    t_nyq: float = 20.0           # 语义奈奎斯特点：目标至少占 20 个有效像素
    # 认知（对应论文的识别瓶颈，保持各策略一致以便公平比较）
    p_rec: float = 0.90           # 已看清且目标在框内时答对的概率
    p_guess: float = 0.12         # 看不清/未覆盖时的瞎猜正确率
    # 探针（对应论文 Eq.10 的 Ĵ）
    probe_fp: float = 0.02        # 目标不在框内时探针误报 Yes 的概率
    distractor_match: float = 0.55  # 框内只有干扰物时探针的“像能答”系数
    # 场景
    n_distractors: int = 3
    size_mu: float = math.log(0.009)
    size_sigma: float = 0.55
    size_min: float = 0.003
    size_max: float = 0.030
    # 冷启动：种子提案命中目标的概率随目标大小上升（小目标常被漏掉）
    seed_hit_scale: float = 220.0
    seed_hit_mid: float = 0.0085


@dataclass
class Scene:
    target: Tuple[float, float]
    size: float
    distractors: List[Tuple[Tuple[float, float], float]] = field(default_factory=list)
    seed: Box = (0.0, 0.0, 0.1, 0.1)


class SyntheticWorld:
    """一个可解析的“VLM 观测器”替身。"""

    def __init__(self, cfg: WorldConfig | None = None):
        self.cfg = cfg or WorldConfig()

    # ---------- 场景生成 ----------
    def make_scene(self, rng, cold_start_noise: float = 1.0) -> Scene:
        cfg = self.cfg
        tx, ty = rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)
        s = math.exp(cfg.size_mu + cfg.size_sigma * rng.gauss(0, 1))
        s = min(cfg.size_max, max(cfg.size_min, s))

        dists = []
        for _ in range(cfg.n_distractors):
            dx, dy = rng.uniform(0.03, 0.97), rng.uniform(0.03, 0.97)
            ds = s * rng.uniform(0.35, 0.75)  # 干扰物比目标小 -> “不是最大的那个”
            dists.append(((dx, dy), ds))

        # 种子提案：命中概率随目标尺寸上升（论文 Fig.7a 的冷启动）
        p_hit = 1.0 / (1.0 + math.exp(-cfg.seed_hit_scale * (s - cfg.seed_hit_mid)))
        p_hit = min(0.97, max(0.03, p_hit * cold_start_noise))
        if rng.random() < p_hit:
            box_w = min(1.0, s * rng.uniform(3.0, 6.0))
            cx = tx + rng.gauss(0, 0.15 * box_w)
            cy = ty + rng.gauss(0, 0.15 * box_w)
        else:
            box_w = min(1.0, s * rng.uniform(3.0, 6.0))
            cx, cy = rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)
        u = min(max(0.0, cx - box_w / 2), 1.0 - box_w)
        v = min(max(0.0, cy - box_w / 2), 1.0 - box_w)
        return Scene(target=(tx, ty), size=s, distractors=dists, seed=(u, v, box_w, box_w))

    # ---------- 观测物理 ----------
    def resolve_prob(self, box: Box, scene: Scene) -> float:
        """φ(d)：这个裁剪把目标放大到“看得清”的概率。"""
        t = tokens_for_target(scene.size, box[2], self.cfg.budget_px)
        return sigmoid(self.cfg.k_res * (t - self.cfg.t_nyq))

    def visible_prob(self, box: Box, scene: Scene) -> float:
        """P(S=1|d) = 1[target in d] * φ(d)，对应论文 Eq.2。"""
        if not inside(box, scene.target):
            return 0.0
        return self.resolve_prob(box, scene)

    # ---------- 探针（论文 Eq.10 的 Ĵ 真值） ----------
    def probe_yes_prob(self, box: Box, scene: Scene) -> float:
        cfg = self.cfg
        if inside(box, scene.target):
            return (1.0 - cfg.probe_fp) * self.resolve_prob(box, scene)
        # 框内只有干扰物：可能“看起来能答”，这是论文说的 semantic distractor trap
        for (pt, ds) in scene.distractors:
            if inside(box, pt):
                fake = SyntheticWorld(WorldConfig(**{**cfg.__dict__, "size_mu": math.log(max(1e-4, ds))}))
                return cfg.distractor_match * fake.resolve_prob(box, scene)
        return cfg.probe_fp

    def probe(self, box: Box, scene: Scene, rng, n: int = 1) -> float:
        """采样 n 次 Yes/No，返回 Yes 的比例（Monte Carlo 估计）。"""
        p = self.probe_yes_prob(box, scene)
        hits = sum(1 for _ in range(n) if rng.random() < p)
        return hits / n

    # ---------- 最终作答 ----------
    def answer_correct(self, box: Box, scene: Scene, rng) -> bool:
        """在选定裁剪上作答是否正确（含识别瓶颈）。"""
        cfg = self.cfg
        if not inside(box, scene.target):
            return rng.random() < cfg.p_guess
        if rng.random() < self.resolve_prob(box, scene):
            return rng.random() < cfg.p_rec
        return rng.random() < cfg.p_guess
