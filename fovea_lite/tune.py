# -*- coding: utf-8 -*-
"""fovea_lite.tune —— 在调参集(seed=2)上选分层下降的配置，避免在报告集(seed=1)上调参。

只做一件小事：给定 (m, depth, k_cell, prior_w)，在 200 个场景上测准确率与探测数。
"""
from __future__ import annotations

import random
from typing import Dict, List

from .geometry import Box, center, inside, scale
from .policies import Prober, RunRecord, _subcells, policy_greedy3, policy_multiseed
from .world import SyntheticWorld, WorldConfig


def _prior_mass(cell: Box, seed: Box, sigma: float = 0.28, floor: float = 0.08) -> float:
    """先验质量：种子框中心的高斯先验在某格上的积分近似。

    对应论文 J(d) 里的 ∫_{x∈d} p_t(x) dx —— 初始信念由 VLM 的提案给出。
    """
    u, w = cell[0], cell[2]
    v, h = cell[1], cell[3]
    cx, cy = center(cell)
    sx, sy = center(seed)
    # 用格中心处的高斯密度 × 格面积 作为质量近似
    import math
    d2 = (cx - sx) ** 2 + (cy - sy) ** 2
    g = math.exp(-d2 / (2 * sigma ** 2))
    return floor + (1.0 - floor) * g


def policy_bpa(
    seed: Box,
    probe_fn,
    rng,
    m: int = 3,
    depth: int = 3,
    k_cell: int = 2,
    prior_w: float = 0.35,
    tau: float = 0.40,
    w_stop: float = 0.08,
) -> RunRecord:
    """Hierarchical Resolvability Descent with Prior Mass（HRD-P）。

    每一层：对候选格各探 k_cell 次得到 resolvability 得分，再按
        rank = probe_score + prior_w * prior_mass(cell)
    排序（prior_mass 对应论文 J(d) 里的覆盖项），取第一名继续下降。
    终止：领先第二名 >= tau 且格子足够小（<= w_stop），或到最大层数。
    """
    rec = RunRecord("bpa", seed)
    p = Prober(probe_fn, rec)
    best: Box = seed
    for lv in range(depth):
        cands: List[Box] = _subcells(best, m) if lv > 0 else _grid(m)
        scored = []
        for c in cands:
            s = p(c, k_cell)
            scored.append((s + prior_w * _prior_mass(c, seed), s, c))
        scored.sort(key=lambda t: t[0], reverse=True)
        rank1, s1, b1 = scored[0]
        s2 = scored[1][1] if len(scored) > 1 else 0.0
        best = b1
        if (s1 - s2 >= tau) and (b1[2] <= w_stop):
            break
    rec.chosen = best
    rec.levels = 0
    return rec


def _grid(m: int) -> List[Box]:
    s = 1.0 / m
    return [(c * s, r * s, s, s) for r in range(m) for c in range(m)]


def _run(policy_fn, world, scenes, rng) -> float:
    ok = 0
    for sc in scenes:
        rec = policy_fn(sc.seed, lambda b, n: world.probe(b, sc, rng, n), rng)
        ok += world.answer_correct(rec.chosen, sc, rng)
    return ok / len(scenes)


def main():
    world = SyntheticWorld(WorldConfig())
    scenes = [world.make_scene(random.Random(2 * 100003 + i)) for i in range(200)]
    rng = random.Random(4242)

    base = {
        "greedy3": _run(lambda s, p, r: policy_greedy3(s, p, r), world, scenes, rng),
    }
    print(f"调参集(200 场景, seed=2)  greedy3 基线 acc={base['greedy3']:.3f}")

    configs: List[Dict] = []
    for m in (2, 3):
        for depth in (2, 3, 4):
            for k_cell in (1, 2):
                for prior_w in (0.0, 0.35):
                    configs.append(dict(m=m, depth=depth, k_cell=k_cell, prior_w=prior_w))

    rows = []
    for cfg in configs:
        recs = []
        probes = []
        for sc in scenes:
            r = policy_bpa(sc.seed, lambda b, n: world.probe(b, sc, rng, n), rng, **cfg)
            recs.append(world.answer_correct(r.chosen, sc, rng))
            probes.append(r.probes)
        acc = sum(recs) / len(recs)
        rows.append((acc, sum(probes) / len(probes), cfg))
    rows.sort(key=lambda t: (-t[0], t[1]))
    print(f"\n{'acc':>6}{'probes':>9}   config")
    for acc, pr, cfg in rows:
        print(f"{acc:>6.3f}{pr:>9.1f}   m={cfg['m']} depth={cfg['depth']} "
              f"k_cell={cfg['k_cell']} prior_w={cfg['prior_w']}")


if __name__ == "__main__":
    main()
