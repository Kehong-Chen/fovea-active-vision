# -*- coding: utf-8 -*-
"""fovea_lite.analyze —— 统计检验 + 预算-精度曲线（报告里要用的数字都在这里产出）。

产出：
  1) McNemar 配对检验（同一批场景，两个策略逐题比较）
  2) 准确率差的 bootstrap 95% 置信区间
  3) 预算-精度曲线：改变回退次数 max_bt，得到 HRD-PB 的不同工作点
"""
from __future__ import annotations

import json
import math
import random
from typing import List, Tuple

from .registry import build
from .world import SyntheticWorld, WorldConfig

N_SCENES = 600
SEED = 1


def outcomes(world, scenes, name: str, rng) -> List[bool]:
    fn = build(name)
    out = []
    for sc in scenes:
        rec = fn(sc.seed, lambda b, n: world.probe(b, sc, rng, n), rng)
        out.append(world.answer_correct(rec.chosen, sc, rng))
    return out


def mcnemar(a: List[bool], b: List[bool]) -> Tuple[int, int, float]:
    """返回 (a赢b的题数, b赢a的题数, 双尾精确 p 值)。"""
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return n01, n10, min(1.0, p)


def boot_ci(a: List[bool], b: List[bool], iters: int = 2000, rng=None) -> Tuple[float, float]:
    rng = rng or random.Random(0)
    diffs = []
    n = len(a)
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a[i] for i in idx) / n - sum(b[i] for i in idx) / n)
    diffs.sort()
    return diffs[int(0.025 * iters)], diffs[int(0.975 * iters)]


def main():
    world = SyntheticWorld(WorldConfig())
    scenes = [world.make_scene(random.Random(SEED * 100003 + i)) for i in range(N_SCENES)]
    rng = random.Random(2026)

    names = ["react", "greedy3", "mcmc", "lookahead", "multiseed9", "hrd_p", "hrd_pb"]
    res = {n: outcomes(world, scenes, n, rng) for n in names}
    acc = {n: sum(v) / len(v) for n, v in res.items()}

    print(f"场景数={N_SCENES}  seed={SEED}")
    for n in names:
        print(f"  {n:<10} acc={acc[n]:.3f}")

    pairs = [("hrd_pb", "greedy3"), ("hrd_pb", "multiseed9"), ("hrd_pb", "react"),
             ("hrd_p", "greedy3"), ("hrd_pb", "hrd_p")]
    tests = []
    print("\n配对检验（McNemar，双尾精确 p）与 bootstrap 95% CI：")
    for a, b in pairs:
        n01, n10, p = mcnemar(res[a], res[b])
        lo, hi = boot_ci(res[a], res[b])
        tests.append({"a": a, "b": b, "acc_a": acc[a], "acc_b": acc[b],
                      "a_only": n01, "b_only": n10, "p": p, "ci": [lo, hi]})
        print(f"  {a} vs {b}: Δ={acc[a]-acc[b]:+.3f} [{lo:+.3f},{hi:+.3f}]  "
              f"a-only={n01} b-only={n10} p={p:.2g}")

    # 预算-精度曲线：max_bt 与 depth 的组合
    from .hrd import hrd_pb
    print("\n预算-精度曲线（HRD-PB，改变回退次数与深度）：")
    frontier = []
    for depth, max_bt in [(2, 0), (2, 1), (2, 2), (2, 4), (2, 8), (3, 4)]:
        ok, pr = 0, 0
        for sc in scenes:
            rec = hrd_pb(sc.seed, lambda b, n: world.probe(b, sc, rng, n), rng,
                         depth=depth, max_bt=max_bt, k_cell=1, prior_w=0.55)
            ok += world.answer_correct(rec.chosen, sc, rng)
            pr += rec.probes
        pt = {"depth": depth, "max_bt": max_bt, "acc": ok / len(scenes),
              "probes": pr / len(scenes)}
        frontier.append(pt)
        print(f"  depth={depth} max_bt={max_bt}: acc={pt['acc']:.3f} probes={pt['probes']:.1f}")

    with open("results_analysis.json", "w", encoding="utf-8") as f:
        json.dump({"scenes": N_SCENES, "seed": SEED, "acc": acc,
                   "tests": tests, "frontier": frontier}, f, ensure_ascii=False, indent=2)
    print("\n已写 results_analysis.json")


if __name__ == "__main__":
    main()
