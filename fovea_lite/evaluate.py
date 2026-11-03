# -*- coding: utf-8 -*-
"""fovea_lite.evaluate —— 唯一的评测入口（报告里的所有数字都来自这里）。

    python -m fovea_lite.evaluate

输出：
  results_final.json / .csv   主表（准确率、探测次数、失败分解、尺寸分桶）
  results_analysis.json       McNemar 配对检验、bootstrap CI、预算-精度曲线
"""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
from typing import Dict, List

from .geometry import inside
from .registry import build
from .world import SyntheticWorld, WorldConfig

N_SCENES = 600
SEED = 1
BASE_POLICIES = ["react", "greedy3", "mcmc", "lookahead", "multiseed9",
                 "multiseed9_k1", "bpa_oneshot", "hrd_p_noprior", "hrd_p", "hrd_pb"]


def eval_policy(world, scenes, name: str, pi: int) -> Dict:
    fn = build(name)
    rng = random.Random(SEED * 1_000_003 + pi)   # 每个策略独立随机流（与 sim 一致）
    ok, probes, modes = [], [], []
    for sc in scenes:
        rec = fn(sc.seed, lambda b, n: world.probe(b, sc, rng, n), rng)
        good = world.answer_correct(rec.chosen, sc, rng)
        pool = rec.pool if rec.pool else [rec.chosen]
        in_pool = any(inside(b, sc.target) for b in pool)
        in_chosen = inside(rec.chosen, sc.target)
        mode = "proposal" if not in_pool else ("search" if not in_chosen else "reached")
        ok.append(good)
        probes.append(rec.probes)
        modes.append(mode)
    return {"name": name, "ok": ok, "probes": probes, "modes": modes,
            "acc": sum(ok) / len(ok)}


def bucket_acc(ok: List[bool], sizes: List[float]) -> Dict[str, float]:
    order = sorted(range(len(sizes)), key=lambda i: sizes[i])
    n = len(order)
    parts = {"small": order[: n // 3], "mid": order[n // 3: 2 * n // 3], "large": order[2 * n // 3:]}
    return {k: sum(ok[i] for i in idx) / len(idx) for k, idx in parts.items()}


def mcnemar(a: List[bool], b: List[bool]):
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return n01, n10, min(1.0, p)


def boot_ci(a: List[bool], b: List[bool], iters: int = 2000, seed: int = 0):
    rng = random.Random(seed)
    n = len(a)
    d = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        d.append(sum(a[i] for i in idx) / n - sum(b[i] for i in idx) / n)
    d.sort()
    return d[int(0.025 * iters)], d[int(0.975 * iters)]


def main():
    world = SyntheticWorld(WorldConfig())
    scenes = [world.make_scene(random.Random(SEED * 100003 + i)) for i in range(N_SCENES)]
    sizes = [sc.size for sc in scenes]

    oracle_rng = random.Random(SEED * 7 + 1)
    oracle_ok = []
    for sc in scenes:
        w = min(1.0, sc.size * 6.0)
        box = (max(0.0, sc.target[0] - w / 2), max(0.0, sc.target[1] - w / 2), w, w)
        oracle_ok.append(world.answer_correct(box, sc, oracle_rng))
    oracle = sum(oracle_ok) / len(oracle_ok)

    results = {}
    rows = []
    for pi, name in enumerate(BASE_POLICIES):
        r = eval_policy(world, scenes, name, pi)
        results[name] = r
        acc = r["acc"]
        pr = statistics.mean(r["probes"])
        prop = sum(1 for m in r["modes"] if m == "proposal") / len(r["modes"])
        srch = sum(1 for m in r["modes"] if m == "search") / len(r["modes"])
        bk = bucket_acc(r["ok"], sizes)
        rows.append({"policy": name, "acc": acc, "probes": pr, "proposal": prop,
                     "search": srch, "acc_small": bk["small"], "acc_mid": bk["mid"],
                     "acc_large": bk["large"]})
        print(f"{name:<15} acc={acc:.3f} probes={pr:5.1f} proposal={prop:.3f} "
              f"search={srch:.3f} small={bk['small']:.3f} mid={bk['mid']:.3f} large={bk['large']:.3f}")

    print(f"\noracle(黄金裁剪) = {oracle:.3f}")

    pairs = [("hrd_pb", "greedy3"), ("hrd_pb", "multiseed9"), ("hrd_pb", "react"),
             ("hrd_p", "greedy3"), ("hrd_pb", "hrd_p"), ("hrd_pb", "hrd_p_noprior")]
    tests = []
    print("\n配对检验（McNemar 精确双尾 p；bootstrap 95% CI）：")
    for a, b in pairs:
        n01, n10, p = mcnemar(results[a]["ok"], results[b]["ok"])
        lo, hi = boot_ci(results[a]["ok"], results[b]["ok"], seed=7)
        tests.append({"a": a, "b": b, "acc_a": results[a]["acc"], "acc_b": results[b]["acc"],
                      "a_only": n01, "b_only": n10, "p": p, "ci": [lo, hi]})
        print(f"  {a} vs {b}: Δ={results[a]['acc']-results[b]['acc']:+.3f} "
              f"[{lo:+.3f},{hi:+.3f}] a-only={n01} b-only={n10} p={p:.2g}")

    # 预算-精度曲线
    from .hrd import hrd_pb
    print("\n预算-精度曲线（HRD-PB，改回退次数/深度）：")
    frontier = []
    for bi, (depth, max_bt) in enumerate([(2, 0), (2, 1), (2, 2), (2, 4), (2, 8), (3, 4)]):
        # 与主表 hrd_pb 用同一随机流（公共随机数），使 max_bt=8 那一点与主表可对上
        rng = random.Random(SEED * 1_000_003 + 9)
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

    with open("results_final.json", "w", encoding="utf-8") as f:
        json.dump({"scenes": N_SCENES, "seed": SEED, "oracle": oracle, "rows": rows}, f,
                  ensure_ascii=False, indent=2)
    with open("results_final.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open("results_analysis.json", "w", encoding="utf-8") as f:
        json.dump({"scenes": N_SCENES, "seed": SEED, "oracle": oracle,
                   "tests": tests, "frontier": frontier}, f, ensure_ascii=False, indent=2)
    print("\n已写 results_final.json / results_final.csv / results_analysis.json")


if __name__ == "__main__":
    main()
