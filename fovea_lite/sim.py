# -*- coding: utf-8 -*-
"""fovea_lite.sim —— 实验台：在同一批合成场景上跑所有策略并汇总。

用法：
    python -m fovea_lite.sim --scenes 500 --seed 1
    python -m fovea_lite.sim --scenes 500 --seed 1 --out results.json --csv results.csv

输出三件事：
  1) 主表：准确率 / 平均探测次数 / 加权成本（论文 Table 5 的口径）
  2) 失败分解：proposal-limited / search-limited / 识别上限（论文附录 H.2 的口径）
  3) 按目标尺寸分桶的准确率（看冷启动瓶颈是否被缓解）
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from typing import Dict, List

from .geometry import inside
from .registry import all_names, build
from .world import SyntheticWorld, WorldConfig


def run_one(world: SyntheticWorld, policy_name: str, scene, rng) -> Dict:
    rec = None

    def probe_fn(box, n):
        return world.probe(box, scene, rng, n)

    rec = build(policy_name)(scene.seed, probe_fn, rng)
    ok = world.answer_correct(rec.chosen, scene, rng)

    pool = rec.pool if rec.pool else [rec.chosen]
    target_in_pool = any(inside(b, scene.target) for b in pool)
    target_in_chosen = inside(rec.chosen, scene.target)

    if not target_in_pool:
        mode = "proposal"
    elif not target_in_chosen:
        mode = "search"
    else:
        mode = "reached"

    return {
        "ok": ok,
        "probes": rec.probes,
        "cost": rec.cost,
        "mode": mode,
        "size": scene.size,
    }


def oracle_accuracy(world: SyntheticWorld, scenes, rng) -> float:
    """识别上限：给一个紧贴目标、分辨率足够的“黄金裁剪”。"""
    hits = 0
    for sc in scenes:
        w = min(1.0, sc.size * 6.0)
        box = (max(0.0, sc.target[0] - w / 2), max(0.0, sc.target[1] - w / 2), w, w)
        hits += world.answer_correct(box, sc, rng)
    return hits / len(scenes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--policies", type=str, default=",".join(all_names()))
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--csv", type=str, default="")
    args = ap.parse_args()

    world = SyntheticWorld(WorldConfig())
    scenes = [world.make_scene(random.Random(args.seed * 100003 + i)) for i in range(args.scenes)]

    oracle_rng = random.Random(args.seed * 7 + 1)
    oracle = oracle_accuracy(world, scenes, oracle_rng)

    names = [p for p in args.policies.split(",") if p]
    rows: List[Dict] = []
    for pi, name in enumerate(names):
        rng = random.Random(args.seed * 1000003 + pi)
        res = [run_one(world, name, sc, rng) for sc in scenes]
        acc = sum(r["ok"] for r in res) / len(res)
        probes = statistics.mean(r["probes"] for r in res)
        cost = statistics.mean(r["cost"] for r in res)
        prop = sum(1 for r in res if r["mode"] == "proposal") / len(res)
        search = sum(1 for r in res if r["mode"] == "search") / len(res)
        # 按目标大小分桶（三分位）
        sizes = sorted(r["size"] for r in res)
        q1, q2 = sizes[len(sizes) // 3], sizes[2 * len(sizes) // 3]
        buckets = {"small": [0, 0], "mid": [0, 0], "large": [0, 0]}
        for r in res:
            k = "small" if r["size"] <= q1 else ("mid" if r["size"] <= q2 else "large")
            buckets[k][0] += r["ok"]
            buckets[k][1] += 1
        rows.append({
            "policy": name,
            "acc": acc,
            "probes": probes,
            "cost": cost,
            "proposal_limited": prop,
            "search_limited": search,
            "acc_small": buckets["small"][0] / max(1, buckets["small"][1]),
            "acc_mid": buckets["mid"][0] / max(1, buckets["mid"][1]),
            "acc_large": buckets["large"][0] / max(1, buckets["large"][1]),
        })

    print(f"\n合成场景数 = {args.scenes}，oracle（黄金裁剪）准确率 = {oracle:.3f}")
    print(f"{'policy':<15}{'acc':>7}{'probes':>9}{'cost':>8}{'proposal':>10}{'search':>8}"
          f"{'small':>8}{'mid':>7}{'large':>7}")
    print("-" * 79)
    for r in rows:
        print(f"{r['policy']:<15}{r['acc']:>7.3f}{r['probes']:>9.1f}{r['cost']:>8.1f}"
              f"{r['proposal_limited']:>10.3f}{r['search_limited']:>8.3f}"
              f"{r['acc_small']:>8.3f}{r['acc_mid']:>7.3f}{r['acc_large']:>7.3f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"scenes": args.scenes, "seed": args.seed,
                       "oracle": oracle, "rows": rows}, f, ensure_ascii=False, indent=2)
        print(f"\n已写 {args.out}")
    if args.csv:
        with open(args.csv, "w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f"已写 {args.csv}")


if __name__ == "__main__":
    main()
