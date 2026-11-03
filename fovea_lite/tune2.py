# -*- coding: utf-8 -*-
"""调参集(seed=2)上扫 HRD-PB 的 (depth, max_bt, k_cell, prior_w)。"""
from __future__ import annotations

import random

from .hrd import hrd_pb
from .world import SyntheticWorld, WorldConfig


def main():
    world = SyntheticWorld(WorldConfig())
    scenes = [world.make_scene(random.Random(2 * 100003 + i)) for i in range(200)]
    rng = random.Random(99)

    rows = []
    for depth in (2, 3):
        for max_bt in (2, 4):
            for k_cell in (1, 2):
                for prior_w in (0.35, 0.55):
                    ok, pr = 0, 0
                    for sc in scenes:
                        rec = hrd_pb(sc.seed, lambda b, n: world.probe(b, sc, rng, n), rng,
                                     depth=depth, max_bt=max_bt, k_cell=k_cell, prior_w=prior_w)
                        ok += world.answer_correct(rec.chosen, sc, rng)
                        pr += rec.probes
                    rows.append((ok / len(scenes), pr / len(scenes),
                                 dict(depth=depth, max_bt=max_bt, k_cell=k_cell, prior_w=prior_w)))
    rows.sort(key=lambda t: (-t[0], t[1]))
    print(f"{'acc':>6}{'probes':>9}   config")
    for acc, pr, cfg in rows:
        print(f"{acc:>6.3f}{pr:>9.1f}   depth={cfg['depth']} max_bt={cfg['max_bt']} "
              f"k_cell={cfg['k_cell']} prior_w={cfg['prior_w']}")


if __name__ == "__main__":
    main()
