# -*- coding: utf-8 -*-
"""策略注册表：把 policies 里的基线与 hrd 里的新策略统一到一个入口。

默认配置来自调参集(seed=2)的选择，报告集(seed=1)只看不调：
  hrd_p          18 次探测左右，只下降不回退（省预算的版本）
  hrd_pb         ~25 次探测，叶子验证 + 最多回退 4 次（默认推荐）
  hrd_pb_max     ~53 次探测，更深更多回退（高预算上限点）
  hrd_p_noprior  消融：去掉先验质量项（证明它不能省）
"""
from .policies import ALL_POLICIES, build_policy as _build


def _wrap(name):
    def _fn(seed, probe_fn, rng):
        from . import hrd
        if name == "hrd_p":
            return hrd.hrd_p(seed, probe_fn, rng, depth=2, k_cell=1,
                             prior_w=0.55, name="hrd_p")
        if name == "hrd_pb":
            return hrd.hrd_pb(seed, probe_fn, rng, depth=2, max_bt=8, k_cell=1,
                              prior_w=0.55, k_verify=2, name="hrd_pb")
        if name == "hrd_pb_max":
            return hrd.hrd_pb(seed, probe_fn, rng, depth=3, max_bt=4, k_cell=2,
                              prior_w=0.55, k_verify=2, name="hrd_pb_max")
        if name == "hrd_p_noprior":
            return hrd.hrd_p(seed, probe_fn, rng, depth=2, k_cell=1,
                             prior_w=0.0, name="hrd_p_noprior")
        raise KeyError(name)
    return _fn


_EXTRA = ["hrd_p", "hrd_pb", "hrd_pb_max", "hrd_p_noprior"]


def build(name: str):
    if name in _EXTRA:
        return _wrap(name)
    return _build(name)


def all_names():
    return ALL_POLICIES + _EXTRA
