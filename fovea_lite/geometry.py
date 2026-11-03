# -*- coding: utf-8 -*-
"""fovea_lite.geometry —— 归一化裁剪框的几何操作。

约定：所有框都是 (u, v, w, h)，归一化到 [0,1]：
  u, v = 左上角；w, h = 宽高。
这样写的好处是策略、探针、仿真三者都不依赖像素尺寸。
"""
from __future__ import annotations

import math
from typing import Tuple

Box = Tuple[float, float, float, float]

MIN_SIDE = 1e-3


def clamp(box: Box) -> Box:
    """把框裁进 [0,1]，并保证最小边长。"""
    u, v, w, h = box
    w = max(MIN_SIDE, min(1.0, float(w)))
    h = max(MIN_SIDE, min(1.0, float(h)))
    u = min(max(0.0, float(u)), 1.0 - w)
    v = min(max(0.0, float(v)), 1.0 - h)
    return (u, v, w, h)


def center(box: Box) -> Tuple[float, float]:
    u, v, w, h = box
    return (u + w / 2.0, v + h / 2.0)


def area(box: Box) -> float:
    return box[2] * box[3]


def scale(box: Box, factor: float) -> Box:
    """以中心为锚点缩放。factor>1 拉远（看上下文），<1 拉近（看细节）。"""
    cx, cy = center(box)
    nw = min(1.0, box[2] * factor)
    nh = min(1.0, box[3] * factor)
    return clamp((cx - nw / 2.0, cy - nh / 2.0, nw, nh))


def jitter(box: Box, step: float, rng) -> Box:
    """高斯扰动，step 是相对边长的标准差（论文 MCMC 用 15%）。"""
    du = rng.gauss(0.0, step * box[2])
    dv = rng.gauss(0.0, step * box[3])
    dw = rng.gauss(1.0, step * 0.5) * box[2]
    dh = rng.gauss(1.0, step * 0.5) * box[3]
    return clamp((box[0] + du, box[1] + dv, dw, dh))


def inside(box: Box, pt: Tuple[float, float]) -> bool:
    u, v, w, h = box
    x, y = pt
    return (u <= x <= u + w) and (v <= y <= v + h)


def iou(a: Box, b: Box) -> float:
    ax1, ay1, bx1, by1 = a[0], a[1], b[0], b[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    return inter / (area(a) + area(b) - inter)


def tiles(m: int) -> list:
    """把整图切成 m×m 网格，返回按行优先排列的框（用于粗扫）。"""
    s = 1.0 / m
    out = []
    for r in range(m):
        for c in range(m):
            out.append((c * s, r * s, s, s))
    return out


def tokens_for_target(target_size: float, crop_width: float, budget_px: float) -> float:
    """目标在裁剪后的“有效像素/有效 token 数”。

    物理推导（对应论文的密度 ρ(d)=B/A(d)）：
      裁剪宽 = crop_width * W 像素，被重采样到 budget_px；
      目标占原图 target_size*W 像素 → 重采样后占 target_size*budget_px/crop_width。
    所以“看得清”只取决于目标相对裁剪的长度占比，与绝对分辨率无关。
    """
    if crop_width <= 0:
        return float("inf")
    return target_size * budget_px / crop_width


def sigmoid(x: float) -> float:
    if x > 50:
        return 1.0
    if x < -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))
