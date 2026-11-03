# -*- coding: utf-8 -*-
"""fovea_service.backends —— 两种探针后端。

SimBackend：用 fovea_lite 的合成观测模型，无需 GPU/模型，用来把整条链路跑通、演示 UI 与两种搜索。
VLBackend ：接任意 OpenAI 兼容的多模态端点（本地 vLLM / 云端 API），按论文的
            “可解性探测”问法只取 Yes/No，采样 k 次取均值。
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import random
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

from fovea_lite.geometry import Box, inside
from fovea_lite.world import Scene, SyntheticWorld, WorldConfig

PROBE_SYS = """你是一个「可解性评估器」。给定原始图像、一个候选裁剪区域、以及用户问题，
你只需判断：这个候选区域里的信息是否足以回答问题。
- 如果候选区域与问题所指位置无关，回答 No。
- 如果区域太模糊、目标太小看不清，回答 No。
- 只输出 Yes 或 No，不要回答问题本身。
输出格式：<answer>Yes</answer> 或 <answer>No</answer>"""


# ------------------------------------------------------------------ 合成
def make_demo_scene(seed: int = 7) -> Tuple[Scene, bytes, Tuple[int, int]]:
    """生成一张合成“大图” + 对应场景（目标/干扰物）。

    目标放在画面中心区（0.42–0.58），模拟论文的前提：VLM 给的初始种子框是
    “大致靠谱但不够准”的提案；干扰物散布全图，制造语义干扰。
    """
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    W, H = 3072, 3072
    world = SyntheticWorld(WorldConfig())
    sc = world.make_scene(rng, cold_start_noise=1.0)

    # 目标放到中心区，种子框（默认 0.3–0.7）一定覆盖它
    sc.target = (rng.uniform(0.44, 0.56), rng.uniform(0.44, 0.56))
    sc.size = rng.uniform(0.006, 0.012)
    # 干扰物：更小、散布全图（避免与目标重叠）
    sc.distractors = []
    for _ in range(3):
        while True:
            pt = (rng.uniform(0.06, 0.94), rng.uniform(0.06, 0.94))
            if abs(pt[0] - sc.target[0]) > 0.08 or abs(pt[1] - sc.target[1]) > 0.08:
                break
        sc.distractors.append((pt, sc.size * rng.uniform(0.35, 0.7)))

    img = Image.new("RGB", (W, H), (28, 62, 40))
    d = ImageDraw.Draw(img)
    # 底纹：模拟遥感纹理
    for _ in range(2600):
        x, y = rng.randrange(W), rng.randrange(H)
        r = rng.randrange(6, 26)
        c = rng.randrange(24, 70)
        d.ellipse([x, y, x + r, y + r], fill=(c, c + 24, c - 4))

    def rect(pt, size, color, ring=False):
        cx, cy = pt[0] * W, pt[1] * H
        half = size * W / 2
        d.rectangle([cx - half, cy - half, cx + half, cy + half], fill=color)
        if ring:
            d.ellipse([cx - half * 0.5, cy - half * 0.5, cx + half * 0.5, cy + half * 0.5],
                      outline=(255, 255, 255), width=max(3, int(size * W * 0.08)))

    for pt, ds in sc.distractors:  # 干扰物：更小的浅色方块（“不是最大的那个”）
        rect(pt, ds, (198, 176, 96))
    rect(sc.target, sc.size, (214, 226, 210), ring=True)  # 目标：白色圆环

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return sc, buf.getvalue(), (W, H)


class SimBackend:
    """无模型后端：probe 概率来自合成世界，采样得到 Yes/No 分数。"""

    name = "sim"

    def __init__(self, scene: Scene, seed: int = 0):
        self.scene = scene
        self.world = SyntheticWorld(WorldConfig())
        self.rng = random.Random(seed)

    def probe(self, box: Box) -> float:
        return self.world.probe(box, self.scene, self.rng, 1)

    def answer_hit(self, box: Box) -> bool:
        return inside(box, self.scene.target)


# ------------------------------------------------------------------ 真 VLM
class VLBackend:
    """OpenAI 兼容端点后端（vLLM / 云端 API 均可）。"""

    name = "vlm"

    def __init__(self, image_bytes: bytes, question: str,
                 base_url: Optional[str] = None, model: Optional[str] = None,
                 api_key: Optional[str] = None, max_side: int = 4096):
        from PIL import Image

        self.base_url = (base_url or os.environ.get("FOVEA_BASE_URL", "http://127.0.0.1:8000/v1")).rstrip("/")
        self.model = model or os.environ.get("FOVEA_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
        # 答题可用单独的（更强）模型，探测仍用便宜的小模型
        self.answer_model = os.environ.get("FOVEA_ANSWER_MODEL") or self.model
        self.api_key = api_key or os.environ.get("FOVEA_API_KEY", "EMPTY")
        self.question = question
        self.img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if max(self.img.size) > max_side:
            r = max_side / max(self.img.size)
            self.img = self.img.resize((int(self.img.size[0] * r), int(self.img.size[1] * r)))
        self.calls = 0

    def _b64(self, im) -> str:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _crop(self, box: Box):
        W, H = self.img.size
        u, v, w, h = box
        return self.img.crop((int(u * W), int(v * H), int((u + w) * W), int((v + h) * H)))

    def _chat(self, body: Dict) -> str:
        """带超时重试的单次对话调用，返回正文。"""
        payload = json.dumps(body).encode("utf-8")
        # 第三方端点会偶发卡死：单次超时别设太长，失败重试，别让一次卡顿拖垮整个搜索任务
        timeout = float(os.environ.get("FOVEA_TIMEOUT", "60"))
        tries = int(os.environ.get("FOVEA_RETRIES", "3"))
        last = None
        data = None
        for i in range(max(1, tries)):
            try:
                req = urllib.request.Request(
                    self.base_url + "/chat/completions",
                    data=payload,
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except Exception as e:  # noqa: BLE001
                last = e
                if i + 1 < tries:
                    time.sleep(1.5 * (i + 1))
        if data is None:
            raise last if last else RuntimeError("vlm call failed")
        self.calls += 1
        return str(data["choices"][0]["message"]["content"])

    def probe(self, box: Box) -> float:
        text = (f"用户问题：{self.question}\n"
                f"第一张图是原图，第二张是候选裁剪区域（归一化坐标 {tuple(round(x,3) for x in box)}）。\n"
                f"请判断该区域信息是否足以回答问题，只回答 Yes 或 No。")
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": PROBE_SYS},
                {"role": "user", "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + self._b64(self.img)}},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + self._b64(self._crop(box))}},
                ]},
            ],
            "temperature": 0.7,
            "max_tokens": 32,
        }
        out = self._chat(body).lower()
        if "yes" in out and "no" not in out:
            return 1.0
        if "no" in out and "yes" not in out:
            return 0.0
        # 兜底：出现在 <answer> 里
        return 1.0 if "<answer>yes" in out else 0.0

    def answer(self, box: Box, max_tokens: int = 400) -> Dict:
        """搜索结束后真正回答问题，并给出答案在原图上的位置（供 UI 打红点）。

        ！关键：手绘地图/遥感小图上的标注字很小，原尺寸送进去模型会"编名字"。
        这里先把图放大（默认 ×2，长边上限 3000）再送，并要求逐字辨认、不许编造。
        返回 {"text":..., "point":[u,v]|None, "reason":..., "raw":...}
        """
        from PIL import Image as _Image

        scale = float(os.environ.get("FOVEA_ANSWER_SCALE", "2.0"))
        max_side = int(os.environ.get("FOVEA_ANSWER_MAX_SIDE", "3000"))
        big = self.img
        if scale > 1.0:
            long_side = max(big.size) * scale
            if long_side > max_side:
                scale = max_side / max(big.size)
            if scale > 1.0:
                big = big.resize((int(big.size[0] * scale), int(big.size[1] * scale)), _Image.LANCZOS)

        text = (f"用户问题：{self.question}\n"
                f"第一张图是原图（第三张是系统选定的候选区域，仅供参考）。\n"
                f"步骤：① 先逐字读出图上能看清的建筑标注名称（字很小，看不清就跳过，**绝不编造**，最多写 6 个）；"
                f"② 比较各建筑的层数/体量，选出最高的那个；③ 在**原图**上指出它的位置。\n"
                f"输出严格 JSON，不要 markdown、不要多余文字：\n"
                f'{{"labels": ["最多 6 个你看清的楼名"], "answer": "楼名", "point": [u, v], "reason": "20 字内"}}\n'
                f"u,v 是原图归一化坐标（0~1，左上角为 0,0）。")
        body = {
            "model": self.answer_model,
            "messages": [
                {"role": "system", "content": "你是图像视觉问答助手，只输出严格 JSON。"},
                {"role": "user", "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + self._b64(big)}},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + self._b64(self._crop(box))}},
                ]},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        raw = self._chat(body).strip()
        out = {"text": raw, **parse_answer_json(raw, self.img.size)}
        out["upscale"] = round(scale, 2)
        # 自洽校验：答案必须能在它自己读到的标签里找到，否则大概率是编的
        labels = out.get("labels") or []
        ans = (out.get("text") or "").strip()
        norm = lambda s: "".join(ch for ch in s if ch.isalnum())
        na = norm(ans)
        out["verified"] = bool(na) and any(
            na and (na in norm(lb) or norm(lb) in na) for lb in labels if lb)
        return out

    def locate_in_box(self, target: str, box: Box, max_side: int = 900) -> Optional[List[float]]:
        """在最终框内部再定位一次：裁剪后放大再问，搜索空间小得多，比整图问准。"""
        from PIL import Image as _Image

        W, H = self.img.size
        x0, y0 = int(box[0] * W), int(box[1] * H)
        x1, y1 = int((box[0] + box[2]) * W), int((box[1] + box[3]) * H)
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None
        crop = self.img.crop((x0, y0, x1, y1))
        s = max(1.0, max_side / max(crop.size))
        if s > 1.0:
            crop = crop.resize((int(crop.width * s), int(crop.height * s)), _Image.LANCZOS)
        buf = io.BytesIO(); crop.save(buf, format="JPEG", quality=92)
        b64 = base64.b64encode(buf.getvalue()).decode()
        text = (f"这张裁剪图里，目标「{target}」在哪里？\n"
                f'只输出 JSON：{{"bbox":[x1,y1,x2,y2]}}，坐标按 0-1000 归一化到这张裁剪图。')
        body = {"model": self.answer_model,
                "messages": [{"role": "system", "content": "你只输出严格 JSON。"},
                             {"role": "user", "content": [
                                 {"type": "text", "text": text},
                                 {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}]}],
                "temperature": 0.0, "max_tokens": 80}
        try:
            raw = self._chat(body)
        except Exception:  # noqa: BLE001
            return None
        import re as _re
        m = _re.search(r"\[\s*[\d.]+\s*,\s*[\d.]+\s*,\s*[\d.]+\s*,\s*[\d.]+\s*\]", raw)
        if not m:
            return None
        try:
            bb = json.loads(m.group(0))
            cu = (float(bb[0]) + float(bb[2])) / 2000.0
            cv = (float(bb[1]) + float(bb[3])) / 2000.0
        except Exception:  # noqa: BLE001
            return None
        if not (0.0 <= cu <= 1.0 and 0.0 <= cv <= 1.0):
            return None
        gu = box[0] + box[2] * cu
        gv = box[1] + box[3] * cv
        return [round(gu, 4), round(gv, 4)]

def parse_answer_json(raw: str, size=None) -> Dict:
    """从模型回复里抠出 {answer, point, reason}；坐标可能是归一化也可能是像素。"""
    import re as _re
    m = _re.search(r"\{.*\}", raw, _re.S)
    if m:
        try:
            d = json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            d = {}
        ans = str(d.get("answer") or "").strip()
        pt = d.get("point")
        point = None
        if isinstance(pt, (list, tuple)) and len(pt) == 2:
            try:
                u, v = float(pt[0]), float(pt[1])
                if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0:
                    point = [u, v]                      # 归一化
                elif size and (u > 1 or v > 1):         # 像素坐标 → 归一化
                    W, H = size
                    point = [min(max(u / W, 0.0), 1.0), min(max(v / H, 0.0), 1.0)]
            except Exception:  # noqa: BLE001
                point = None
        if ans or point:
            labels = [str(x) for x in (d.get("labels") or []) if str(x).strip()][:20]
            return {"text": ans, "point": point, "labels": labels,
                    "reason": str(d.get("reason") or ""), "raw": raw}
    return {"text": raw, "point": None, "labels": [], "reason": "", "raw": raw}
