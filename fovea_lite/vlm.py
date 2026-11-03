# -*- coding: utf-8 -*-
"""fovea_lite.vlm —— 把合成探针换成真 VLM（OpenAI 兼容接口）。

用法（本地 vLLM 或任何兼容 /v1/chat/completions 的服务）：
    export FOVEA_BASE_URL=http://127.0.0.1:8000/v1
    export FOVEA_MODEL=Qwen/Qwen3-VL-8B-Instruct
    export FOVEA_API_KEY=EMPTY

    from fovea_lite.vlm import VLMCropAgent
    agent = VLMCropAgent()
    out = agent.run(image_path="p.jpg", question="图中最大的足球场在哪个位置？")
    print(out["answer"], out["box"], out["trajectory"])

设计要点（与论文对齐）：
  1) 复用一个「可解性探测」prompt：只允许 Yes/No，禁止回答原问题；
  2) 工具拦截：模型给出的 crop 坐标不直接执行，先经 BPA 策略精修；
  3) 历史即先验：把每轮 (box, score, observation) 追加进对话历史。
"""
from __future__ import annotations

import base64
import io
import json
import os
import random
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .geometry import Box, scale
from .policies import policy_bpa

PROBE_SYS = """你是一个“可解性评估器”。给定原始图像、一个候选裁剪区域、以及用户问题，
你只需判断：这个候选区域里的信息是否足以回答用户问题。
- 如果候选区域与问题所指的位置无关，回答 No。
- 如果区域太模糊、目标太小看不清，回答 No。
- 只输出 Yes 或 No，不要回答问题本身。
输出格式：<answer>Yes</answer> 或 <answer>No</answer>"""

AGENT_SYS = """你是一个带视觉工具的智能体。你有 crop / detect / segment / depth 四个工具。
规则：先思考再动作；前几轮优先 crop；对工具输出保持怀疑，必要时再看一次。
回答时用 <answer>...</answer> 包裹最终答案。"""


@dataclass
class VLMCropAgent:
    base_url: str = field(default_factory=lambda: os.environ.get("FOVEA_BASE_URL", "http://127.0.0.1:8000/v1"))
    model: str = field(default_factory=lambda: os.environ.get("FOVEA_MODEL", "Qwen/Qwen3-VL-8B-Instruct"))
    api_key: str = field(default_factory=lambda: os.environ.get("FOVEA_API_KEY", "EMPTY"))
    max_turns: int = 10
    probe_n: int = 3
    temperature: float = 1.0

    # ---------------- 低层：一次多模态请求 ----------------
    def _chat(self, system: str, text: str, images: List[str]) -> str:
        content = [{"type": "text", "text": text}]
        for b64 in images:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": 512,
        }
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    # ---------------- 图像与裁剪 ----------------
    @staticmethod
    def _encode(img) -> str:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=92)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _crop(img, box: Box):
        W, H = img.size
        u, v, w, h = box
        return img.crop((int(u * W), int(v * H), int((u + w) * W), int((v + h) * H)))

    # ---------------- 探针 ----------------
    def probe_score(self, img, question: str, box: Box, rng: random.Random) -> float:
        """采样 probe_n 次 Yes/No，返回 Yes 比例（论文 Eq.10 的 Ĵ）。"""
        b64 = self._encode(self._crop(img, box))
        text = (f"用户问题：{question}\n"
                f"候选区域已作为第二张图给出（第一张为原图）。请判断该区域信息是否足以回答问题。")
        hits = 0
        for _ in range(self.probe_n):
            out = self._chat(PROBE_SYS, text, [self._encode(img), b64])
            hits += 1 if "yes" in out.lower() and "no" not in out.lower().replace("not", "") else 0
        return hits / self.probe_n

    # ---------------- 主流程 ----------------
    def run(self, image_path: str, question: str) -> Dict:
        from PIL import Image  # 延迟导入，合成实验不需要 Pillow

        img = Image.open(image_path)
        rng = random.Random(0)
        traj: List[Dict] = []

        # 第 1 轮：让模型给一个种子裁剪（这里用一个保守的中心框做默认种子，
        # 真实部署时应解析模型输出的坐标；见 README 的“工具拦截”一节）
        seed: Box = (0.25, 0.25, 0.5, 0.5)
        rec = policy_bpa(seed, lambda b, n: self.probe_score(img, question, b, rng), rng)
        traj.append({"box": rec.chosen, "probes": rec.probes, "mode": "bpa"})

        # 用最终裁剪 + 历史去作答（历史即近似后验）
        final_b64 = self._encode(self._crop(img, rec.chosen))
        hist_text = (f"用户问题：{question}\n"
                     f"历史：共探测 {rec.probes} 次，最终选定的裁剪框为 {rec.chosen}（归一化坐标）。\n"
                     f"请基于这张裁剪图给出最终答案。")
        answer = self._chat(AGENT_SYS, hist_text, [self._encode(img), final_b64])
        return {"answer": answer, "box": rec.chosen, "probes": rec.probes, "trajectory": traj}
