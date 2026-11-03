# -*- coding: utf-8 -*-
"""fovea_lite —— 感知带宽约束下的主动视觉取证（论文 FOVEA 的可运行精简实现）。

模块：
  geometry  归一化框运算（缩放/扰动/网格/有效分辨率）
  world     合成“千兆像素”场景 + 观测量模型（论文 Sec.2 的可解析替身）
  policies  react / greedy3 / mcmc / lookahead / multiseed9 / bpa(本报告的小突破)
  sim       实验台：同一批场景跑所有策略，输出主表、失败分解、尺寸分桶
  vlm       把合成探针换成真 VLM（OpenAI 兼容接口）的适配器

论文对应关系：
  J(d) = 覆盖率 × 分辨率   -> geometry.tokens_for_target + world.resolve_prob
  Ĵ(d) = P(Yes | I_d, Q)   -> world.probe_yes_prob（合成）/ vlm.probe_score（真模型）
  工具拦截 + 历史即先验     -> vlm.VLMCropAgent.run
"""
__all__ = ["geometry", "world", "policies", "sim", "vlm"]
__version__ = "0.1.0"
