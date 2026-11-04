# FOVEA 主动视觉取证 · 可运行实现 + 在线演示服务

把论文 *The Perceptual Bandwidth Bottleneck in Vision-Language Models: Active Visual Reasoning via Sequential Experimental Design*（ICML 2026）的**"看哪里"搜索机制**做成能跑的程序，并做了一处可验证的改进（HRD-PB）。

## 它是什么

给定一张图和一句问题，系统不是"看一眼就答"，而是**反复裁剪 + 探测再决定**：

1. 用一个 VLM 对候选裁剪问 Yes/No：**"这块区域里，目标是否完整且清晰地出现？"**（盲探：只给裁剪，不给原图）
2. 多次采样取 Yes 比例作为该区域的"可解性"，据此决定下一步裁哪里；
3. 结束时用最终区域真正答题。

三种搜索策略可对比：

| 策略 | 特点 | 典型步数 |
|---|---|---|
| `greedy` | 在 seed 内三尺度缩放，一步定生死（不平移） | 3 |
| `lookahead` | 会回退试探（面积轨迹有回升） | 6 |
| `hrd` | seed 内 3×3 网格 → 逐层收缩 → 叶子验证回退 | 19+ |

**seed 即搜索范围**：所有策略只在用户画出的种子框内搜索，绝不外扩。

## 功能

- **搜索与对比**：单跑或勾 ≥2 种一键"并行实验"，输出轨迹、收缩曲线、指标表
- **答案标注**：命中真值→红箭头指向目标；未命中→不画箭头并显示「✗ 未找到目标」；无真值→指向最终框中心
- **自动审查**：设置真值点后自动判定命中、累计命中率
- **访客/参与者/管理员**三级访问控制，密钥绑定姓名
- **后台**：队列、worker 在线状态、命中率、最近任务（含操作人）
- **任务取消**：点「重置视图」立即取消并清空，不留僵尸任务
- **两种后端**：合成场景服务器本地算（0 成本）；真实图片交给 worker 调 VLM

## 快速开始

```bash
pip install pillow
python -m fovea_service.server --host 0.0.0.0 --port 8765 --mode inline --backend sim
```

打开 `http://localhost:8765/`。接真 VLM：

```bash
export FOVEA_BASE_URL=https://api.siliconflow.cn/v1
export FOVEA_API_KEY=<你的 key>
export FOVEA_MODEL=Qwen/Qwen3-VL-8B-Instruct
export FOVEA_ANSWER_MODEL=Qwen/Qwen3-VL-32B-Instruct
python -m fovea_service.worker --server http://127.0.0.1:8765 --token <WORKER_TOKEN> --backend vlm --concurrency 2
```

## 配置

| 环境变量 | 作用 | 默认 |
|---|---|---|
| `FOVEA_MODEL` | 探测用模型 | `Qwen/Qwen3-VL-8B-Instruct` |
| `FOVEA_ANSWER_MODEL` | 答题/定位用模型 | 同 `FOVEA_MODEL` |
| `FOVEA_ANSWER_SCALE` | 答题前放大倍数 | `2.0` |
| `FOVEA_MAX_SIDE` | 上传图长边上限 | `4096` |
| `FOVEA_TIMEOUT` / `FOVEA_RETRIES` | 单次调用超时/重试 | `60` / `3` |
| `FOVEA_KEYS` | 访问密钥文件路径 | `../.keys.json` |

## 目录

```
fovea_service/      Web 服务：server / worker / auth / db / backends / static
fovea_lite/         离线算法：合成世界、基线策略、HRD-P、评测入口
deploy/             部署脚本（bootstrap / nginx / systemd / worker 看门狗）
docs/fovea_lite.md  算法实现与实验结论
```

## 设计要点（踩过的坑）

- **探针必须盲探**：同时给原图和裁剪，模型在原图里已经看到目标，会对任何区域都判 Yes，搜索失去梯度（分数全 1.00）。只给裁剪后，负样本能正确判否。
- **seed 是硬边界**：早期 hrd 忽略 seed、greedy 还会放 1.5 倍，导致"设了种子框仍搜全图"。
- **标注跟着最终结果走**：先判定命中再标注；未命中不画箭头。
- **取消要三方一致**：服务端立即落终态、客户端停轮询并丢弃迟到响应、执行端协作中止。
- **worker 要并发 + 看门狗**：长任务（hrd 约 110 s）单线程会堵住队列；进程崩溃要能自动拉起。
- **移动端**：canvas 必须 `touch-action:none`，否则拖动被当成滚屏；用 Pointer Events + `setPointerCapture`。

## 许可

仅用于研究与教学演示。论文版权归原作者；本仓库为独立实现，非官方代码。
