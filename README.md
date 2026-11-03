# FOVEA 主动视觉取证 · 可运行实现 + 在线演示服务

把论文 *The Perceptual Bandwidth Bottleneck in Vision-Language Models: Active Visual Reasoning via Sequential Experimental Design*（ICML 2026）的**"看哪里"搜索机制**做成能跑的程序，并做了一处可验证的改进（HRD-PB），外加一套可以直接开给别人用的 Web 演示服务。

## 它是什么

给定一张图和一句问题，系统不是"看一眼就答"，而是**反复裁剪 + 探测再决定**：

1. 用一个 VLM 对候选裁剪问一句 Yes/No：**"这块区域的信息够不够回答问题？"**
2. 把多次采样的 Yes 比例当作该区域的"可解性"，据此选下一步裁哪里；
3. 结束时用最终选定的区域真正答题。

三种搜索策略可对比：

| 策略 | 特点 | 典型步数 |
|---|---|---|
| `greedy` | 三尺度逐级放大缩小，一步定生死 | 3 |
| `lookahead` | 会回退试探（面积轨迹有回升） | 6 |
| `hrd` | 先 3×3 网格粗扫，再局部细扫 + 验证 | 19 |

`fovea_lite/` 里另有离线评测实现（纯 CPU、合成观测模型、600 场景约 2–4 分钟），用来在**不花钱、可复现**的条件下比较搜索策略。

## 功能

- **搜索与对比**：单跑一种策略，或多选 ≥2 种一键"并行实验"，输出轨迹图、收缩曲线、指标表
- **答案标注**：红色箭头指向答案位置；合成场景用蓝色「测试真值」标签标注原定真值
- **自动审查**：可选设置真值点，自动判定命中并累计命中率
- **访问控制**：访客（每日 3 次）/ 参与者（密钥，无限次）/ 管理员（额外后台）
- **后台**：队列、worker、命中率、内存磁盘、最近任务（含操作人）
- **两种后端**：合成场景在服务器本地算（0 成本）；真实图片交给外部 worker 调 VLM

## 快速开始

```bash
pip install pillow
python -m fovea_service.server --host 0.0.0.0 --port 8765 --mode inline --backend sim
```

打开 `http://localhost:8765/`，先选进入方式（访客可直接进）。

接真 VLM（任意 OpenAI 兼容端点，如硅基流动 / vLLM）：

```bash
export FOVEA_BASE_URL=https://api.siliconflow.cn/v1
export FOVEA_API_KEY=<你的 key>
export FOVEA_MODEL=Qwen/Qwen3-VL-8B-Instruct          # 探测用，便宜
export FOVEA_ANSWER_MODEL=Qwen/Qwen3-VL-32B-Instruct  # 答题用，读小字更准
python -m fovea_service.worker --server http://127.0.0.1:8765 --token <WORKER_TOKEN> --backend vlm
```

不接外部模型也能跑：示例图走本地合成观测模型（`SimBackend`），秒级、免费、可重复。

## 配置

| 环境变量 | 作用 | 默认 |
|---|---|---|
| `FOVEA_MODEL` | 探测用模型 | `Qwen/Qwen3-VL-8B-Instruct` |
| `FOVEA_ANSWER_MODEL` | 答题/定位用模型 | 同 `FOVEA_MODEL` |
| `FOVEA_ANSWER_SCALE` | 答题前放大倍数（小字更清楚） | `2.0` |
| `FOVEA_MAX_SIDE` | 上传图长边上限 | `4096` |
| `FOVEA_TIMEOUT` / `FOVEA_RETRIES` | 单次调用超时/重试 | `60` / `3` |
| `FOVEA_KEYS` | 访问密钥文件路径 | `../.keys.json` |

首次启动会自动生成 11 个访问密钥并打印一次（存 `<.keys.json>`，权限 600）。

## 目录

```
fovea_service/      Web 服务：server / worker / auth / db / backends / static
fovea_lite/         离线算法：合成世界、四种基线策略、HRD-P、评测入口
deploy/             部署脚本（bootstrap / nginx / systemd / worker 常驻）
docs/fovea_lite.md  算法实现与实验结论说明
```

## 设计要点（踩过的坑）

- **探针只回答 Yes/No**，最终答案必须单独再问一次；否则只有框没有答案。
- **进度要实时上报**：worker 每步 `POST /api/progress`，否则前端全程 0 步 0 秒，看起来像卡死。
- **别让模型猜坐标**：直接回归位置、网格选格、二分法都试过，会偏一格甚至偏到无关区域；改用"自读标签自洽校验"，或直接以搜索框/场景真值为准。
- **加鉴权要盘点机器客户端**：worker 下载图片也要带 token，否则整条推理链 401。
- **前端并发要加互斥锁**：否则同时挂两条轮询链，任务面板会串。

## 许可

仅用于研究与教学演示。论文版权归原作者；本仓库为独立实现，非官方代码。
