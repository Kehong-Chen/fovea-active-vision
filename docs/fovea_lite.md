# fovea_lite —— 感知带宽约束下的主动视觉取证（可运行精简实现）

论文：*The Perceptual Bandwidth Bottleneck in Vision-Language Models: Active Visual
Reasoning via Sequential Experimental Design*（ICML 2026，FOVEA）。

本仓库不是逐行复现，而是**把论文的机制做成能跑的程序**，并在它自己承认的一个
未解问题（冷启动 / 选择器不可靠）上给出一处可验证的改进。

## 论文三步（本质）

1. **把"看哪里"写成实验设计**：一次裁剪 d 的价值 = 看见目标的概率
   `P(S=1|d) = 1[目标在框内] × φ(d)`，其中 `φ(d)=sigmoid(k·(有效分辨率−阈值))`。
2. **化简成几何量**：在"位置/语义解耦 + 校准可见 + 理想观察者"三条近似下，
   信息增益 ∝ `J(d) = 覆盖率 × 分辨率`。覆盖项 = 当前信念在框内的质量。
3. **用一句 Yes/No 探测估它**：`Ĵ(d) = P(VLM 说"这块够答" | 裁剪, 问题)`，
   采样 3 次取均值；拦截 agent 的 crop 调用，把精修后的框执行，观测写回历史。
   全程不训练。

论文结果：30B 基座综合 73.3 → 77.7；遥感千兆像素 ReAct 45.1 → Lookahead 54.7
（人工 oracle 裁剪 68.0）；代价是输入 token ×6.5–9.5、输出 token ×9.8–55。

## 目录

```
fovea_lite/
  geometry.py   归一化框运算 + 有效分辨率推导（对应 ρ=B/A）
  world.py      合成"千兆像素"场景 + 观测量模型（论文 Sec.2 的可解析替身）
  policies.py   基线：react / greedy3 / mcmc / lookahead / multiseed9
  hrd.py        本报告的小突破：HRD-P / HRD-PB（分层探测 + 先验质量 + 验证回退）
  registry.py   策略注册表（统一入口）
  sim.py        单次对比实验（可指定 seed / 场景数）
  evaluate.py   唯一评测入口（产出报告里所有数字）
  tune.py       调参集扫参（m / depth / k_cell / prior_w）
  tune2.py      调参集扫参（HRD-PB 的 depth / max_bt）
  analyze.py    备用统计脚本（McNemar / bootstrap / 前沿）
  vlm.py        把合成探针换成真 VLM（OpenAI 兼容接口）的适配器
```

## 快速开始

```bash
# 一行出全部数字（约 2–4 分钟，纯 CPU）
PYTHONPATH=. python -m fovea_lite.evaluate

# 只跑两三个策略看行为
PYTHONPATH=. python -m fovea_lite.sim --scenes 300 --seed 1 \
    --policies "react,greedy3,hrd_pb"
```

产出：`results_final.json / .csv`（主表）、`results_analysis.json`（检验与前沿）。

## 结果摘要（600 场景，seed=1，oracle=0.900）

| 策略 | 准确率 | 探测次数 | 说明 |
|---|---|---|---|
| react（直接执行种子框） | 0.563 | 0 | 44.5% 的场景目标根本没进框（冷启动） |
| greedy3（论文默认档） | 0.572 | 9.0 | 三尺度 + 3 次探测 |
| mcmc | 0.568 | 12.7 | 局部扰动，收益有限 |
| lookahead | 0.560 | 18.0 | 一步前瞻，成本翻倍收益未现 |
| multiseed9（论文附录 H.2 的补救） | 0.550 | 81.0 | 9 种子×3 尺度：**更多候选反而更差** |
| bpa_oneshot（本仓库试错版，保留为反例） | 0.438 | 15.0 | 粗扫后一次性缩放：定位没解决 |
| hrd_p_noprior（消融：去掉覆盖项） | 0.392 | 18.0 | 二值探测打平 → 选错格 |
| hrd_p（只下降，不回退） | 0.595 | 18.0 | 比贪心 +0.023，**统计不显著**（p=0.34） |
| **hrd_pb（本报告方法）** | **0.723** | 27.2 | 小目标 0.635（贪心 0.380） |

配对检验（McNemar 精确双尾）：hrd_pb vs greedy3 Δ=+0.152（95% CI [0.103,0.198]，
p=6.8e-10）；vs multiseed9 Δ=+0.173（p=6.6e-13）。

## 接真 VLM

```bash
export FOVEA_BASE_URL=http://127.0.0.1:8000/v1   # vLLM / 任何 OpenAI 兼容服务
export FOVEA_MODEL=Qwen/Qwen3-VL-8B-Instruct
export FOVEA_API_KEY=EMPTY
```

```python
from fovea_lite.vlm import VLMCropAgent
agent = VLMCropAgent()
out = agent.run(image_path="sat.jpg", question="图中最大的足球场在哪里？")
print(out["answer"], out["box"], out["probes"])
```

探针 prompt 强制只允许 Yes/No（`PROBE_SYS`），与论文附录 J 的角色分离一致；
crop 调用被 `policy_bpa` 精修后再执行（工具拦截），观测写回历史。

## 局限（写清楚，不藏）

- 结论来自**合成观测模型**，不是真 VLM。它的价值是隔离"搜索策略"——论文自己用
  oracle 裁剪证明搜索与识别是两个独立瓶颈，所以这层结论可迁移，但绝对数值不可
  与论文表格直接比较。
- 本仓库的 "multiseed9" 是围绕种子做扰动生成的代理，弱于论文里由 VLM 给出的
  9 个提案；因此"多种子被支配"应读作"在种子本身不可靠时，扩种子数量不够"。
- 识别瓶颈固定为 0.90，所有策略共享，故比较公平但不反映真实识别误差分布。
- 预算以"探测次数"计，未折算真实 token（论文实测 ×6.5–9.5 输入、×9.8–55 输出）。
