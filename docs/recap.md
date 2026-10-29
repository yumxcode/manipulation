# RECAP 迭代（π\*0.6 训练方法的自研实现）

论文：**"π\*0.6: a VLA That Learns From Experience"**（Physical Intelligence，arXiv:2511.14759，2025）。
RECAP = RL with Experience & Corrections via Advantage-conditioned Policies。
**openpi 未开源该算法**（截至 2026-10，issue #857 仍无 ETA），本项目按论文复现，本文档逐条对照实现与论文。

## 1. 循环

```
 ┌─ run_policy.py 跑一批（每任务 50~100 回合，含失败与接管）
 │      ↓
 │  train_value.py      价值模型：预测“离成功还有多远”（失败回合额外惩罚）
 │      ↓
 │  label_advantage.py  A_t = (G_t − G_{t+N}) + V(o_{t+N}) − V(o_t)，每任务取阈值 ε_ℓ
 │      ↓
 │  convert_to_lerobot.py --with-advantage   全部数据（示范+rollout，成功+失败）
 │      ↓
 │  train.py pi05_dobot_plug_recap(_lora)    每轮都从 SFT checkpoint 初始化（论文做法，防漂移）
 │      ↓
 └─ run_policy.py --recap   （prompt 末尾带 "Advantage: positive"）
```

```bash
# GPU 机
dobot-recap-train-value --raw-dir <data/raw> --out checkpoints/value/r1
dobot-recap-label-advantage --raw-dir <data/raw> --value checkpoints/value/r1/best.pt
uv run python ~/dobot_plug/scripts/convert_to_lerobot.py --raw-dir <data/raw> --repo-id local/dobot_plug_recap \
    --sources demo rollout --include-failures --with-advantage
uv run scripts/compute_norm_stats.py --config-name pi05_dobot_plug_recap_lora
DOBOT_RECAP_INIT=checkpoints/pi05_dobot_plug_lora/sft_v1/29999/params \
    uv run scripts/train.py pi05_dobot_plug_recap_lora --exp-name=recap_r1 --overwrite
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_dobot_plug_recap_lora --policy.dir=checkpoints/pi05_dobot_plug_recap_lora/recap_r1/9999
# 机械臂电脑
dobot-run-policy --task insert --host <GPU机IP> --recap --policy-tag recap_r1
```

## 2. 实现与论文逐条对照

| # | 论文（arXiv:2511.14759） | 本项目实现 | 状态 |
|---|---|---|---|
| 1 | 回报：每步 −1，成功终止 0，失败终止 −C_fail，按任务最大步长归一化到 (−1,0) | `recap.normalized_returns`：`G_t=-(T-1-t)-[失败]·C_fail`，除以 `L_max+C_fail` | ✅ 一致 |
| 2 | 价值函数为**分布式**（201 bins），以离散化 MC 回报的交叉熵训练 | `Bins(201)` + two-hot 交叉熵（`recap/train_value.py`） | ✅ 一致 |
| 3 | N 步优势 `A_t = Σ_{t..t+N-1} r + V(o_{t+N}) − V(o_t)` | `recap.n_step_advantage`（尾部用 MC 回报 `G_t − V(o_t)`） | ✅ 一致；N 论文未披露，取 30（15Hz 下 2 秒） |
| 4 | 二值化：`I_t = 1(A_t > ε_ℓ)`，**ε_ℓ = 优势值的 30% 分位数 ⇒ ~70% positive** | `recap.positive_threshold`；`label_advantage.py --positive-fraction` 默认 **0.7** | ✅ 一致（已从旧默认 0.3 修正） |
| 5 | 人工接管帧强制 `I_t=True` | rollout 回合 `intervention=1` 帧强制 positive | ✅ 一致 |
| 6 | 演示/SFT 阶段 `I_t=True` | `label_advantage.py` 默认演示帧全 positive（`--score-demos` 可改为按优势打分） | ✅ 一致 |
| 7 | 指示符为文本 `"Advantage: positive/negative"`，位于任务文本之后、动作之前 | prompt 末尾追加 `"\nAdvantage: positive"`（π0.5 中 prompt 就是任务文本前缀，动作 token 在其后） | ✅ 语义一致 |
| 8 | 训练时随机丢弃指示符（概率未披露），推理 β=1 直接用 positive | 补丁 `AdvantagePrompt(dropout=0.3)`；`run_policy.py --recap` 只加 positive 后缀 | ✅ 机制一致；dropout=0.3 为自选值 |
| 9 | 每轮迭代 V 和 π 都从**预训练** checkpoint 重训（防漂移） | 每轮 `DOBOT_RECAP_INIT=<SFT checkpoint>`（π/π_pre 的对应物），价值模型每轮重训 | ✅ 一致 |
| 10 | 价值模型：670M Gemma3 VLM 骨干 + web 数据共训；训练 π 时**在线**算优势 | ResNet18/DINOv2 + state + 任务嵌入，离线标注（`label_advantage.py`） | ⚠️ 简化（见下） |

### 与论文的有意偏差（单任务场景的工程取舍）

1. **价值模型骨干**：论文用小型 VLM（670M）；本项目单任务（插/拔）用 torchvision 预训练 ResNet18
   （可选 DINOv2），训练快（单卡几十分钟）。论文经验"价值函数要能跨任务判断进度"，我们任务单一，
   轻骨干足够；`val_corr > 0.8` 是验收线。
2. **优势标注离线**：论文在 VLA 训练中 on-the-fly 打分；我们预先算好 `advantage.npz` 再转数据集。
   对固定数据集等价，代价是每轮重标一次（本来就每轮重训价值模型）。
3. **SFT 阶段没有指示符**：论文的 π\*0.6 底座在预训练时就带优势指示符输入，SFT 时 `I=True`；
   我们从 π0.5（无指示符）出发，SFT 阶段用纯 prompt，RECAP 轮才引入指示符——相当于把论文的
   "SFT(I=True)" 折叠进 RECAP 轮的演示数据处理（演示帧全 positive，见对照 #6）。
4. **未披露超参**：N、指示符 dropout、C_fail 的具体取值论文正文/附录均未给出，本项目取
   `N=30`（2 秒）、`dropout=0.3`、`C_fail = 最长回合步数`，均已做成 CLI 参数可调。

## 3. 检查点（每轮）

- `label_advantage.py` 结尾按 任务/来源/成败 打印 positive 比例。**失败回合的 positive 比例应明显低于成功回合**，否则价值模型没学好（失败样本太少/数据太少）。
- 价值模型训练日志 `val_corr` 应 > 0.8。
- SFT 与 RECAP 用同一种变体（都 LoRA 或都全量）。
- 用[评估协议](evaluation.md)在同一组固定布局上比较，只保留真正提升的 checkpoint。

## 4. 数据规模参考（论文 VI-C2）

论文单任务每轮 **300~600 条轨迹**（4 台机器人并行）、1~2 轮即显著提升；box 装配每轮
自主 600 + 干预 360。本项目起步：每任务每轮 **50~100 回合**（含失败与接管）。

## 5. 相关但不同的东西

- **RTC / 实时引导**（lerobot `--inference.type=rtc`）：部署期共享控制，人把引导混叠进动作块——与本项目的"推 3D 鼠标自动接管"同属 human-gated 干预，但 RECAP 用的是**训练期**优势条件化。
- **AWR/PPO**：论文 VI-C3 对比过，均不如 advantage-conditioning（AWR 慢、PPO off-policy 不稳需信任域）；我们不做这些基线的实现。
