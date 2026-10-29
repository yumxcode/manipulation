# 数据采集与质检

原则：**`data/raw/` 是唯一真值**。原始回合（jpg + npz + meta.json）记录了采集中的一切（包括失败回合、
人工接管帧、实测帧率），所有训练数据集都从这里重新生成，可以随时用不同过滤参数重转。

## 1. 采集示范（阶段 1）

```bash
dobot-record --task insert --operator 张三
```

`空格` 开始 → 3D 鼠标操作 → `s` 成功 / `f` 失败 / `d` 丢弃；`t` 切换插/拔任务（插完直接录拔，自然复位）。
左键夹爪开合，**按住右键精细模式**（插入最后几毫米用）。

### 数量建议（依据开源社区经验）

| 依据 | 结论 |
|---|---|
| openpi 维护者（issue #491） | 变异不大的任务 **100 条示范**是不错的起点 |
| LeRobot 官方教程 | ≥50 集，**每个位置 10 集**；先窄分布，稳定后再加变化 |
| ALOHA/ACT 传统 | 每任务 50 条起步，难任务 100 条 |
| π0 论文真机微调 | 最简单任务 5 小时数据，最复杂 100+ 小时 |

本项目建议：**插、拔各 100~150 条起步**（接触任务比拾放难，向上取）；插座位置、插头初始姿态、
抓取位置都要有变化（每 ~10 条换一次布局）。操作风格保持一致：先对准、再沿插入方向直推，
不要来回试探太多次。**失败的示范也按 `f` 保存**——RECAP 价值模型需要失败回合来学"离失败有多近"。

### 质量清单（LeRobot 官方数据集博客）

- 画面稳定不抖、光照中性一致，背景静态；3D 鼠标线和手不入画；
- 画面中唯一运动的应是机械臂与插头/插座；
- 存盘分辨率 320×240 起步（模型端 resize/pad 到 224×224，与 π0.5 SigLIP 输入一致）；
- **采集完看节拍**：每回合 `meta.json` 的 `fps_measured` 应接近标称 15Hz（见下文质检）。

## 2. 数据质检

```bash
dobot-inspect-data                 # 汇总 + 实测帧率 + 干预率 + 图像抽查警告
dobot-inspect-data --montage       # 另存抽样回合的 6 帧拼图（人工过目用）
dobot-inspect-data --json rep.json # 机器可读报告
```

检查项：
- 按 任务/来源/成败 分组的回合数、帧数、小时数；
- **节拍（cadence）**：实测 fps 明显低于标称（<90%）会逐条警告——数据集会"声称"标称 fps，
  实际低了说明控制循环被相机/串口阻塞，训练与部署的时间对齐会出错（LeRobot 官方反复强调这一点）；
- rollout 干预率：RECAP 数据健康度；
- 图像抽查：全黑/过曝/无内容警告。

## 3. 同步到 GPU 机

```bash
rsync -av --prune-empty-dirs --include '*/' --include 'meta.json' --include '*.npz' \
    --include '*.jpg' --exclude '*' data/raw/ gpu:/path/dobot_plug/data/raw/
```

## 4. 转换为 LeRobot 数据集（GPU 机，openpi 环境里）

```bash
cd ~/openpi
# 阶段 1 SFT：只用成功的示范
uv run python ~/dobot_plug/scripts/convert_to_lerobot.py --repo-id local/dobot_plug_sft
# 阶段 2 RECAP：示范 + 自主执行(含人工接管)，成功和失败都要，带 advantage 字段
uv run python ~/dobot_plug/scripts/convert_to_lerobot.py --repo-id local/dobot_plug_recap \
    --sources demo rollout --include-failures --with-advantage
```

转换要点（与 openpi 官方 LIBERO 转换示例一致）：
- 在 openpi 的环境里跑（openpi 锁定的 lerobot 版本提供 `LeRobotDataset.create/add_frame/save_episode` API，
  导入路径是 `lerobot.common.datasets.lerobot_dataset`——**不要**用新版独立安装的 lerobot 来跑转换）；
- 数据集落在 `$HF_LEROBOT_HOME/<repo-id>`（默认 `~/.cache/huggingface/lerobot/`）；
- 默认丢弃保护停期间的帧、裁剪首尾静止帧（`--keep-hold` / `--no-trim` 可关）；
- `--no-ft` 与部署侧 `--no-ft` 必须一致（state 13 维 vs 7 维）。

## 5. 部署期数据（阶段 2 的另一半）

`dobot-run-policy` 跑策略时的每个回合（含人工接管）自动存为 `source=rollout`，
接管帧标 `intervention=1`——这就是 RECAP 需要的"经验 + 纠正"数据（HG-DAgger 式）。
字段命名与 LeRobot 生态 HIL 惯例一致（`intervention` 列）。

正式评估产生的回合用 `dobot-eval` 记录（meta 里 `eval=1` + 布局描述），同样并入 raw。
