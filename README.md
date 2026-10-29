# dobot_plug — CR5AF 插座插拔 VLA 全流程（数据采集 · 训练 · 部署 · 评估）

越疆 CR5AF（V4 控制器，末端六维力）+ 多台 RealSense D435i + 大寰夹爪，
以 **[openpi](https://github.com/Physical-Intelligence/openpi)**（Physical Intelligence 官方 π0.5 实现）做 SFT，
用按论文复现的 **RECAP**（π\*0.6 的训练方法，arXiv:2511.14759）迭代提升成功率。

本项目是一条完整的 VLA pipeline：**遥操作采集 → LeRobot 数据集 → π0.5 微调 → 部署接管 → RECAP 迭代 → 固定布局评估**。

```
┌─────────────────── 机械臂电脑（Ubuntu + RealSense + 3D 鼠标）───────────────────┐
│                                                                                 │
│  ① 采集    record.py ──────────┐         ③ 部署    run_policy.py ──┐            │
│             3D 鼠标遥操作       │                    策略执行 + 人工接管 │            │
│                                ▼                                  ▼            │
│                        data/raw/<task>/<时间戳>/  （jpg + npz，唯一真值，        │
│                        meta.json 含成败/来源/实测fps/干预标记     ）             │
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     │ rsync 拷贝
┌────────────────────────────────────▼────────────── GPU 训练机（openpi）──────────┐
│  ② 数据处理   inspect_data.py（质检） → convert_to_lerobot.py → LeRobot 数据集   │
│  ③ 训练       compute_norm_stats → train.py pi05_dobot_plug(_lora)  → SFT ckpt  │
│  ④ RECAP      train_value.py → label_advantage.py → convert --with-advantage   │
│               → train.py pi05_dobot_plug_recap(_lora)（从 SFT ckpt 初始化）     │
│  ⑤ 服务       serve_policy.py（websocket 策略服务器，供机械臂电脑调用）          │
└────────────────────────────────────────────────────────────────────────────────┘
                                     │
                      ⑥ 评估  eval.py（固定布局 × N 回合 + Wilson 95% CI 报告）
```

## 上游开源基础（本项目的依赖与对标）

| 层 | 项目 | 用途 |
|---|---|---|
| 模型/训练 | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)（Apache-2.0） | π0.5 底座、训练/推理全链路；本项目通过 `openpi_patch/` 补丁接入 CR5AF 数据集与 4 个训练配置 |
| 数据格式 | [huggingface/lerobot](https://github.com/huggingface/lerobot) 数据集格式 | 训练数据以 LeRobot 数据集交付（openpi 锁定版本；转换脚本在 openpi 环境内运行） |
| 算法 | π\*0.6 / RECAP（[arXiv:2511.14759](https://arxiv.org/abs/2511.14759)，未开源） | 优势条件化迭代训练；本项目按论文自研实现（见 [docs/recap.md](docs/recap.md) 的逐条对照与偏差说明） |
| 机器人接口 | 越疆官方 [TCP-IP-Python-V4](https://github.com/Dobot-Arm/TCP-IP-Python-V4) SDK | CR5AF 控制/反馈；RealSense、大寰夹爪、3D 鼠标驱动为本项目实现 |

## 快速开始

```bash
# 安装（机械臂电脑全量；GPU 机只装基础依赖即可）
pip install -e ".[robot]"          # 或 pip install -r requirements.txt
# GPU 训练机准备 openpi：
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi && git checkout $(cat <dobot_plug>/openpi_patch/BASE_COMMIT)
git apply <dobot_plug>/openpi_patch/openpi_dobot.patch
GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv pip install torchvision==0.22.1
```

之后按文档走：

| 阶段 | 文档 | 命令 |
|---|---|---|
| 0 硬件自检 | [docs/hardware_setup.md](docs/hardware_setup.md) | `dobot-check-hardware --robot --servo` |
| 1 采集示范 | [docs/data_collection.md](docs/data_collection.md) | `dobot-record --task insert` |
| 2 数据质检/转换 | [docs/data_collection.md](docs/data_collection.md) | `dobot-inspect-data`；`dobot-convert-to-lerobot` |
| 3 SFT 训练与部署 | [docs/training.md](docs/training.md) | `train.py pi05_dobot_plug_lora` → `dobot-run-policy` |
| 4 RECAP 迭代 | [docs/recap.md](docs/recap.md) | `dobot-recap-train-value` → `dobot-recap-label-advantage` → … |
| 5 正式评估 | [docs/evaluation.md](docs/evaluation.md) | `dobot-eval --episodes 20 --policy-tag sft_v1` |
| 6 测试 | — | `pytest tests/test_offline.py tests/test_inspect_and_eval.py` |

> `scripts/` 下的同名薄壳（`python scripts/record.py …`）与安装后的 console 命令（`dobot-record …`）等价，任选其一。

## 目录结构

```
dobot_plug/
├── dobot_plug/                # 库代码
│   ├── robot.py               #   CR5AF：ServoJ 流式伺服、逆解、六维力、保护停
│   ├── cameras.py             #   多 RealSense 采集
│   ├── gripper.py             #   大寰 Modbus / 末端 IO / 无夹爪
│   ├── teleop.py              #   3D 鼠标 → 目标位姿 → IK → 目标关节角
│   ├── features.py            #   state / action 唯一定义（训练与部署共用）
│   ├── episode_io.py          #   原始回合格式（jpg + npz + meta）
│   ├── recap.py               #   RECAP 回报 / N 步优势 / 分 bin / 阈值
│   ├── policy_client.py       #   openpi websocket 客户端 + 动作块执行
│   ├── stats.py               #   Wilson 95% CI
│   └── apps/                  #   可执行入口（console_scripts 调这里）
│       ├── check_hardware.py  ├── record.py     ├── run_policy.py
│       ├── convert_to_lerobot.py  ├── inspect_data.py  ├── eval.py
│       └── recap/train_value.py  recap/label_advantage.py
├── scripts/                   # 薄壳（python scripts/xxx.py 与 dobot-xxx 等价）
├── configs/hardware.yaml      # 硬件/任务参数（IP、相机、夹爪、安全阈值）
├── configs/layouts.example.yaml # 评估固定布局清单模板
├── openpi_patch/              # openpi 补丁 + BASE_COMMIT
├── tests/                     # 离线/假控制器/主循环仿真三层测试
├── docs/                      # 各阶段文档
└── pyproject.toml             # 打包与依赖（extras: robot / recap / dev）
```

## 关键设计（先读）

| 项 | 选择 | 原因 |
|---|---|---|
| 控制方式 | **关节空间 ServoJ 流式**（30Hz 发送，15Hz 控制） | 策略直接输出目标关节角；遥操作也经控制器 `InverseKin` 变成关节角，**记录的动作 = 部署时下发的量** |
| 动作 | `[目标关节角 6 (rad), 夹爪 0开~1闭]`，训练时前 6 维转相对当前关节角的增量 | 与 openpi UR5/DROID 官方示例一致（`DeltaActions(make_bool_mask(6, -1))`，夹爪绝对值） |
| 动作标签 | 下发给机械臂的**指令**，不是下一帧实际位置 | 插入接触时“往里推”的意图会保留下来 |
| 状态 | `[q 6, 夹爪 1, 六维力 6]` = 13 维（`--no-ft` 可去掉力） | 插拔是接触任务，力信号能帮助判断是否对准/插到位 |
| 相机 → π0.5 槽位 | 主视角→`base_0_rgb`，腕部→`left_wrist_0_rgb`，侧视→`right_wrist_0_rgb` | π0.5 固定 3 个图像槽，单臂时借用右腕槽放第二个第三人称视角（不足槽位补零 + mask，与 DROID 策略一致） |
| 频率 / 动作块 | 15Hz，`action_horizon=15`（≈1 秒），每执行 8 步重新推理（异步，不停顿） | 与 pi05_droid 设置一致 |
| 归一化 | π0.5 分位数归一化（q01/q99），`compute_norm_stats.py` 生成 | openpi 对 PI0.5 模型自动启用分位数归一化 |
| RECAP 优势条件 | prompt 末尾加 `\nAdvantage: positive/negative`（训练时随机丢弃，论文做法） | 不改模型结构，部署时只改 prompt；详见 [docs/recap.md](docs/recap.md) |
| 安全 | 合力/力矩超限、TCP 越界、控制器报警 → 保护停（保持当前位置），`c` 键恢复 | 另外每周期关节增量限幅，策略目标偏离当前位置 >8° 时限幅 |
| 数据质量 | 每回合 meta 记录**实测 fps**、干预帧标记；`dobot-inspect-data` 出体检报告 | 数据集“声称”的 fps 不可信时训练/部署时间对齐会出错（LeRobot 社区经验） |

## 已知需要现场确认的点

见 [docs/hardware_setup.md](docs/hardware_setup.md) 末尾。**尚未在真实 CR5AF 上运行过**：控制器通信、ServoJ、逆解、反馈解析已用官方 SDK 对着模拟控制器验证；openpi 侧的变换用桩模块验证、补丁可在干净的 openpi 上 apply（已对官方 main 复核），但没在 GPU 上实际训练。第一次上真机请降低速度上限（`max_joint_vel_deg_s`、`max_lin_vel_mm_s`），手放在急停上。

## License

Apache-2.0（与 openpi 一致）。
