# dobot_plug — CR5AF + RealSense D435i + openpi π0.5 + RECAP 插头插拔

越疆 CR5AF（V4 控制器，末端六维力）+ 多台 D435i，用 openpi 的 π0.5 做 SFT，再用自己实现的 RECAP（π\*0.6 的训练方法）迭代提升成功率。

```
dobot_plug/
├── configs/hardware.yaml        # 所有硬件/任务参数（IP、相机序列号、夹爪、遥操作、安全阈值）
├── dobot_plug/                  # 驱动与公共代码
│   ├── robot.py                 # CR5AF：ServoJ 流式伺服线程、InverseKin、力传感器、保护停
│   ├── cameras.py               # 多 RealSense 采集
│   ├── gripper.py               # 大寰 Modbus / 末端 IO / 无夹爪
│   ├── teleop.py                # 3D 鼠标 → 目标位姿 → 控制器逆解 → 目标关节角
│   ├── features.py              # state / action 的唯一定义（训练与部署共用）
│   ├── episode_io.py            # 原始回合格式（jpg + npz）
│   └── recap.py                 # 回报、N 步优势、分 bin
├── scripts/
│   ├── check_hardware.py        # 硬件自检
│   ├── record.py                # 遥操作采集示范
│   ├── convert_to_lerobot.py    # 原始回合 → LeRobot 数据集
│   ├── run_policy.py            # 部署 + 人工接管 + 自动录制 rollout
│   └── recap/
│       ├── train_value.py       # 价值模型
│       └── label_advantage.py   # 优势标注
├── openpi_patch/openpi_dobot.patch   # 给 openpi 加 Dobot 数据变换 + 4 个训练配置
└── tests/                       # 离线测试（假控制器/假相机/假策略服务器）
```

---

## 0. 关键设计（先读）

| 项 | 选择 | 原因 |
|---|---|---|
| 控制方式 | **关节空间 ServoJ 流式**（30Hz 发送，15Hz 控制） | 策略直接输出目标关节角；遥操作也经控制器 `InverseKin` 变成关节角，**记录的动作 = 部署时下发的量** |
| 动作 | `[目标关节角 6 (rad), 夹爪 0开~1闭]`，训练时前 6 维转为相对当前关节角的增量 | 与 openpi UR5/DROID 示例一致 |
| 动作标签 | 下发给机械臂的**指令**，不是下一帧实际位置 | 插入接触时“往里推”的意图会保留下来 |
| 状态 | `[q 6, 夹爪 1, 六维力 6]` = 13 维（`--no-ft` 可去掉力） | 插拔是接触任务，力信号能帮助判断是否对准/插到位 |
| 相机 → π0.5 槽位 | 主视角→`base_0_rgb`，腕部→`left_wrist_0_rgb`，侧视→`right_wrist_0_rgb` | π0.5 固定 3 个图像槽，单臂时借用右腕槽放第二个第三人称视角 |
| 频率 / 动作块 | 15Hz，`action_horizon=15`（≈1 秒），每执行 8 步重新推理（异步，不停顿） | 与 pi05_droid 设置一致 |
| RECAP 优势条件 | 在 prompt 末尾加 `\nAdvantage: positive/negative`（训练时 30% 丢弃） | 与 π\*0.6 一致，**不改模型结构**，部署时只改 prompt |
| 安全 | 合力/力矩超限、TCP 越界、控制器报警 → 保护停（保持当前位置），`c` 键恢复 | 另外每周期关节增量限幅，策略目标偏离当前位置 >8° 时限幅 |

---

## 1. 安装

### 机械臂电脑（接 CR5AF、相机、3D 鼠标、夹爪）
```bash
sudo apt install libhidapi-dev
pip install -r requirements.txt
pip install -e <openpi>/packages/openpi-client
# 3D 鼠标权限（3Dconnexion 厂商 ID 256f / 046d）
echo 'KERNEL=="hidraw*", ATTRS{idVendor}=="256f", MODE="0666"' | sudo tee /etc/udev/rules.d/99-spacemouse.rules
sudo udevadm control --reload && sudo udevadm trigger
```
网卡设到 `192.168.5.x`，CR5AF 默认 `192.168.5.1`。示教器 / DobotStudio Pro 里切到 **TCP/IP 二次开发模式**。

### GPU 训练机（Ubuntu 22.04）
```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git && cd openpi
git checkout $(cat <dobot_plug>/openpi_patch/BASE_COMMIT)   # 补丁基于此 commit；更新的 main 一般也能直接 apply
git apply <dobot_plug>/openpi_patch/openpi_dobot.patch
GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv pip install torchvision==0.22.1     # RECAP 价值模型用（匹配 openpi 的 torch 2.7.1）
```
显存：LoRA 微调 ≥24GB（4090），全量 ≥70GB（A100/H100 80G），推理 ≥8GB。

---

## 2. 硬件准备与自检

1. **工具坐标系**：在 DobotStudio Pro 里把 TCP 设到夹爪指尖（插头夹持中心）并设为当前工具坐标系。遥操作绕 TCP 旋转更顺手，`tcp_pose`/工作空间检查也以此为准。**采集和部署必须用同一个工具坐标系。**
2. **负载**：`robot.payload_kg / payload_center_mm` 填准，否则静止时力读数偏、碰撞检测误触发。
3. 依次自检：
```bash
python scripts/check_hardware.py --cams                # 打印序列号并截图 → 填 configs/hardware.yaml
python scripts/check_hardware.py --robot               # 打印关节角/位姿/力、校验逆解；复制 home_joints_deg
python scripts/check_hardware.py --robot --servo       # J6 ±3° 流式运动
python scripts/check_hardware.py --gripper
python scripts/check_hardware.py --robot --teleop      # 3D 鼠标方向对不对 → 改 teleop.axis_map/axis_sign
```
4. **用不通电的插座/排插**。插座底座固定牢。

---

## 3. 采集示范（阶段 1）

```bash
python scripts/record.py --task insert --operator 张三
```
`空格` 开始 → 3D 鼠标操作 → `s` 成功 / `f` 失败 / `d` 丢弃；`t` 切换插/拔任务（插完直接录拔，自然复位）。
左键夹爪开合，**按住右键精细模式**（插入最后几毫米用）。

建议：
- 插、拔各 **150~200 条**起步；插座位置、插头初始姿态、抓取位置都要有变化（每 10 条换一次布局）。
- 操作风格保持一致：先对准、再沿插入方向直推，不要来回试探太多次。
- 失败的示范也按 `f` 保存，后面 RECAP 价值模型要用。

数据在 `data/raw/<task>/<时间戳>/`（jpg + npz），是后续所有数据集的唯一来源，可随时重新转换。

---

## 4. SFT 训练与部署

```bash
# GPU 机（把 data/raw 拷过来）
cd openpi
uv run python <dobot_plug>/scripts/convert_to_lerobot.py --raw-dir <data/raw> --repo-id local/dobot_plug_sft
uv run scripts/compute_norm_stats.py --config-name pi05_dobot_plug_lora
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_dobot_plug_lora --exp-name=sft_v1 --overwrite
# 起策略服务器
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_dobot_plug_lora --policy.dir=checkpoints/pi05_dobot_plug_lora/sft_v1/29999
```
全量微调把 `_lora` 去掉（每个 config 都要单独跑一次 `compute_norm_stats.py`）。只有两台相机时，在 config 里设 `image_slots=("base_0_rgb", "left_wrist_0_rgb")`。

```bash
# 机械臂电脑
python scripts/run_policy.py --task insert --host <GPU机IP> --policy-tag sft_v1
```
`空格` 开始 → 策略控制；快出错时**推 3D 鼠标自动接管**（或按 `i`），纠正后按 `i` 交还；结束按 `s`/`f`。
每个回合都自动存为 `source=rollout`，接管帧标 `intervention=1` —— 这就是 RECAP 的训练数据。

---

## 5. RECAP 迭代（阶段 2）

```
 ┌─ run_policy.py 跑一批（每任务 50~100 回合，含失败与接管）
 │      ↓
 │  train_value.py      价值模型：预测“离成功还有多远”（失败回合额外惩罚）
 │      ↓
 │  label_advantage.py  A_t = (G_t − G_{t+N}) + V(o_{t+N}) − V(o_t)，每任务取阈值使 ~30% 为 positive；接管帧强制 positive
 │      ↓
 │  convert_to_lerobot.py --with-advantage   全部数据（示范+rollout，成功+失败）
 │      ↓
 │  train.py pi05_dobot_plug_recap(_lora)    从上一轮 checkpoint 继续，prompt 带优势指示
 │      ↓
 └─ run_policy.py --recap   （prompt 末尾带 "Advantage: positive"）
```

```bash
# GPU 机
uv run python <dobot_plug>/scripts/recap/train_value.py --raw-dir <data/raw> --out checkpoints/value/r1
uv run python <dobot_plug>/scripts/recap/label_advantage.py --raw-dir <data/raw> --value checkpoints/value/r1/best.pt
uv run python <dobot_plug>/scripts/convert_to_lerobot.py --raw-dir <data/raw> --repo-id local/dobot_plug_recap \
    --sources demo rollout --include-failures --with-advantage
uv run scripts/compute_norm_stats.py --config-name pi05_dobot_plug_recap_lora
DOBOT_RECAP_INIT=checkpoints/pi05_dobot_plug_lora/sft_v1/29999/params \
    uv run scripts/train.py pi05_dobot_plug_recap_lora --exp-name=recap_r1 --overwrite
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_dobot_plug_recap_lora --policy.dir=checkpoints/pi05_dobot_plug_recap_lora/recap_r1/9999
# 机械臂电脑
python scripts/run_policy.py --task insert --host <GPU机IP> --recap --policy-tag recap_r1
```

检查点：
- `label_advantage.py` 最后会按 任务/来源/成败 打印 positive 比例。**失败回合的 positive 比例应明显低于成功回合**，否则价值模型没学好（失败样本太少/数据太少）。
- 价值模型训练日志 `val_corr` 应 > 0.8。
- SFT 与 RECAP 用同一种变体（都 LoRA 或都全量）。
- 每轮比较同一组测试布局下的成功率（`run_policy.py` 会累计打印），只保留真正提升的 checkpoint。

---

## 6. 测试

```bash
python tests/test_offline.py               # 纯逻辑
python tests/test_with_fake_controller.py  # 官方 dobot_sdk + 假控制器（模拟 29999/30004 端口）
python tests/test_loops_sim.py             # record.py / run_policy.py 主循环端到端（假相机/鼠标/按键/策略服务器）
```

## 7. 已知需要现场确认的点

- **3D 鼠标轴向**：不同型号轴定义不同，用 `--teleop` 自检后改 `axis_map/axis_sign`。
- **夹爪**：默认按大寰 AG/PGE 的 Modbus 寄存器写的，电脑通过 USB-RS485 直连。如果夹爪走 CR5AF 末端 485 由控制器转发，或是其他品牌，在 `gripper.py` 里按同样接口加一个类。
- **ServoJ 参数**：`servo_t / aheadtime / gain` 影响跟随平滑度与滞后，先用默认值，抖动就增大 `servo_t`，滞后大就减小。
- **力阈值**：`max_force_n=45N` 是保守值，插紧的国标插头可能需要 30~60N，按实际插拔力调整；插入方向的阈值宁小勿大。
- 没有在真实 CR5AF 上运行过：控制器通信、ServoJ、逆解、反馈解析已用官方 SDK 对着模拟控制器验证；openpi 侧的变换用桩模块验证、补丁可在干净的 openpi 上 apply，但没在 GPU 上实际训练。第一次上真机请降低速度上限（`max_joint_vel_deg_s`、`max_lin_vel_mm_s`），手放在急停上。
