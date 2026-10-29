# 硬件准备与自检

## 1. 安装

### 机械臂电脑（接 CR5AF、相机、3D 鼠标、夹爪；Ubuntu 22.04，Python 3.10+）

```bash
sudo apt install libhidapi-dev
pip install -e ".[robot]"          # numpy/scipy/pyyaml/pillow + RealSense + 3D 鼠标 + 夹爪 + dobot_sdk
# 3D 鼠标权限（3Dconnexion 厂商 ID 256f；罗技是 046d）
echo 'KERNEL=="hidraw*", ATTRS{idVendor}=="256f", MODE="0666"' | sudo tee /etc/udev/rules.d/99-spacemouse.rules
sudo udevadm control --reload && sudo udevadm trigger
# openpi 客户端（websocket 连接 GPU 机策略服务器），从 openpi 仓库本地安装：
pip install -e <openpi>/packages/openpi-client
```

网卡设到 `192.168.5.x`，CR5AF 默认 `192.168.5.1`。示教器 / DobotStudio Pro 里切到 **TCP/IP 二次开发模式**。

### GPU 训练机（Ubuntu 22.04 + NVIDIA）

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git && cd openpi
git checkout $(cat <dobot_plug>/openpi_patch/BASE_COMMIT)   # 补丁基于此 commit（= 官方 main tip，已复核）
git apply <dobot_plug>/openpi_patch/openpi_dobot.patch
GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv pip install torchvision==0.22.1     # RECAP 价值模型用（匹配 openpi 的 torch 2.7.1）
pip install -e <dobot_plug>            # 基础依赖即可（不需要 [robot]）
```

显存：LoRA 微调 ≥24GB（4090），全量 ≥70GB（A100/H100 80G），推理 ≥8GB。
注意 openpi 官方对 DROID 的经验：**LoRA 效果通常不如全量微调**（"we have experimented with LoRA… haven't found the policies to perform well so far"），显存够优先全量。

## 2. 硬件准备

1. **工具坐标系**：在 DobotStudio Pro 里把 TCP 设到夹爪指尖（插头夹持中心）并设为当前工具坐标系。遥操作绕 TCP 旋转更顺手，`tcp_pose`/工作空间检查也以此为准。**采集和部署必须用同一个工具坐标系。**
2. **负载**：`robot.payload_kg / payload_center_mm` 填准，否则静止时力读数偏、碰撞检测误触发。
3. **相机**：2~3 台 D435i，尽量分到不同 USB3 控制器；镜头覆盖插座与插头，操作员的手和鼠标线不要入画。视角建议：1 台全局（看插座）+ 1 台腕部 + 可选 1 台侧视。
4. **用不通电的插座/排插**。插座底座固定牢。

## 3. 自检（依次通过再采数据）

```bash
dobot-check-hardware --cams                # 打印序列号并截图 → 填 configs/hardware.yaml
dobot-check-hardware --robot               # 打印关节角/位姿/力、校验逆解；复制 home_joints_deg
dobot-check-hardware --robot --servo       # J6 ±3° 流式运动
dobot-check-hardware --gripper
dobot-check-hardware --robot --teleop      # 3D 鼠标方向对不对 → 改 teleop.axis_map/axis_sign
```

（等价旧写法：`python scripts/check_hardware.py --robot --servo`。）

## 4. 已知需要现场确认的点

- **3D 鼠标轴向**：不同型号轴定义不同，用 `--teleop` 自检后改 `axis_map/axis_sign`。
- **夹爪**：默认按大寰 AG/PGE 的 Modbus 寄存器写的，电脑通过 USB-RS485 直连。如果夹爪走 CR5AF 末端 485 由控制器转发，或是其他品牌，在 `dobot_plug/gripper.py` 里按同样接口加一个类。
- **ServoJ 参数**：`servo_t / aheadtime / gain` 影响跟随平滑度与滞后，先用默认值，抖动就增大 `servo_t`，滞后大就减小。
- **力阈值**：`max_force_n=45N` 是保守值，插紧的国标插头可能需要 30~60N，按实际插拔力调整；插入方向的阈值宁小勿大。
- 第一次上真机请降低速度上限（`max_joint_vel_deg_s`、`max_lin_vel_mm_s`），手放在急停上。
