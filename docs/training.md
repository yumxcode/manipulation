# SFT 训练与部署（π0.5）

## 1. 归一化统计 → 训练

```bash
# GPU 机（openpi 仓库里；data/raw 已同步过来）
uv run python ~/dobot_plug/scripts/convert_to_lerobot.py --raw-dir ~/dobot_plug/data/raw --repo-id local/dobot_plug_sft
uv run scripts/compute_norm_stats.py --config-name pi05_dobot_plug_lora
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_dobot_plug_lora --exp-name=sft_v1 --overwrite
```

补丁注册的 4 个配置（`openpi_patch/openpi_dobot.patch`，基于官方 main=BASE_COMMIT，已复核可干净 apply）：

| 配置 | 说明 |
|---|---|
| `pi05_dobot_plug` | SFT 全量微调（≥70GB 显存） |
| `pi05_dobot_plug_lora` | SFT LoRA（单卡 24GB；显存够优先全量，见下） |
| `pi05_dobot_plug_recap` | RECAP 优势条件化全量（`DOBOT_RECAP_INIT` 指定初始化权重） |
| `pi05_dobot_plug_recap_lora` | RECAP LoRA |

注意：
- **SFT 与 RECAP 用同一种变体**（都 LoRA 或都全量）。
- 只有 2 台相机时，在 config 里设 `image_slots=("base_0_rgb", "left_wrist_0_rgb")`（空槽自动补零 + mask）。
- `action_horizon=15`（≈1 秒 @15Hz），与部署端 `run_policy.py` 默认 `--replan-steps 8` 对应。
- π0.5 自动使用分位数归一化（q01/q99）；如果训练发散，先查 `assets/<repo_id>/norm_stats.json`
  里是否有维度的 q01/q99 极小（openpi 官方排障建议）。

## 2. 起策略服务器（GPU 机）

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_dobot_plug_lora --policy.dir=checkpoints/pi05_dobot_plug_lora/sft_v1/29999
```

## 3. 部署 + 人工接管（机械臂电脑）

```bash
dobot-run-policy --task insert --host <GPU机IP> --policy-tag sft_v1
```

`空格` 开始 → 策略控制；快出错时**推 3D 鼠标自动接管**（或按 `i`），纠正后按 `i` 交还；结束按 `s`/`f`。
每个回合自动存为 `source=rollout`，接管帧标 `intervention=1`——RECAP 的训练数据。

按键：`空格` 开始｜`i` 接管/交还｜`p` 暂停｜`s`/`f` 成功/失败｜`d` 丢弃｜`t` 切任务｜`h` 回零｜`c` 清保护停｜`q` 退出。

## 4. checkpoint 选择

- 用 [评估协议](evaluation.md) 在同一组固定布局上比较 checkpoint，**只保留真正提升的**；
- 部署期监控：`infer` 延迟（应 <200ms 量级）、干预率随 RECAP 轮次应下降（HIL-SERL 惯例指标）。
