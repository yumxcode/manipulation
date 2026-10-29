# 贡献指南

## 开发环境

```bash
pip install -e ".[dev]"        # 基础依赖 + pytest/ruff
pip install -e ".[robot]"      # 机械臂电脑全量（RealSense/3D鼠标/夹爪/dobot_sdk）
```

## 提交前

```bash
ruff check .                                   # lint（CI 强制）
pytest tests/test_offline.py tests/test_inspect_and_eval.py   # 纯逻辑测试（无硬件）
python tests/test_loops_sim.py                 # 主循环端到端（假控制器/相机/策略服务器，需 dobot_sdk）
python tests/test_with_fake_controller.py      # 官方 SDK + 假控制器联调
```

## 约定

- **可执行入口**：新增工具放 `dobot_plug/apps/`（带 `main()`），在 `pyproject.toml` 注册 console script，
  并在 `scripts/` 放同名薄壳（保持 `python scripts/xxx.py` 用法）。
- **state/action 定义只改 `dobot_plug/features.py` 一处**：训练（openpi 补丁里的维度）与部署共用它，
  改动必须同步核对 `openpi_patch/` 的 ACTION_DIM、delta mask。
- **数据格式**：`data/raw/` 是唯一真值，转换参数只进 `convert_to_lerobot.py`；不改已落盘的原始回合。
- **openpi 补丁**：`openpi_patch/BASE_COMMIT` 必须指向验证过可干净 apply 的 commit；上游 openpi
  若更新（尤其 lerobot pin 换新布局），需要重新生成补丁并更新本文档与 README。
- **RECAP 算法改动**：先对照 `docs/recap.md` 的论文对照表，标注是"对齐论文"还是"有意偏差"。
- **文档**：改了流程就要改对应 `docs/*.md`，README 只保留概览与快速开始。
