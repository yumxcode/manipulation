# 正式评估协议

小样本成功率直接比较会自欺（10 回合里 9/10 的 Wilson 95% CI 约 [60%, 98%]）。
本协议对齐机器人学习通行做法：**固定布局集合 × N 回合 + Wilson 95% 置信区间**
（π0 论文每任务 10 回合；ALOHA 25 回合；Dream Machines 真机实证 40 固定布局各 1 回合）。

## 1. 协议

1. **固定布局清单**：把插座位置/朝向、插头初始位姿、抓取点写成清单（复制 `configs/layouts.example.yaml`
   为 `configs/layouts.yaml`），建议 10~20 个布局，覆盖居中/边缘/旋转/垫高。
   **所有 checkpoint 用同一份清单、同一顺序**，结果才可比较。清单进 git。
2. **回合数**：每轮评估 **20 回合起步**（n=20 时 1σ≈±11pp），重要结论（如保留哪个 checkpoint）≥40 回合
   或重复 3 轮。不建议超过 40（边际信息量低，费机器人时间）。
3. **纯自主执行**：评估中**不允许人工接管**（`dobot-eval` 也不提供接管）。回合间可任意复位场景。
4. **判定**：`s`=成功（插到底/拔出），`f`=失败，`a`=**无效**（布局摆错、外因中断等非策略因素；不计入分母，回合不落盘）。
5. **报告**：成功率 + Wilson 95% CI + 无效数；报告与逐回合记录写 `evals/<tag>_<时间戳>/report.json`。

## 2. 运行

```bash
# GPU 机起被评估的策略服务器（见 training.md），然后：
dobot-eval --task insert --episodes 20 --host <GPU机IP> \
    --layouts-file configs/layouts.yaml --policy-tag sft_v1
# RECAP 策略加 --recap
dobot-eval --task insert --episodes 20 --host <GPU机IP> --recap \
    --layouts-file configs/layouts.yaml --policy-tag recap_r1
```

回合照常存入 `data/raw`（meta 带 `eval=1` 与布局描述），可并入 RECAP 数据。

## 3. 比较两个 checkpoint

```bash
dobot-inspect-data --sources rollout --json -    # 日常数据体检
python - <<'EOF'
import json, glob
for p in sorted(glob.glob("evals/*/report.json")):
    r = json.load(open(p))
    s = r["summary"]["overall"]
    print(f"{r['policy_tag'] or '-':<12} {s['success']}/{s['valid']} = {s['success_rate']:.0%}  "
          f"CI [{s['wilson95'][0]:.0%}, {s['wilson95'][1]:.0%}]  无效 {s['invalid']}  ({r['created']})")
EOF
```

判读规则：两个 checkpoint 的 Wilson 95% CI **不重叠**才算显著差异；CI 重叠时回合数不够，加测，
不要凭直觉选。RECAP 迭代"只保留真正提升的 checkpoint"以此为准。

## 4. 其它指标（自动记录，辅助判读）

- 回合帧数（`mean_frames`）：同样成功率下更快 = 论文强调的 throughput 提升；
- 干预率（rollout 数据）：`dobot-run-policy` 阶段的干预率应随 RECAP 轮次下降（HIL-SERL 惯例）；
- `dobot-inspect-data` 的实测帧率：排除采集/部署节拍问题后再谈成功率差异。
